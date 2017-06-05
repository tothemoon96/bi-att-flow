import random

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

from basic.read_data import DataSet
from my.tensorflow import get_initializer
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell


def get_multi_gpu_models(config):
    models = []
    for gpu_idx in range(config.num_gpus):
        # 每一块显卡上都创建一份Model
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, \
                tf.device("/{}:{}".format(config.device_type, gpu_idx)):
            if gpu_idx > 0:
                # 在其它gpu上创建模型时重用变量
                tf.get_variable_scope().reuse_variables()
            # 在第一次创建模型时打开移动平均
            model = Model(config, scope, rep=gpu_idx == 0)
            models.append(model)

    return models


class Model(object):
    def __init__(self, config, scope, rep=True):
        '''
        :param config:
        :param scope:
        :param rep: 是否进行移动平均
        '''
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable(
            'global_step',
            shape=[],
            dtype='int32',
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        # Define forward inputs here
        # N:batch的大小
        # M:每一段最多有多少个句子
        # JX:每一句最长有多少个词
        # JQ:每个问题最多包含多少个词
        # VW:整个词表的大小
        # VC:字符表的大小
        # W:每个词最多包含多少个单词
        N, M, JX, JQ, VW, VC, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
        # 文章：[第几个样本，第几个句子，第几个词]，文章词的token表示
        self.x = tf.placeholder('int32', [N, None, None], name='x')
        # 文章：[第几个样本，第几个句子，第几个词，第几个单词]，文章单词的token表示
        self.cx = tf.placeholder('int32', [N, None, None, W], name='cx')
        # 文章：[第几个样本，第几个句子，第几个词]，文章中哪些词转化成了token
        self.x_mask = tf.placeholder('bool', [N, None, None], name='x_mask')
        # 问题：[第几个样本，第几个词]，问题词的token表示
        self.q = tf.placeholder('int32', [N, None], name='q')
        # 问题：[第几个样本，第几个词，第几个单词]，问题单词的token表示
        self.cq = tf.placeholder('int32', [N, None, W], name='cq')
        # 问题：[第几个样本，第几个词]，问题中哪些词转化成了token
        self.q_mask = tf.placeholder('bool', [N, None], name='q_mask')
        # 答案：[第几个样本，第几个词，第几个词]，起始单词为True
        self.y = tf.placeholder('bool', [N, None, None], name='y')
        # 答案：[第几个样本，第几个词，第几个词]，终止单词为True
        self.y2 = tf.placeholder('bool', [N, None, None], name='y2')
        # 答案：[第几个样本，第几个词，第几个词]，整个答案跨越的词的span为True
        self.wy = tf.placeholder('bool', [N, None, None], name='wy')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        # config.word_emb_size词向量的维度
        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')
        # [第几个样本]，某个数据点是否无效
        self.na = tf.placeholder('bool', [N], name='na')

        # Define misc
        self.tensor_dict = {}

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None
        self.var_list = None
        self.na_prob = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()
        self.var_ema = None
        # todo:这下面的两个移动平均不清楚有什么用
        if rep:
            self._build_var_ema()
        if config.mode == 'train':
            self._build_ema()

        # todo:没搞懂这里为什么要merge两次
        self.summary = tf.summary.merge_all()
        self.summary = tf.summary.merge(
            # todo:这样能够得到summaries吗
            tf.get_collection(
                "summaries",
                scope=self.scope
            )
        )

    def _build_forward(self):
        '''
        构建整个网络
        :return:
        '''
        config = self.config
        # N:batch的大小
        # M:每一段最多有多少个句子
        # JX:每一句最长有多少个词
        # JQ:每个问题最多包含多少个词
        # VW:整个词表的大小
        # VC:字符表的大小
        # d:隐含层单元数目
        # W:每个词最多包含多少个单词
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.max_word_size
        # JX:每一句最长有多少个词
        JX = tf.shape(self.x)[2]
        # JQ:每个问题最多包含多少个词
        JQ = tf.shape(self.q)[1]
        # M:每一段最多有多少个句子
        M = tf.shape(self.x)[1]
        # dc:单词的embedding的维度
        # dw:词的embedding的维度
        # dco:单词embedding之后输出的维度
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

        with tf.variable_scope("emb"):
            # 1.Character Embedding Layer
            if config.use_char_emb:
                # 在cpu上创建变量
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    # VC:字符表的大小
                    # dc:单词的embedding的维度
                    # todo:没有指定初始化策略
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

                with tf.variable_scope("char"):
                    # [N, M, JX, W, dc]
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)
                    # [N, JQ, W, dc]
                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)
                    # 不分句，把句子维度压缩
                    # Acx:[N*M,JX,W,dc]
                    Acx = tf.reshape(Acx, [-1, JX, W, dc])
                    Acq = tf.reshape(Acq, [-1, JQ, W, dc])

                    # filter的个数
                    # config.out_channel_dims中每个元素加起来的和要等于config.char_out_size
                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    # filter的大小（卷积核的大小）
                    heights = list(map(int, config.filter_heights.split(',')))

                    # dco:单词embedding之后输出的维度
                    assert sum(filter_sizes) == dco, (filter_sizes, dco)
                    with tf.variable_scope("conv"):
                        # 对文档使用char embedding计算word_embedding
                        xx = multi_conv1d(
                            Acx,
                            filter_sizes,
                            heights,
                            "VALID",
                            self.is_train,
                            config.keep_prob,
                            scope="xx"
                        )
                        # 使用文章的char embedding的参数进行问题的char embedding
                        if config.share_cnn_weights:
                            # 第一次创建模型时，其父scope没有reuse，下面进行reuse说明是reuse了上面的变量
                            # 在之后创建模型时，其父scope已经reuse了，下面默认就reuse了
                            tf.get_variable_scope().reuse_variables()
                            # 对问题使用char embedding计算word_embedding
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
                        else:
                            # 对问题使用char embedding计算word_embedding
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")
                        # [N,M,JX,filter_size*多少种卷积核（dco）]
                        xx = tf.reshape(xx, [-1, M, JX, dco])
                        # [N,JQ,filter_size*多少种卷积核（dco）]
                        qq = tf.reshape(qq, [-1, JQ, dco])

            # 2.Word Embedding Layer
            if config.use_word_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        # [VW,dw]
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
                    if config.use_glove_for_unk:
                        # [VW+new_emb_mat中词的数目,dw]
                        word_emb_mat = tf.concat(axis=0, values=[word_emb_mat, self.new_emb_mat])

                with tf.name_scope("word"):
                    # [N,M,JX,dw]
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)
                    # [N,JQ,dw]
                    Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)
                    # [N,M,JX,dw]
                    # 问题的word embedding
                    self.tensor_dict['x'] = Ax
                    # [N,JQ,dw]
                    # 答案的word embedding
                    self.tensor_dict['q'] = Aq
                if config.use_char_emb:
                    # [N,M,JX,dw+dco]
                    xx = tf.concat(axis=3, values=[xx, Ax])
                    # [N,JQ,dw+dco]
                    qq = tf.concat(axis=2, values=[qq, Aq])
                else:
                    # [N,M,JX,dw]
                    xx = Ax
                    # [N,JQ,dw]
                    qq = Aq

        # Highway Network
        if config.highway:
            # config.highway_num_layers个high way线性层，不改变shape
            with tf.variable_scope("highway"):
                xx = highway_network(
                    xx,
                    config.highway_num_layers,
                    True,
                    wd=config.wd,
                    is_train=self.is_train
                )
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(
                    qq,
                    config.highway_num_layers,
                    True,
                    wd=config.wd,
                    is_train=self.is_train
                )

        # 问题对后续网络层的输入
        self.tensor_dict['xx'] = xx
        # 答案对后续网络层的输入
        self.tensor_dict['qq'] = qq

        # d:隐含层单元数目
        # 构造双向RNN的基本单元
        # fw是前向，bw是后向
        cell_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell_fw = SwitchableDropoutWrapper(
            cell_fw,
            self.is_train,
            input_keep_prob=config.input_keep_prob
        )
        d_cell_bw = SwitchableDropoutWrapper(
            cell_bw,
            self.is_train,
            input_keep_prob=config.input_keep_prob
        )

        cell2_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell2_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell2_fw = SwitchableDropoutWrapper(
            cell2_fw,
            self.is_train,
            input_keep_prob=config.input_keep_prob
        )
        d_cell2_bw = SwitchableDropoutWrapper(
            cell2_bw,
            self.is_train,
            input_keep_prob=config.input_keep_prob
        )

        cell3_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell3_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell3_fw = SwitchableDropoutWrapper(
            cell3_fw,
            self.is_train,
            input_keep_prob=config.input_keep_prob
        )
        d_cell3_bw = SwitchableDropoutWrapper(
            cell3_bw,
            self.is_train,
            input_keep_prob=config.input_keep_prob
        )

        cell4_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell4_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell4_fw = SwitchableDropoutWrapper(
            cell4_fw,
            self.is_train,
            input_keep_prob=config.input_keep_prob
        )
        d_cell4_bw = SwitchableDropoutWrapper(
            cell4_bw,
            self.is_train,
            input_keep_prob=config.input_keep_prob
        )

        # [N,M]，某一个句子有多长
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)
        # [N]，某一个问题有多长
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)

        # 3.Contextual Embedding Layer
        with tf.variable_scope("prepro"):
            # qq:[N,JQ,dw+dco]，如果使用char_embedding和word_embedding
            # ([N,JQ,d],[N,JQ,d])
            # 下面的有Dropout
            (fw_u, bw_u), \
            (
                (_, fw_u_f),# 前向的最终状态的hidden:[N,d]
                (_, bw_u_f) # 后向的最终状态的hidden:[N,d]
            ) = bidirectional_dynamic_rnn(
                d_cell_fw, d_cell_bw, qq, q_len, dtype='float', scope='u1'
            )
            # u:[N,JQ,2d]，将前向和后向的隐含层拼接起来
            u = tf.concat(axis=2, values=[fw_u, bw_u])
            # 复用问题的Contextual Embedding的权重对文章编码
            # 下面的没有Dropout
            if config.share_lstm_weights:
                # 父variable scope reuse了，子variable scope也会reuse
                tf.get_variable_scope().reuse_variables()
                # xx:[N,M,JX,dw+dco]，如果使用char_embedding和word_embedding
                # fw_h,bw_h:[N,M,JX,d]
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, xx, x_len, dtype='float', scope='u1'
                )
                # h:[N,M,JX,2d]
                h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
            # 不复用问题的Contextual Embedding的权重对文章编码
            else:
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, xx, x_len, dtype='float', scope='h1'
                )
                # h:[N,M,JX,2d]
                h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
            # 问题中每个词的编码，[N,JQ,2d]
            self.tensor_dict['u'] = u
            # 文章中每个词的编码，[N,M,JX,2d]
            self.tensor_dict['h'] = h

        # 4.Attention Flow Layer
        with tf.variable_scope("main"):
            # 这部分出现在论文中的Ablations中
            if config.dynamic_att:
                p0 = h
                # u:[N*M,JQ,2d]，每个样本点中问题的编码重复了M遍
                u = tf.reshape(
                    # [N,M,JQ,2d]
                    tf.tile(
                        tf.expand_dims(u, 1),# [N,1,JQ,2d]
                        [1, M, 1, 1]
                    ),
                    [N * M, JQ, 2 * d]
                )
                # q_mask:[N,M,JQ]，每个样本点中问题的有效词mask重复了M遍
                q_mask = tf.reshape(
                    # [N,M,JQ]
                    tf.tile(
                        tf.expand_dims(self.q_mask, 1),#[N,1,JQ]
                        [1, M, 1]
                    ),
                    [N * M, JQ]
                )
                first_cell_fw = AttentionCell(
                    cell2_fw,
                    u,
                    mask=q_mask,
                    mapper='sim',
                    input_keep_prob=self.config.input_keep_prob,
                    is_train=self.is_train
                )
                first_cell_bw = AttentionCell(
                    cell2_bw,
                    u,
                    mask=q_mask,
                    mapper='sim',
                    input_keep_prob=self.config.input_keep_prob,
                    is_train=self.is_train
                )
                second_cell_fw = AttentionCell(
                    cell3_fw,
                    u,
                    mask=q_mask,
                    mapper='sim',
                    input_keep_prob=self.config.input_keep_prob,
                    is_train=self.is_train
                )
                second_cell_bw = AttentionCell(
                    cell3_bw,
                    u,
                    mask=q_mask,
                    mapper='sim',
                    input_keep_prob=self.config.input_keep_prob,
                    is_train=self.is_train
                )
            else:
                # 论文中的\beta(h,\tilde{u},\tilde{h})，[N,M,JX,8d]
                p0 = attention_layer(
                    config,
                    self.is_train,
                    h,# [N,M,JX,2d]
                    u,# [N,JQ,2d]
                    h_mask=self.x_mask,# [N,M,JX]
                    u_mask=self.q_mask,# [N,JQ]
                    scope="p0",
                    tensor_dict=self.tensor_dict
                )
                first_cell_fw = d_cell2_fw
                second_cell_fw = d_cell3_fw
                first_cell_bw = d_cell2_bw
                second_cell_bw = d_cell3_bw

            # 5.Modeling Layer
            # Match-LSTM，每一层的输入和输出带Dropout
            # ([N, M, JX, d],[N, M, JX, d])
            (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(
                first_cell_fw,
                first_cell_bw,
                p0, # [N,M,JX,nd]
                x_len, # [N,M]，某一个句子有多长
                dtype='float',
                scope='g0'
            )
            # g0:[N, M, JX, 2d]
            g0 = tf.concat(axis=3, values=[fw_g0, bw_g0])
            (fw_g1, bw_g1), _ = bidirectional_dynamic_rnn(
                second_cell_fw,
                second_cell_bw,
                g0,
                x_len,
                dtype='float',
                scope='g1'
            )
            # g1:[N, M, JX, 2d]
            # 论文中的M
            g1 = tf.concat(axis=3, values=[fw_g1, bw_g1])

            # 6.Output Layer
            # 输入有Dropout
            # 在默认的配置下，config.answer_func=‘linear’
            # logits_t = w^T_{(p^1)} [M_{:t};G_{:t}] + b_{(p^1)}
            # logits:[N, M, JX]
            # 对文章中的每个词分配未归一化的Attention
            logits = get_logits(
                [g1, p0], # [N, M, JX, 10d]
                d,
                True,
                wd=config.wd,
                input_keep_prob=config.input_keep_prob,
                mask=self.x_mask,
                is_train=self.is_train,
                func=config.answer_func, # linear
                scope='logits1'
            )
            # 根据logits计算整个文章的表达
            # a1i:[N,2d]
            a1i = softsel(
                tf.reshape(
                    g1,
                    [N, M * JX, 2 * d]
                ),
                tf.reshape(
                    logits,
                    [N, M * JX]
                )
            )
            # a1i:[N,M,JX,2d]复制M*JX份
            a1i = tf.tile(
                tf.expand_dims(
                    tf.expand_dims(a1i, 1),
                    1
                ),
                [1, M, JX, 1]
            )
            # ([N, M, JX, d],[N, M, JX, d])
            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(
                d_cell4_fw,
                d_cell4_bw,
                tf.concat(
                    axis=3,
                    values=[p0, g1, a1i, g1 * a1i]
                ),# [N,M,JX,14d],[G_{:t};M_{:t};a1i;M_{:t} \odot a1i]
                x_len,
                dtype='float',
                scope='g2'
            )
            # [N, M, JX, 2d]
            # 论文中的M^2
            g2 = tf.concat(
                axis=3,
                values=[fw_g2, bw_g2]
            )
            # logits2_t = w^T_{(p^2)} [M^2_{:t};G_{:t}] + b_{(p^2)}
            # 在默认的配置下，config.answer_func=‘linear’
            # 输入有Dropout
            # logits2:[N,M,JX]
            logits2 = get_logits(
                [g2, p0],
                d,
                True,
                wd=config.wd,
                input_keep_prob=config.input_keep_prob,
                mask=self.x_mask,
                is_train=self.is_train,
                func=config.answer_func,# ‘linear’
                scope='logits2'
            )

            # [N,M*JX]，对文章中的每个词分配Attention
            flat_logits = tf.reshape(logits, [-1, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)
            # [N,M*JX]，对文章中的每个词分配Attention
            flat_logits2 = tf.reshape(logits2, [-1, M * JX])
            flat_yp2 = tf.nn.softmax(flat_logits2)

            # 允许没有答案存在
            if config.na:
                # 添加一个表示na的标量偏置
                na_bias = tf.get_variable("na_bias", shape=[], dtype='float')
                # na_bias_tiled:[N, 1]复制了N份
                na_bias_tiled = tf.tile(
                    tf.reshape(na_bias, [1, 1]),
                    [N, 1]
                )
                # concat_flat_logits:[N,1+M*JX]
                concat_flat_logits = tf.concat(
                    axis=1,
                    values=[na_bias_tiled, flat_logits]
                )
                # concat_flat_logits:[N, 1 + M * JX]
                concat_flat_yp = tf.nn.softmax(concat_flat_logits)
                # na_prob:[N]，起始指针选择没有答案的概率
                na_prob = tf.squeeze(
                    tf.slice(
                        concat_flat_yp,
                        [0, 0],
                        [-1, 1]
                    ),
                    [1]
                )
                # flat_yp:[N,M*JX]，选择文中某个词作为答案起始位置的概率
                flat_yp = tf.slice(
                    concat_flat_yp,
                    [0, 1],
                    [-1, -1]
                )

                concat_flat_logits2 = tf.concat(
                    axis=1,
                    values=[na_bias_tiled, flat_logits2]
                )
                concat_flat_yp2 = tf.nn.softmax(concat_flat_logits2)
                # na_prob2:[N]，结束指针选择没有答案的概率
                na_prob2 = tf.squeeze(
                    tf.slice(
                        concat_flat_yp2,
                        [0, 0],
                        [-1, 1]
                    ),
                    [1]
                )
                # flat_yp2:[N,M*JX]，选择文中某个词作为答案结束位置的概率
                flat_yp2 = tf.slice(
                    concat_flat_yp2,
                    [0, 1],
                    [-1, -1]
                )

                self.concat_logits = concat_flat_logits
                self.concat_logits2 = concat_flat_logits2
                self.na_prob = na_prob * na_prob2

            # yp:[N,M,JX]
            yp = tf.reshape(flat_yp, [-1, M, JX])
            # yp2:[N,M,JX]
            yp2 = tf.reshape(flat_yp2, [-1, M, JX])
            # wyp:[N,M,JX]
            wyp = tf.nn.sigmoid(logits2)

            self.tensor_dict['g1'] = g1
            self.tensor_dict['g2'] = g2

            self.logits = flat_logits
            self.logits2 = flat_logits2
            self.yp = yp
            self.yp2 = yp2
            self.wyp = wyp

    def _build_loss(self):
        config = self.config
        JX = tf.shape(self.x)[2]
        M = tf.shape(self.x)[1]
        JQ = tf.shape(self.q)[1]

        # [N]，表示某个问题是否有效，有效是1，无效是0
        loss_mask = tf.reduce_max(
            tf.cast(self.q_mask, 'float'),
            1
        )

        if config.wy:
            # [N,M,JX]，注意这里只使用了self.logits2
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.reshape(self.logits2, [-1, M, JX]),
                labels=tf.cast(self.wy, 'float')
            )
            # 整个数据集中文章里有多少个词是答案
            num_pos = tf.reduce_sum(
                tf.cast(self.wy, 'float')
            )
            # 整个数据集中文章里有多少个词不是答案
            num_neg = tf.reduce_sum(
                tf.cast(self.x_mask, 'float')
            ) - num_pos
            # 比例
            damp_ratio = num_pos / num_neg
            # todo:不懂为什么要这样去damp，而且只用了logits2
            dampened_losses = losses * (
                # [N,M,JX],这些是不是答案的词，值为1，乘上了damp_ratio，把所有这些不是答案的词的值加起来等于num_pos
                (
                    tf.cast(self.x_mask, 'float') -
                    tf.cast(self.wy, 'float')
                ) * damp_ratio +
                tf.cast(self.wy, 'float')
            )
            # [N]
            new_losses = tf.reduce_sum(dampened_losses, [1, 2])
            # 过滤掉无效的loss
            ce_loss = tf.reduce_mean(loss_mask * new_losses)
            """
            if config.na:
                na = tf.reshape(self.na, [-1, 1])
                concat_y = tf.concat(1, [na, tf.reshape(self.wy, [-1, M * JX])])
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    self.concat_logits, tf.cast(concat_y, 'float') / tf.reduce_sum(tf.cast(self.wy, 'float')))
            else:
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    self.logits2, tf.cast(tf.reshape(self.wy, [-1, M * JX]), 'float') / tf.reduce_sum(tf.cast(self.wy, 'float')))
            ce_loss = tf.reduce_mean(loss_mask * losses)
            """
            tf.add_to_collection('losses', ce_loss)

        else:
            if config.na:
                # [N,1]
                na = tf.reshape(self.na, [-1, 1])
                # [N,1+M*JX]

                concat_y = tf.concat(
                    axis=1,
                    values=[
                        na,
                        tf.reshape(self.y, [-1, M * JX])
                    ]
                )
                # 答案起始位置loss
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.concat_logits,
                    labels=tf.cast(concat_y, 'float')
                )

                concat_y2 = tf.concat(
                    axis=1,
                    values=[
                        na,
                        tf.reshape(self.y2, [-1, M * JX])
                    ]
                )
                # 答案终止位置loss
                losses2 = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.concat_logits2,
                    labels=tf.cast(concat_y2, 'float')
                )
            else:
                # 答案起始位置loss
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=tf.cast(
                        tf.reshape(self.y, [-1, M * JX]),
                        'float'
                    )
                )
                # 答案终止位置loss
                losses2 = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits2,
                    labels=tf.cast(
                        tf.reshape(self.y2, [-1, M * JX]),
                        'float')
                )
            # 过滤掉无效的loss
            ce_loss = tf.reduce_mean(loss_mask * losses)
            ce_loss2 = tf.reduce_mean(loss_mask * losses2)
            # 添加到losses字典
            tf.add_to_collection('losses', ce_loss)
            tf.add_to_collection("losses", ce_loss2)

        # 把这么多losses全部加起来
        self.loss = tf.add_n(
            tf.get_collection('losses', scope=self.scope),
            name='loss'
        )
        # 得到loss的summary
        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _build_ema(self):
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        # todo:这个collection之前没有创建，会存在吗？
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
        ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        # loss必须要在上面的计算完了以后再计算
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def _build_var_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
        ema = self.var_ema
        ema_op = ema.apply(tf.trainable_variables())
        # loss必须要在上面的计算完了以后再计算
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        '''
        这里可以看到placeholder和batch之间的对应关系
        :param batch:
        :param is_train:
        :param supervised:
        :return:
        '''
        assert isinstance(batch, DataSet)
        config = self.config
        # N:batch的大小
        # M:每一段最多有多少个句子
        # JX:每一句最长有多少个词
        # JQ:每个问题最多包含多少个词
        # VW:整个词表的大小
        # VC:字符表的大小
        # d:隐含层单元数目
        # W:每个词最多包含多少个单词
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, \
            config.hidden_size, config.max_word_size
        feed_dict = {}

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            # 计算这个batch中每一句最长有多少个词
            if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for para in batch.data['x'] for sent in para)
            JX = min(JX, new_JX)

            # 计算这个batch中每个问题最多包含多少个词
            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_JQ = 1
            else:
                new_JQ = max(len(ques) for ques in batch.data['q'])
            JQ = min(JQ, new_JQ)

        if config.cpu_opt:
            # 计算这个batch中每一段最多有多少个句子
            if sum(len(para) for para in batch.data['x']) == 0:
                new_M = 1
            else:
                new_M = max(len(para) for para in batch.data['x'])
            M = min(M, new_M)

        # 词级别的输入
        x = np.zeros([N, M, JX], dtype='int32')
        # 单词级别的输入
        cx = np.zeros([N, M, JX, W], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        feed_dict[self.cq] = cq
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.is_train] = is_train
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        X = batch.data['x']
        CX = batch.data['cx']

        if supervised:
            y = np.zeros([N, M, JX], dtype='bool')
            y2 = np.zeros([N, M, JX], dtype='bool')
            wy = np.zeros([N, M, JX], dtype='bool')
            na = np.zeros([N], dtype='bool')
            feed_dict[self.y] = y
            feed_dict[self.y2] = y2
            feed_dict[self.wy] = wy
            feed_dict[self.na] = na

            # i表示的是一个batch中的第几个样本
            # 循环的功能是想numpy数组填充内容
            for i, (xi, cxi, yi, nai) in enumerate(
                    zip(X, CX, batch.data['y'], batch.data['na'])
            ):
                # 某个问题没有答案的话
                if nai:
                    na[i] = nai
                    continue
                # 从几个备选答案中随机选择一个答案，得到其标志答案在文章哪个位置的索引
                start_idx, stop_idx = random.choice(yi)
                # 起始词：句号，词号
                j, k = start_idx
                # 终止词：句号，词号
                j2, k2 = stop_idx
                if config.single:
                    X[i] = [xi[j]]
                    CX[i] = [cxi[j]]
                    j, j2 = 0, 0
                if config.squash:
                    # 计算第j个句子之间的句子包含了多少个词
                    offset = sum(map(len, xi[:j]))
                    # 修正索引
                    j, k = 0, k + offset
                    # 计算第j2个句子之间的句子包含了多少个词
                    offset = sum(map(len, xi[:j2]))
                    # 修正索引
                    j2, k2 = 0, k2 + offset
                # 第i个样本，第j个句子，第k个词
                y[i, j, k] = True
                y2[i, j2, k2-1] = True
                # 如果整个答案在同一句之中
                if j == j2:
                    wy[i, j, k:k2] = True
                # 如果整个答案不在同一句之中
                else:
                    wy[i, j, k:len(batch.data['x'][i][j])] = True
                    # todo:其实这里有点问题，应该是j，k到j2，k2之间所有的词全部都为True，但是因为没有答案跨越的句子超过2句，所以不会有bug
                    wy[i, j2, :k2] = True

        def _get_word(word):
            '''
            查找某个词的token
            :param word:
            :return:
            '''
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            # 模型的创建过程中吧word2idx_embedding和new_word2idx_embedding两个矩阵拼在了一起
            # 所以word2idx中如果没有查到某个词，可以在new_word2idx中查找并加上word2idx的长度作为偏移
            if config.use_glove_for_unk:
                d2 = batch.shared['new_word2idx']
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in d2:
                        return d2[each] + len(d)
            return 1

        def _get_char(char):
            '''
            查找某个单词的token
            :param char:
            :return:
            '''
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        for i, xi in enumerate(X):
            if self.config.squash:
                # 把几句话压缩到一句中
                xi = [list(itertools.chain(*xi))]
            # 对每个句子
            for j, xij in enumerate(xi):
                if j == config.max_num_sents:
                    break
                # 对每个词
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    # 得到词的token的表示
                    x[i, j, k] = each
                    x_mask[i, j, k] = True

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                # 对每个词
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    # 对每个单词
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        # 得到单词的token的表示
                        cx[i, j, k, l] = _get_char(cxijkl)

        for i, qi in enumerate(batch.data['q']):
            # 对问题中的每个词
            for j, qij in enumerate(qi):
                q[i, j] = _get_word(qij)
                q_mask[i, j] = True

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        if supervised:
            # 不要出现wy指示答案出现位置的词没有被tokenized的情况
            assert np.sum(~(x_mask | ~wy)) == 0

        return feed_dict


def bi_attention(
        config,
        is_train,
        h,
        u,
        h_mask=None,
        u_mask=None,
        scope=None,
        tensor_dict=None
):
    '''
    计算c2q和q2c的Attention，这里的attention计算没有Dropout
    :param config:
    :param is_train:
    :param h:文章中每个词的编码，[N,M,JX,2d]
    :param u:问题中每个词的编码，[N,JQ,2d]
    :param h_mask:[N,M,JX]
    :param u_mask:[N,JQ]
    :param scope:
    :param tensor_dict:
    :return:(u_a:[N,M,JX,2d](c2q),h_a:[N,M,JX,2d](q2c))
    '''
    with tf.variable_scope(scope or "bi_attention"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        # h_aug:[N,M,JX,JQ,2d]复制了JQ份
        h_aug = tf.tile(
            tf.expand_dims(h, 3),
            [1, 1, 1, JQ, 1]
        )
        # u_aug:[N,M,JX,JQ,2d]复制了M*JX份
        u_aug = tf.tile(
            tf.expand_dims(
                tf.expand_dims(u, 1),
                1
            ),
            [1, M, JX, 1, 1]
        )
        if h_mask is None:
            hu_mask = None
        else:
            # h_mask_aug:[N,M,JX,JQ]复制了JQ份
            h_mask_aug = tf.tile(
                tf.expand_dims(h_mask, 3),
                [1, 1, 1, JQ]
            )
            # u_mask_aug:[N,M,JX,JQ]复制了M*JX份
            u_mask_aug = tf.tile(
                tf.expand_dims(
                    tf.expand_dims(u_mask, 1),
                    1
                ),
                [1, M, JX, 1]
            )
            # hu_mask:[N,M,JX,JQ]，表明文章中每一句某个词对应的某个问题的某个词是否存在答案
            hu_mask = h_mask_aug & u_mask_aug

        # 没有Dropout
        # u_logits:[N,M,JX,JQ]
        u_logits = get_logits(
            [h_aug, u_aug],
            None,
            True,
            wd=config.wd,
            mask=hu_mask,
            is_train=is_train,
            func=config.logit_func,
            scope='u_logits'
        )
        # [N,M,JX,2d]，对文章中的每个词，得到问题的表达，对应文章中的\tilde{U}
        u_a = softsel(u_aug, u_logits)
        # [N,M,2d]，对整个问题，得到文章的表达
        # 如果这里的M不为1，那么得到结果是有问题的，这是计算对整个问题，某个句子的表达
        h_a = softsel(
            # h:[N,M,JX,2d]
            h,
            # 对文章中的每个词，得到对整个问题的最大的匹配值
            # [N,M,JX]
            tf.reduce_max(u_logits, 3)
        )
        # h_a:[N,M,JX,2d]，对应文章中的\tilde{H}
        h_a = tf.tile(
            tf.expand_dims(h_a, 2),
            [1, 1, JX, 1]
        )

        if tensor_dict is not None:
            # [N,M,JX,JQ]，文章中的某个词，对问题中的每个词的Attention，对应论文中的a_t
            a_u = tf.nn.softmax(u_logits)
            # [N,M,JX]，整个问题对文章中每个词的Attention，对应论文中的b
            a_h = tf.nn.softmax(tf.reduce_max(u_logits, 3))
            tensor_dict['a_u'] = a_u
            tensor_dict['a_h'] = a_h
            variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name
            )
            for var in variables:
                tensor_dict[var.name] = var

        return u_a, h_a


def attention_layer(
        config,
        is_train,
        h,
        u,
        h_mask=None,
        u_mask=None,
        scope=None,
        tensor_dict=None
):
    '''
    通过控制config.q2c_att、config.c2q_att、config.logit_func，进行文章中的Ablations实验
    :param config:
    :param is_train:
    :param h: 文章中每个词的编码，[N,M,JX,2d]
    :param u: 问题中每个词的编码，[N,JQ,2d]
    :param h_mask: [N,M,JX]
    :param u_mask: [N,JQ]
    :param scope:
    :param tensor_dict:
    :return:论文中的\beta(h,\tilde{u},\tilde{h})，[N,M,JX,nd]
    '''
    with tf.variable_scope(scope or "attention_layer"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        if config.q2c_att or config.c2q_att:
            u_a, h_a = bi_attention(
                config,
                is_train,
                h,
                u,
                h_mask=h_mask,
                u_mask=u_mask,
                tensor_dict=tensor_dict
            )
        if not config.c2q_att:
            # u_a:[N,M,JX,2d]，对文章中的每个词，得到问题的表达
            u_a = tf.tile(
                tf.expand_dims( # [N,1,1,2d]
                    tf.expand_dims( # [N,1,2d]
                        # \tilde{u} = \sum_j U_{:j}
                        tf.reduce_mean(u, 1), # [N,2d],对2d中每一个元素求出JQ轴上的平均
                        1
                    ),
                    1
                ),
                [1, M, JX, 1]
            )
        # \beta(h,\tilde{u},\tilde{h}) = [h;\tilde{u};h \odot \tilde{u};h \odot \tilde{h}]
        if config.q2c_att:
            # [N,M,JX,8d]
            p0 = tf.concat(
                axis=3,
                values=[h, u_a, h * u_a, h * h_a]
            )
        # \beta(h,\tilde{u},\tilde{h}) = [h;\tilde{u};h \odot \tilde{u}]
        else:
            # [N,M,JX,6d]
            p0 = tf.concat(
                axis=3,
                values=[h, u_a, h * u_a]
            )
        return p0
