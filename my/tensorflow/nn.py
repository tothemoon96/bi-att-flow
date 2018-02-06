from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.util import nest
import tensorflow as tf

from my.tensorflow import flatten, reconstruct, add_wd, exp_mask


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    '''
    经过一个线性层，没有激活函数，Dropout添加在输入层
    :param args:输入的tensor或者tensor的list，如果是个list，那个list中的tensor的shape要相同，会将这些tensor的最后一个维度拼起来作为输入进行仿射变换
    :param output_size:输出的单元大小
    :param bias:是否有bias
    :param bias_start:bias的初始值
    :param scope:变量存放的scope
    :param squeeze:是否移除掉最后一个为shape为1的轴
    :param wd:L2正则化系数
    :param input_keep_prob:输入的Dropout
    :param is_train:是否是训练状态，和Dropout有关
    :return:shape的前面几个轴和args一样，最后一个轴由output_size决定，如果squeeze为True，那么移除掉最后一个shape为1的轴
    '''
    # not args->args是空值或者None是为True
    if args is None or \
            (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    # 如果args不是一个list，那么必须要把args转换成一个list，并且args不能为空值
    # 内置的_linear只能接受这样的参数
    if not nest.is_sequence(args):
        args = [args]

    flat_args = [
        flatten(arg, 1) for arg in args
    ]

    # 输入层dropout
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [
            tf.cond(
                is_train,
                lambda: tf.nn.dropout(arg, input_keep_prob),
                lambda: arg
            )
            for arg in flat_args
        ]

    # 经过内部的线性层
    with tf.variable_scope(scope or 'Linear'):
        flat_out = _linear(flat_args, output_size, bias, bias_initializer=tf.constant_initializer(bias_start))
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        # 移除掉最后一个为shape为1的轴
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    if wd:
        add_wd(wd)

    return out


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    '''
    对x进行Dropout
    :param x: 输入的Tensor
    :param keep_prob: 保留的概率
    :param is_train:
    :param noise_shape:
    :param seed:
    :param name:
    :return:keep_prob < 1.0 and is_train时，返回dropout包装后的out，否则返回x
    '''
    with tf.name_scope(name or "dropout"):
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            # 条件转移
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x


def softmax(logits, mask=None, scope=None):
    '''
    对最后一个轴计算softmax
    :param logits:
    :param mask:Tensor，和logits相同的shape，False表示要被mask掉
    :param scope:
    :return:
    '''
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)

        return out


def softsel(target, logits, mask=None, scope=None):
    """
    J应该是指要分配attention对象的数量，d是指hidden neural的维度，功能是根据target和logits(attention)形成一个attented vector
    :param target: [ ..., J, d] dtype=float
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        # a:[ ..., J]
        # 应该是对J中每一个记忆单元分配的attention
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        # out:[...,d]，应该是根据J形成的attented vector
        out = tf.reduce_sum(
            # 元素级相乘
            # tf.expand_dims(a, -1):[ ..., J,1]，沿着最后一个轴广播，某一个值乘以所有的d个值
            # 得到:[...,J,d]
            tf.expand_dims(a, -1) * target,
            target_rank - 2
        )
        return out


def double_linear_logits(
        args,
        size,
        bias,
        bias_start=0.0,
        scope=None,
        mask=None,
        wd=0.0,
        input_keep_prob=1.0,
        is_train=None
):
    '''
    linear(tanh(linear))，和linear_logits区别在于经历了两次线性变换和一次tanh非线性变换
    :param args:
    :param size:tanh层神经元的数目
    :param bias:
    :param bias_start:
    :param scope:
    :param mask:
    :param wd:
    :param input_keep_prob:
    :param is_train:
    :return:
    '''
    with tf.variable_scope(scope or "Double_Linear_Logits"):
        first = tf.tanh(
            linear(
                args,
                size,
                bias,
                bias_start=bias_start,
                scope='first',
                wd=wd,
                input_keep_prob=input_keep_prob,
                is_train=is_train
            )
        )
        second = linear(
            first,
            1,
            bias,
            bias_start=bias_start,
            squeeze=True,
            scope='second',
            wd=wd,
            input_keep_prob=input_keep_prob,
            is_train=is_train
        )
        if mask is not None:
            second = exp_mask(second, mask)
        return second


def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    '''
    :param args: tensor或tensor的list,[...,J,d]或[[...,J,d]]*n
    :param bias:
    :param bias_start:
    :param scope:
    :param mask:
    :param wd:
    :param input_keep_prob:
    :param is_train:
    :return:[...,J]
    '''
    with tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(
            args,
            1,
            bias,
            bias_start=bias_start,
            squeeze=True,
            scope='first',
            wd=wd,
            input_keep_prob=input_keep_prob,
            is_train=is_train
        )
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def sum_logits(args, mask=None, name=None):
    '''
    args列表中将各个tensor最后一个轴上的值累加，然后将处理后的tensor累加
    :param args: [h:[N,M,JX,JQ,2d]复制了JQ份,u:[N,M,JX,JQ,2d]复制了M*JX份]
    :param mask: [N,M,JX,JQ]，表明文章中每一句某个词对应的某个问题的某个词是否存在答案
    :param name: scope
    :return:[N,M,JX,JQ]
    '''
    with tf.name_scope(name or "sum_logits"):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]
        rank = len(args[0].get_shape())
        logits = sum(
            # [N,M,JX,JQ]
            tf.reduce_sum(arg, rank-1)
            for arg in args
        )
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None, func=None):
    '''
    使用各种S_{tj}函数计算Attention Flow Layer中的未归一化Attention
    :param args:[h:[N,M,JX,JQ,2d]复制了JQ份,u:[N,M,JX,JQ,2d]复制了M*JX份]
    :param size:
    :param bias:是否有偏置
    :param bias_start:偏置的初始值
    :param scope:
    :param mask:
    :param wd:l2正则化系数
    :param input_keep_prob:
    :param is_train:是否是训练状态，和Dropout有关
    :param func:
    :return:[N,M,JX,JQ]
    '''
    if func is None:
        func = "sum"
    # \alpha(h,u) = e^T [h;u]
    if func == 'sum':
        return sum_logits(args, mask=mask, name=scope)
    # \alpha(h,u) = w^T_{(s)} [h;u] + b_{(s)}
    elif func == 'linear':
        return linear_logits(
            args,
            bias,
            bias_start=bias_start,
            scope=scope,
            mask=mask,
            wd=wd,
            input_keep_prob=input_keep_prob,
            is_train=is_train
        )
    # \alpha(h,u) = w^T_{(s)} tanh(W^T_{(s)} [h;u] + b_{(s1)}) + b_{(s2))}
    elif func == 'double':
        return double_linear_logits(
            args,
            size,
            bias,
            bias_start=bias_start,
            scope=scope,
            mask=mask,
            wd=wd,
            input_keep_prob=input_keep_prob,
            is_train=is_train
        )
    # \alpha(h,u) = e^T (h \odot u)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask, name=scope)
    # \alpha(h,u) = w^T_{(s)} (h \odot u) + b_{(s)}
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits(
            [arg],
            bias,
            bias_start=bias_start,
            scope=scope,
            mask=mask,
            wd=wd,
            input_keep_prob=input_keep_prob,
            is_train=is_train
        )
    # \alpha(h,u) = e^T (W_p h \odot u)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear(
            [args[0]],
            d,
            False,
            bias_start=bias_start,
            scope=scope,
            wd=wd,
            input_keep_prob=input_keep_prob,
            is_train=is_train
        )
        return sum_logits([proj * args[1]], mask=mask)
    # \alpha(h,u) = w^T_{(s)} [h;u;h \odot u] + b_{(s)}
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits(
            [args[0], args[1], new_arg],
            bias,
            bias_start=bias_start,
            scope=scope,
            mask=mask,
            wd=wd,
            input_keep_prob=input_keep_prob,
            is_train=is_train
        )
    else:
        raise Exception()


def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    '''
    gate使用sigmoid激活，传统网络使用relu激活
    :param arg:网络的输入
    :param bias:是否有偏置
    :param bias_start:偏置的初始值
    :param scope:variable_scope的名字
    :param wd:l2正则化系数
    :param input_keep_prob:每层highway network输入处有Dropout
    :param is_train:是否在训练，和dropout有关
    :return:输出的shape和输入的shape相同
    '''
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]
        trans = linear(
            [arg],
            d,
            bias,
            bias_start=bias_start,
            scope='trans',
            wd=wd,
            input_keep_prob=input_keep_prob,
            is_train=is_train
        )
        trans = tf.nn.relu(trans)
        gate = linear(
            [arg],
            d,
            bias,
            bias_start=bias_start,
            scope='gate',
            wd=wd,
            input_keep_prob=input_keep_prob,
            is_train=is_train
        )
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(
        arg,
        num_layers,
        bias,
        bias_start=0.0,
        scope=None,
        wd=0.0,
        input_keep_prob=1.0,
        is_train=None
):
    '''
    封装了多层的highway network
    :param arg:输入
    :param num_layers:需要多少层
    :param bias:是否有偏置
    :param bias_start:偏置的初始值
    :param scope:变量的variable_scope名字
    :param wd:l2正则化系数
    :param input_keep_prob:每层highway network输入处有Dropout
    :param is_train:是否在训练，和dropout有关
    :return: 最后一层highway network的输出，输出的shape和输入的shape相同
    '''
    with tf.variable_scope(scope or "highway_network"):
        # 当前层的输入
        prev = arg
        # 当前层的输出
        cur = None
        # 对每一层
        for layer_idx in range(num_layers):
            # 当前层的输出
            cur = highway_layer(
                prev,
                bias,
                bias_start=bias_start,
                scope="layer_{}".format(layer_idx),
                wd=wd,
                input_keep_prob=input_keep_prob,
                is_train=is_train
            )
            prev = cur
        return cur


def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
    '''
    卷积层->relu->max polling，得到某个词的embedding
    只在词内进行卷积，卷积核不在词与词之间滑动
    max_polling在W维度上进行，对某个词的某个feature维度，取所有单词中的最大值
    :param in_:[N*M,JX,W,dc]
    :param filter_size:有多少个卷积核
    :param height:卷积核的大小
    :param padding:'VALID'或者'SAME'
    :param is_train:是否是训练状态，与Dropout有关
    :param keep_prob:输入的Dropout
    :param scope:变量的scope
    :return:[N*M,JX,filter_size]
    '''
    with tf.variable_scope(scope or "conv1d"):
        # num_channels:单词的embedding的维度
        num_channels = in_.get_shape()[-1]
        # todo:没有指定初始化策略
        filter_ = tf.get_variable(
            "filter",
            shape=[1, height, num_channels, filter_size],
            dtype='float'
        )
        # todo:没有指定初始化策略
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        # 如果是训练and启用了Dropout，那么就在输入层进行Dropout
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        # xxc:[N*M,JX,W-height+1,filter_size]
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias
        # 经过激活函数后，进行max pooling操作，得到某个词的embedding
        # out:[N*M,JX,filter_size]
        out = tf.reduce_max(tf.nn.relu(xxc), 2)
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
    '''
    计算经过不同类型的卷积核处理之后的词嵌入，内部包含卷积层->relu->max polling
    :param in_:[N*M,JX,W,dc]
    :param filter_sizes:不同大小的卷积核每种需要多少个的list
    :param heights:不同大小的卷积核的list
    :param padding:'VALID'或者'SAME'
    :param is_train:是否是训练状态，与Dropout有关
    :param keep_prob:输入的Dropout
    :param scope:变量的scope
    :return:[N*M,JX,filter_size*多少种卷积核]
    '''
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        # 对每个filter
        for filter_size, height in zip(filter_sizes, heights):
            if filter_size == 0:
                continue
            # 创建每个filter
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob, scope="conv1d_{}".format(height))
            # [多少种卷积核,N*M,JX,filter_size]
            outs.append(out)
        # [N*M,JX,filter_size*多少种卷积核]
        concat_out = tf.concat(axis=2, values=outs)
        return concat_out
