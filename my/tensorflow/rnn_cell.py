import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, RNNCell, LSTMStateTuple

from my.tensorflow import exp_mask, flatten
from my.tensorflow.nn import linear, softsel, double_linear_logits


class SwitchableDropoutWrapper(DropoutWrapper):
    '''
    通过is_train来打开或者关闭Dropout，Dropout施加在输入层和输出层
    '''
    def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0,
             seed=None):
        super(SwitchableDropoutWrapper, self).__init__(
            cell,
            input_keep_prob=input_keep_prob,
            output_keep_prob=output_keep_prob,
            seed=seed
        )
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        '''
        根据self.is_train来控制是否使用输入层和输出层的dropout，self.is_train为False，就不dropout，反之dropout
        :param inputs:输入
        :param state: 上一个step中cell的状态
        :param scope:变量的scope
        :return:
        '''
        # 这是input和output有dropout的输出
        outputs_do, new_state_do = super(SwitchableDropoutWrapper, self).__call__(
            inputs,
            state,
            scope=scope
        )
        # 打开下一级别的variable_scope中的reuse
        tf.get_variable_scope().reuse_variables()
        # 这是input和output没有dropout的输出
        outputs, new_state = self._cell(inputs, state, scope)
        outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
        if isinstance(state, tuple):
            new_state = state.__class__(
                *[
                    tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                    for new_state_do_i, new_state_i in zip(new_state_do, new_state)
                ]
            )
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do, lambda: new_state)
        return outputs, new_state


class TreeRNNCell(RNNCell):
    def __init__(self, cell, input_size, reduce_func):
        self._cell = cell
        self._input_size = input_size
        self._reduce_func = reduce_func

    def __call__(self, inputs, state, scope=None):
        """
        :param inputs: [N*B, I + B]
        :param state: [N*B, d]
        :param scope:
        :return: [N*B, d]
        """
        with tf.variable_scope(scope or self.__class__.__name__):
            d = self.state_size
            x = tf.slice(inputs, [0, 0], [-1, self._input_size])  # [N*B, I]
            mask = tf.slice(inputs, [0, self._input_size], [-1, -1])  # [N*B, B]
            B = tf.shape(mask)[1]
            prev_state = tf.expand_dims(tf.reshape(state, [-1, B, d]), 1)  # [N, B, d] -> [N, 1, B, d]
            mask = tf.tile(tf.expand_dims(tf.reshape(mask, [-1, B, B]), -1), [1, 1, 1, d])  # [N, B, B, d]
            # prev_state = self._reduce_func(tf.tile(prev_state, [1, B, 1, 1]), 2)
            prev_state = self._reduce_func(exp_mask(prev_state, mask), 2)  # [N, B, d]
            prev_state = tf.reshape(prev_state, [-1, d])  # [N*B, d]
            return self._cell(x, prev_state)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size


class NoOpCell(RNNCell):
    def __init__(self, num_units):
        self._num_units = num_units

    def __call__(self, inputs, state, scope=None):
        return state, state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


class MatchCell(RNNCell):
    def __init__(self, cell, input_size, q_len):
        self._cell = cell
        self._input_size = input_size
        # FIXME : This won't be needed with good shape guessing
        self._q_len = q_len

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """

        :param inputs: [N, d + JQ + JQ * d]
        :param state: [N, d]
        :param scope:
        :return:
        """
        with tf.variable_scope(scope or self.__class__.__name__):
            c_prev, h_prev = state
            x = tf.slice(inputs, [0, 0], [-1, self._input_size])
            q_mask = tf.slice(inputs, [0, self._input_size], [-1, self._q_len])  # [N, JQ]
            qs = tf.slice(inputs, [0, self._input_size + self._q_len], [-1, -1])
            qs = tf.reshape(qs, [-1, self._q_len, self._input_size])  # [N, JQ, d]
            x_tiled = tf.tile(tf.expand_dims(x, 1), [1, self._q_len, 1])  # [N, JQ, d]
            h_prev_tiled = tf.tile(tf.expand_dims(h_prev, 1), [1, self._q_len, 1])  # [N, JQ, d]
            f = tf.tanh(linear([qs, x_tiled, h_prev_tiled], self._input_size, True, scope='f'))  # [N, JQ, d]
            a = tf.nn.softmax(exp_mask(linear(f, 1, True, squeeze=True, scope='a'), q_mask))  # [N, JQ]
            q = tf.reduce_sum(qs * tf.expand_dims(a, -1), 1)
            z = tf.concat(axis=1, values=[x, q])  # [N, 2d]
            return self._cell(z, state)


class AttentionCell(RNNCell):
    '''
    Attention中MLP的权重不带L2正则化系数
    '''
    def __init__(self, cell, memory, mask=None, controller=None, mapper=None, input_keep_prob=1.0, is_train=None):
        """
        Early fusion attention cell: uses the (inputs, state) to control the current attention.
        我将根据它第一次在代码中出现的位置来推断输入输出的shape
        :param cell:
        :param memory: [N,M,JQ,2d]
        :param mask:
        :param controller: (inputs, prev_state, memory) -> memory_logits
        """
        self._cell = cell
        self._memory = memory
        self._mask = mask
        # [N*M,JQ,2d]
        self._flat_memory = flatten(memory, 2)
        # [N*M,JQ]
        self._flat_mask = flatten(mask, 1)
        if controller is None:
            controller = AttentionCell.get_linear_controller(
                True,
                is_train=is_train
            )
        self._controller = controller
        if mapper is None:
            mapper = AttentionCell.get_concat_mapper()
        elif mapper == 'sim':
            mapper = AttentionCell.get_sim_mapper()
        self._mapper = mapper

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        '''
        分别对M句话中的每一句话的每一个词，得到对于整个JQ的attented vector，依据mapper的选择，和文章中某个词的表达进行融合，并送入表达文章中某个词和整个问题的匹配程度的LSTM中进行计算，做法很像MatchLSTM
        :param inputs: [N*M,2d]
        :param state: tuple(cell_state->[N*M,2d],hidden_state->[N*M,2d])
        :param scope:
        :return:
        '''
        with tf.variable_scope(scope or "AttentionCell"):
            # memory_logits:[N*M,JQ]
            # 得到对问题的每一个词的未归一化Attention
            memory_logits = self._controller(inputs, state, self._flat_memory)
            # sel_mem:[N*M,2d]
            # 得到文章中某个词对整个问题的Attented Vector
            sel_mem = softsel(self._flat_memory, memory_logits, mask=self._flat_mask)
            # new_inputs:[N*M,nd],n和选用的_mapper函数有关
            # new_state:和state相同
            new_inputs, new_state = self._mapper(inputs, state, sel_mem)
            # 把它作为LSTM下一次的输入送进去
            return self._cell(new_inputs, state)

    @staticmethod
    def get_double_linear_controller(size, bias, input_keep_prob=1.0, is_train=None):
        def double_linear_controller(inputs, state, memory):
            """
            linear(tanh(linear))，使用这种方式计算的Attention是Bahdanau Attention
            得到的是对JQ中每个词的未归一化的Attention
            :param inputs: [N*M,2d]
            :param state: 如果是LSTMcell，tuple(cell_state->[N*M,2d],hidden_state->[N*M,2d])
            :param memory: [N*M,JQ,2d]
            :return: [N*M,JQ]
            """
            rank = len(memory.get_shape())
            _memory_size = tf.shape(memory)[rank-2]
            # [N*M,JQ,2d]
            tiled_inputs = tf.tile(
                tf.expand_dims(inputs, 1),
                [1, _memory_size, 1]
            )
            # 如果cell用的是LSTM的话，这里应该是cell,hidden
            # 重复了JQ遍
            if isinstance(state, tuple):
                tiled_states = [
                    # [N*M,JQ,2d]
                    tf.tile(
                        tf.expand_dims(each, 1),
                        [1, _memory_size, 1]
                    )
                    for each in state
                ]
            else:
                # [N*M,JQ,2d]
                tiled_states = [
                    tf.tile(
                        tf.expand_dims(state, 1),
                        [1, _memory_size, 1]
                    )
                ]

            # input: [N*M,JQ,2d]
            # cell:  [N*M,JQ,2d]
            # hidden:[N*M,JQ,2d]
            # memory:[N*M,JQ,2d]
            # 输出[N*M,JQ,8d]
            in_ = tf.concat([tiled_inputs] + tiled_states + [memory], axis=2)
            out = double_linear_logits(
                in_,
                size,
                bias,
                input_keep_prob=input_keep_prob,
                is_train=is_train
            )
            return out
        return double_linear_controller

    @staticmethod
    def get_linear_controller(bias, input_keep_prob=1.0, is_train=None):
        def linear_controller(inputs, state, memory):
            '''
            linear，未归一化的Attention
            得到的是对JQ中每个词的未归一化的Attention，它和标准Attention的计算不一样，没有激活函数，全部都是线性层
            :param inputs: [N*M,2d]
            :param state:如果是LSTMcell，tuple(cell_state->[N*M,2d],hidden_state->[N*M,2d])
            :param memory:[N*M,JQ,2d]
            :return: [N*M,JQ]
            '''
            rank = len(memory.get_shape())
            # memory倒数第二个shape
            _memory_size = tf.shape(memory)[rank-2]
            # [N*M,JQ,2d]
            tiled_inputs = tf.tile(
                tf.expand_dims(inputs, 1),
                [1, _memory_size, 1]
            )
            # 如果cell用的是LSTM的话，这里应该是cell,hidden
            # 重复了JQ遍
            if isinstance(state, tuple):
                tiled_states = [
                    # [N*M,JQ,2d]
                    tf.tile(
                        tf.expand_dims(each, 1),
                        [1, _memory_size, 1]
                    )
                    for each in state
                ]
            else:
                # [N*M,JQ,2d]
                tiled_states = [
                    tf.tile(
                        tf.expand_dims(state, 1),
                        [1, _memory_size, 1]
                    )
                ]

            # input: [N*M,JQ,2d]
            # cell:  [N*M,JQ,2d]
            # hidden:[N*M,JQ,2d]
            # memory:[N*M,JQ,2d]
            # 输出[N*M,JQ,8d]
            in_ = tf.concat([tiled_inputs] + tiled_states + [memory], axis=2)
            # [N*M,JQ]，未归一化
            # 得到的是对JQ中每个词的未归一化的Attention，它和标准Attention的计算不一样，没有激活函数，全部都是线性层
            out = linear(in_, 1, bias, squeeze=True, input_keep_prob=input_keep_prob, is_train=is_train)
            return out
        return linear_controller

    @staticmethod
    def get_concat_mapper():
        def concat_mapper(inputs, state, sel_mem):
            """
            直接进行拼接
            :param inputs: [N, i]
            :param state: [N, d]
            :param sel_mem: [N, m]
            :return: (new_inputs, new_state) tuple
            """
            return tf.concat(
                axis=1,
                values=[inputs, sel_mem]
            ), state
        return concat_mapper

    @staticmethod
    def get_sim_mapper():
        def sim_mapper(inputs, state, sel_mem):
            """
            Assume that inputs and sel_mem are the same size，除了拼接之外，还做了一些人工提取特征的操作
            :param inputs: 文章中某个词,[N*M,2d]
            :param state: 使用LSTM(cell:[N*M, 2d],hidden:[N*M,2d])
            :param sel_mem: 问题的Attented Vector，[N*M,2d]
            :return: (new_inputs, new_state) tuple
            """
            return \
                tf.concat(
                    axis=1,
                    values=[
                        inputs,
                        sel_mem,
                        inputs * sel_mem,
                        tf.abs(inputs - sel_mem)
                    ]
                ),\
                state
        return sim_mapper
