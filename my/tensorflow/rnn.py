import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn as _dynamic_rnn, \
    bidirectional_dynamic_rnn as _bidirectional_dynamic_rnn

from my.tensorflow import flatten, reconstruct


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_outputs, final_state = _dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bw_dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                   dtype=None, parallel_iterations=None, swap_memory=False,
                   time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_inputs = tf.reverse(flat_inputs, 1) if sequence_length is None \
        else tf.reverse_sequence(flat_inputs, sequence_length, 1)
    flat_outputs, final_state = _dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)
    flat_outputs = tf.reverse(flat_outputs, 1) if sequence_length is None \
        else tf.reverse_sequence(flat_outputs, sequence_length, 1)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    '''
    包装了TensorFlow里的bidirectional_dynamic_rnn，主要的调整了一下数据的格式，来适配TensorFlow的api
    :param cell_fw:
    :param cell_bw:
    :param inputs:
    :param sequence_length:
    :param initial_state_fw:
    :param initial_state_bw:
    :param dtype:
    :param parallel_iterations:
    :param swap_memory:
    :param time_major:必须是False
    :param scope:
    :return:见下面代码返回部分
    '''
    # 不打开time_major
    assert not time_major
    # flat_inputs的几种情况
    # qq:[N,JQ,dw+dco]，如果使用char_embedding和word_embedding
    # xx:[N*M,JX,dw+dco]，如果使用char_embedding和word_embedding
    flat_inputs = flatten(inputs, 2)

    flat_len = None \
        if sequence_length is None \
        else \
        tf.cast(
            # flat_len的几种情况
            # q_len:[N]
            # x_len:[N*M]
            flatten(sequence_length, 0), 'int64'
        )

    (flat_fw_outputs, flat_bw_outputs), \
    final_state = \
        _bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            flat_inputs,
            sequence_length=flat_len,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            dtype=dtype,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            time_major=time_major,
            scope=scope
        )

    # fw_outputs和bw_outputs的几种情况
    # qq:[N,JQ,d]
    # xx:[N,M,JX,d]
    # 每个time step下前向网络和后向网络的输出
    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    # FIXME : final state is not reshaped!
    # final_state:
    # 如果cell是BasicLSTMCell，它是一个表示前向和后向最终状态的(cell,hidden)的元组
    # qq:([N,d],[N,d])、xx:([N*M,d],[N*M,d])
    return (fw_outputs, bw_outputs), final_state


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    (flat_fw_outputs, flat_bw_outputs), final_state = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                                        initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                        dtype=dtype, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    # FIXME : final state is not reshaped!
    return (fw_outputs, bw_outputs), final_state
