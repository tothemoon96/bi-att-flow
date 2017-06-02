from itertools import zip_longest

import itertools
import tensorflow as tf
from functools import reduce
from operator import mul
import numpy as np

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def get_initializer(matrix):
    def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
    return _initializer


def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, var in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            assert g is not None, var.name
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def mask(val, mask, name=None):
    if name is None:
        name = 'mask'
    return tf.multiply(val, tf.cast(mask, 'float'), name=name)


def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)


def flatten(tensor, keep):
    '''
    修改shape，保留后面keep个轴，将前面几个轴压缩起来
    :param tensor:一个tensor
    :param keep:要保留后面几个轴
    :return:reshape的结果
    '''
    fixed_shape = tensor.get_shape().as_list()
    # 取前几个轴
    start = len(fixed_shape) - keep
    # 将前几个轴的shape乘起来
    # 如[2,3,2,3,5]，取前4个轴，keep->1
    # 结果:2*3*2*3=36
    left = reduce(
        mul,
        [
            fixed_shape[i] or tf.shape(tensor)[i]
            for i in range(start)
        ]
    )
    # 拼接上后面的轴的shape
    # 继续上面的例子:[36,5]
    out_shape = [left] + \
                [
                    fixed_shape[i] or tf.shape(tensor)[i]
                    for i in range(start, len(fixed_shape))
                ]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    '''
    使用ref的除后面keep个轴的前面的轴和tensor的后面keep个轴组合起来，重建tensor
    举个例子:
    tensor->[10,2]
    ref->[2,5,2]
    keep->1
    返回[2,5,2]
    :param tensor: 需要重建shape的tensor
    :param ref: 要参考的重建shape
    :param keep: 要保留后面几个轴
    :return:重建后的结果
    '''
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    # ref的前面几个shape
    pre_shape = [
        ref_shape[i] or tf.shape(ref)[i]
        for i in range(ref_stop)
    ]
    # tensor的后面几个shape
    keep_shape = [
        tensor_shape[i] or tf.shape(tensor)[i]
        for i in range(tensor_start, len(tensor_shape))
    ]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def add_wd(wd, scope=None):
    '''
    得到某个scope中可以训练的变量的l2范数，乘上wd，添加到losses集合中去
    :param wd:L2正则化系数
    :param scope:变量的scope
    :return:
    '''
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    with tf.name_scope("weight_decay"):
        # 对于该scope中可以训练的变量中的每一个
        for var in variables:
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var),
                wd,
                name="{}/wd".format(var.op.name)
            )
            tf.add_to_collection('losses', weight_decay)


def grouper(iterable, n, fillvalue=None, shorten=False, num_groups=None):
    '''

    :param iterable: 可迭代的对象
    :param n: 每组多少个元素
    :param fillvalue: 不能对其的元素的填充值
    :param shorten:
    :param num_groups: 桶的数目
    :return:
    '''
    args = [iter(iterable)] * n
    out = zip_longest(*args, fillvalue=fillvalue)
    # 到这里grouper的功能相当于把iterable中的内容依次放入int(math.ceil(iterable/n))个桶中，每个桶有n个元素，放不满的位置填入None
    # 例如:[1,2,3,4,5]放入3个桶
    # out:[[1,2],[3,4],[5,None]]
    out = list(out)
    # 假设num_groups超过计算出来的桶的数目，用default填充each部分
    if num_groups is not None:
        default = (fillvalue, ) * n
        assert isinstance(num_groups, int)
        out = list(
            each for each, _ in zip_longest(
                out,
                range(num_groups),
                fillvalue=default
            )
        )
    # 在each中过滤掉None元素
    # 同样是上面的例子:
    # out:[[1,2],[3,4],[5]]
    if shorten:
        assert fillvalue is None
        out = (
            tuple(
                e for e in each if e is not None
            ) for each in out
        )
    return out

def padded_reshape(tensor, shape, mode='CONSTANT', name=None):
    paddings = [[0, shape[i] - tf.shape(tensor)[i]] for i in range(len(shape))]
    return tf.pad(tensor, paddings, mode=mode, name=name)


def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params
