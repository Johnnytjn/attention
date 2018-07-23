#! -*- coding: utf-8 -*-

import tensorflow as tf

'''
inputs是一个形如(batch_size, seq_len, word_size)的张量；
函数返回一个形如(batch_size, seq_len, position_size)的位置张量。
'''
def Position_Embedding(inputs, position_size):
    batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
    position_j = 1. / tf.pow(10000., \
                             2 * tf.range(position_size / 2, dtype=tf.float32 \
                            ) / position_size)
    position_j = tf.expand_dims(position_j, 0)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    position_i = tf.expand_dims(position_i, 1)
    position_ij = tf.matmul(position_i, position_j)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
    position_embedding = tf.expand_dims(position_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_size))
    return position_embedding
'''
TensorFlow中，想要维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]

'''

'''
inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
'''
def Mask(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12
'''tf.sequence_mask
Returns a mask tensor representing the first N positions of each cell.
tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                #  [True, True, True, False, False],
                                #  [True, True, False, False, False]]

tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
                                  #   [True, True, True]],
                                  #  [[True, True, False],
                                  #   [False, False, False]]]
'''

'''
普通的全连接
inputs是一个二阶或二阶以上的张量，即形如(batch_size,...,input_size)。
只对最后一个维度做矩阵乘法，即输出一个形如(batch_size,...,ouput_size)的张量。
'''
def Dense(inputs, ouput_size, bias=True, seq_len=None):
    input_size = int(inputs.shape[-1])
    W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs, \
                         tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
                        )
    if seq_len != None:
        outputs = Mask(outputs, seq_len, 'mul')
    return outputs

'''
Multi-Head Attention的实现
'''
def Attention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
    #对Q、K、V分别作线性映射
    Q = Dense(Q, nb_head * size_per_head, False)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
    Q = tf.transpose(Q, [0, 2, 1, 3])
    K = Dense(K, nb_head * size_per_head, False)
    K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
    K = tf.transpose(K, [0, 2, 1, 3])
    V = Dense(V, nb_head * size_per_head, False)
    V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
    V = tf.transpose(V, [0, 2, 1, 3])
    #计算内积，然后mask，然后softmax
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))  #size_per_head:d_k(word_embbeding_len)
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, V_len, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    #输出并mask
    O = tf.matmul(A, V)
    O = tf.transpose(O, [0, 2, 1, 3])
    O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
    O = Mask(O, Q_len, 'mul')
    return O
'''
1.tf.multiply（）两个矩阵中对应元素各自相乘
格式: tf.multiply(x, y, name=None) 
参数: 
x: 一个类型为:half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128的张量。 
y: 一个类型跟张量x相同的张量。 
返回值： x * y element-wise. 
注意： 
（1）multiply这个函数实现的是元素级别的相乘，也就是两个相乘的数元素各自相乘，而不是矩阵乘法，注意和tf.matmul区别。 
（2）两个相乘的数必须有相同的数据类型，不然就会报错。

2.tf.matmul（）将矩阵a乘以矩阵b，生成a * b。
格式: tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None) 
参数: 
a: 一个类型为 float16, float32, float64, int32, complex64, complex128 且张量秩 > 1 的张量。 
b: 一个类型跟张量a相同的张量。 
transpose_a: 如果为真, a则在进行乘法计算前进行转置。 
transpose_b: 如果为真, b则在进行乘法计算前进行转置。 
adjoint_a: 如果为真, a则在进行乘法计算前进行共轭和转置。 
adjoint_b: 如果为真, b则在进行乘法计算前进行共轭和转置。 
a_is_sparse: 如果为真, a会被处理为稀疏矩阵。 
b_is_sparse: 如果为真, b会被处理为稀疏矩阵。 
name: 操作的名字（可选参数） 
返回值： 一个跟张量a和张量b类型一样的张量且最内部矩阵是a和b中的相应矩阵的乘积。 
注意： 
（1）输入必须是矩阵（或者是张量秩 >２的张量，表示成批的矩阵），并且其在转置之后有相匹配的矩阵尺寸。 
（2）两个矩阵必须都是同样的类型，支持的类型如下：float16, float32, float64, int32, complex64, complex128。 
引发错误: 
ValueError: 如果transpose_a 和 adjoint_a, 或 transpose_b 和 adjoint_b 都被设置为真
'''

'''

tf.transpose

a 是一个张量（Tensor），实际上就是一个数组
perm 是 a 维度的置换
name :操作的名称
其返回的是一个转置的张量。


a的转置是根据 perm 的设定值来进行的。
返回数组的 dimension（尺寸、维度） i与输入的 perm[i]的维度相一致。如果未给定perm，默认设置为 (n-1...0)，这里的 n 值是输入变量的 rank 。因此默认情况下，这个操作执行了一个正规（regular）的2维矩形的转置

# 'perm' is more useful for n-dimensional tensors, for n > 2
# 'x' is   [[[1  2  3]
#            [4  5  6]]
#           [[7  8  9]
#            [10 11 12]]]
# Take the transpose of the matrices in dimension-0
tf.transpose(b, perm=[0, 2, 1]) ==> [[[1  4]
                                      [2  5]
                                      [3  6]]
                                     [[7 10]
                                      [8 11]
                                      [9 12]]]
Args: 
•a: A Tensor.
•perm: A permutation of the dimensions of a.


'''
