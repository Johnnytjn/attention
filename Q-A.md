Q1:为什么RNN 需要mask 输入，而CNN则不需要？
对于变长的句子或者文本输入，为什么RNN，LSTM 处理的时候，需要mask 输入屏蔽掉一部分文本，而CNN不需要呢？
A:对于RNN来说，如果不用mask而是补0的话，补0的位置也会参与状态向量的计算，用mask和补0相比，得到的状态向量是不一样的。
因为RNN状态向量计算的时候不仅仅考虑了当前输入，也考虑了上一次的状态向量，因此靠补0的方式进行屏蔽是不彻底的。而CNN是卷积操作，
补0的位置对卷积结果没有影响，即补0和mask两种方式的结果是一样的，因此大家为了省事起见，就普遍在CNN使用补0的方法了。

A:for transduction problems (1:1 between sequence input and target) the general approach i think is to allow the RNN to run over these NUL values but then you apply a mask to zero out the cost associated with them.

eg for sequence  [w1, w2, w3, NUL, NUL, NUL]
you first calculate the per element costs, say,  costs = [3.1, 4.1, 5.9, 2.6, 5.3, 5.8]

usually you'd take the mean; np.mean(costs) = 4.8, but in this case you don't care about the last three.

so now you'll maintain a mask, 0 for NUL and 1 otherwise, mask = [1,1,1,0,0,0]
and you'll calculate your sequence cost using this mask to zero out the costs you don't care about;
sequence_cost = np.sum(costs * mask) / np.sum(mask)
(note! NOT np.mean(costs * mask) since the effective sequence "length" has changed from 6 to 3)

it's "wasteful" in the sense you're doing more work than the unpadded version but the argument is the denser packed data makes up for in the speed up of the lower level libraries

there are lots of examples of this in the tensorflow seq2seq models
see http://www.tensorflow.org/tutorials/seq2seq/index.html "bucketing and padding" for the high level view of this (+ the extended idea of bucketing)
and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py for more detail in code
