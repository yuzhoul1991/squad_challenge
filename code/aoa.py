import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

from modules import masked_softmax, RNNEncoder

class AttentionOverAttention(object):
    """
    Module for coattention.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob

    def build_graph(self, question_hiddens, question_mask, context_hiddens, context_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          question_hiddens: b x M x 2h
          question_mask: b x M
            1s where there's real input, 0s where there's padding
          context_hiddens: b x N x 2h
          context_mask: b x N
        """
        with vs.variable_scope("AoA"):

            N = context_hiddens.shape.as_list()[1]
            M = question_hiddens.shape.as_list()[1]
            d = question_hiddens.shape.as_list()[2]
            assert(d == context_hiddens.shape.as_list()[2])
            batch_size = tf.shape(question_hiddens)[0]

            M = tf.matmul(context_hiddens, tf.transpose(question_hiddens, perm=[0, 2, 1]))   # shape = b x N x M

            # query-to-document attention; col-wise softmax
            _, alpha = masked_softmax(M, tf.expand_dims(context_mask, axis=2), dim=1)

            # document-to-query attention; row-wise softmax
            _, beta = masked_softmax(M, tf.expand_dims(question_mask, axis=1), dim=2)

            beta = tf.reduce_mean(beta, axis=1, keep_dims=True) # shape = b x 1 x M

            output = tf.matmul(M, tf.transpose(beta, perm=[0, 2, 1]))   # shape = b x N x 1
            import pdb; pdb.set_trace()
            output = tf.squeeze(output, axis=2)

            return None, output
