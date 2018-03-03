"""This file contains components needed for the biDAF implementation"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

from modules import masked_softmax, RNNEncoder

class BidirectionalAttention(object):
    """
        Module for bidirectional attention.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """

        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

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

        with vs.variable_scope("BiDAF"):
            N = context_hiddens.shape.as_list()[1]
            M = question_hiddens.shape.as_list()[1]
            h = question_hiddens.shape.as_list()[2]
            assert(h == context_hiddens.shape.as_list()[2])
            batch_size = tf.shape(question_hiddens)[0]

            # Calculate similarity matrix
            # S_ij = w_sim^T [c_i;q_j;c_i o q_j]

            # 1. compute w x C
            W1 = tf.get_variable("w_sim_1", shape=[h, 1])
            C = tf.expand_dims(context_hiddens, axis=2)    # shape = b x N x 1 x h
            C = tf.reshape(C, [-1, h])
            S1 = tf.matmul(C, W1)
            S1 = tf.reshape(S1, [-1, N, 1]) # shape = b x N x 1

            # 2. compute w x Q
            W2 = tf.get_variable("w_sim_2", shape=[h, 1])
            Q = tf.expand_dims(question_hiddens, axis=1)   # shape = b x 1 x M x h
            Q = tf.reshape(Q, [-1, h])
            S2 = tf.matmul(Q, W2)
            S2 = tf.reshape(S2, [-1, 1, M]) # shape = b x 1 x M

            # 2. generate c_i o q_j in matrix form
            # Use broadcasting tf.multiply([b, M, 1, h], [b, 1, N, h]) == [b, M, N, h]
            W3 = tf.get_variable("w_sim_3", shape=[h, 1])
            C_o_Q = tf.multiply(tf.expand_dims(context_hiddens, axis=1), tf.expand_dims(question_hiddens, axis=2)) # shape = b x N x M x h
            C_o_Q = tf.reshape(C_o_Q, [-1, h])
            S3 = tf.matmul(C_o_Q, W3)
            S3 = tf.reshape(S3, [-1, N, M]) # shape = b x N x M

            # 3. Use broadcasting to add them up to get S
            S = S1 + S2 + S3    # shape = b x N x M


            # Perform Context-to-Question C2Q attention
            _, alpha = masked_softmax(S, tf.expand_dims(question_mask, axis=1), dim=2) # shape = b x N x M
            A = tf.matmul(alpha, question_hiddens)   # shape = b x N x 2h

            # Perform Question-to-Context Q2C attention
            m = tf.reduce_max(S, axis=2, keep_dims=True)    # shape = b x N x 1
            _, beta = masked_softmax(m, tf.expand_dims(context_mask, axis=2), dim=1)  # shape = b x N x 1
            beta_t = tf.transpose(beta, perm=[0, 2, 1]) # shape = b x 1 x N
            c = tf.matmul(beta_t, context_hiddens)   # shape = b x 1 x h

            c_tiled = tf.tile(c, [1, N, 1])

            # Apply dropout

            G = tf.concat([context_hiddens, A, tf.multiply(context_hiddens, A), tf.multiply(context_hiddens, c_tiled)], axis=2)  # shape = b x N x 4h


            # Send to a biLSTM
            biLSTM = RNNEncoder(h, self.keep_prob, 'lstm')
            biLSTM_mask = tf.fill([batch_size, N], 1)
            # Already applied dropout in RNNEncoder.build_graph
            output = biLSTM.build_graph(G, biLSTM_mask, "BiLSTM")

            return output
