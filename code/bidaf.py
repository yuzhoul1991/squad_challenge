"""This file contains components needed for the biDAF implementation"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

from modules import masked_softmax

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

    def build_graph(self, questions, questions_mask, contexts, contexts_mask):
        """
        Bidirectional Attention implementation

        Inputs:
          questions: Tensor shape (batch_size, num_question, 2*h).
          question_mask: Tensor shape (batch_size, num_questions).
            1s where there's real input, 0s where there's padding
          contexts: Tensor shape (batch_size, num_context, 2*h)
          contexts_mask: Tensor shape (batch_size, num_context).

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BiDAF"):
            M = questions.shape.as_list()[1]
            N = contexts.shape.as_list()[1]
            h = questions.shape.as_list()[2]
            assert(h == contexts.shape.as_list()[2])

            # Calculate similarity matrix
            # S_ij = w_sim^T [c_i;q_j;c_i o q_j]

            # 1. compute w x C
            W1 = tf.get_variable("w_sim_1", shape=[h, 1])
            C = tf.expand_dims(contexts, axis=2)    # shape = b x N x 1 x h
            C = tf.reshape(C, [-1, h])
            S1 = tf.matmul(C, W1)
            S1 = tf.reshape(S1, [-1, N, 1]) # shape = b x N x 1

            # 2. compute w x Q
            W2 = tf.get_variable("w_sim_2", shape=[h, 1])
            Q = tf.expand_dims(questions, axis=1)   # shape = b x 1 x M x h
            Q = tf.reshape(Q, [-1, h])
            S2 = tf.matmul(Q, W2)
            S2 = tf.reshape(S2, [-1, 1, M]) # shape = b x 1 x M

            # 2. generate c_i o q_j in matrix form
            # Use broadcasting tf.multiply([b, M, 1, h], [b, 1, N, h]) == [b, M, N, h]
            W3 = tf.get_variable("w_sim_3", shape=[h, 1])
            C_o_Q = tf.multiply(tf.expand_dims(contexts, axis=1), tf.expand_dims(questions, axis=2)) # shape = b x N x M x h
            C_o_Q = tf.reshape(C_o_Q, [-1, h])
            S3 = tf.matmul(C_o_Q, W3)
            S3 = tf.reshape(S3, [-1, N, M]) # shape = b x N x M

            # 3. Use broadcasting to add them up to get S
            S = S1 + S2 + S3    # shape = b x N x M


            # Perform Context-to-Question C2Q attention
            _, alpha = masked_softmax(S, tf.expand_dims(questions_mask, axis=1), dim=2) # shape = b x N x M
            A = tf.matmul(alpha, questions)   # shape = b x N x 2h

            # Perform Question-to-Context Q2C attention
            m = tf.reduce_max(S, axis=2, keep_dims=True)    # shape = b x N x 1
            _, beta = masked_softmax(m, tf.expand_dims(contexts_mask, axis=2), dim=1)  # shape = b x N x 1
            beta_t = tf.transpose(beta, perm=[0, 2, 1]) # shape = b x 1 x N
            c = tf.matmul(beta_t, contexts)   # shape = b x 1 x h

            c_tiled = tf.tile(c, [1, N, 1])

            # Apply dropout
            A_drop = tf.nn.dropout(A, self.keep_prob)
            c_drop = tf.nn.dropout(c_tiled, self.keep_prob)

            return A_drop, c_drop
