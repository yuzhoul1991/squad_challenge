import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

from modules import masked_softmax, RNNEncoder

class Coattention(object):
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
        with vs.variable_scope("Coattention"):

            N = context_hiddens.shape.as_list()[1]
            M = question_hiddens.shape.as_list()[1]
            d = question_hiddens.shape.as_list()[2]
            assert(d == context_hiddens.shape.as_list()[2])
            batch_size = tf.shape(question_hiddens)[0]

            spare_mask = tf.fill([batch_size, 1], 0)

            question_mask = tf.concat([question_mask, spare_mask], 1)
            context_mask = tf.concat([context_mask, spare_mask], 1)

            c_sentinel = tf.get_variable("c_sentinel", shape=[1, d], initializer=tf.contrib.layers.xavier_initializer())
            q_sentinel = tf.get_variable("q_sentinel", shape=[1, d], initializer=tf.contrib.layers.xavier_initializer())

            Q = tf.concat([question_hiddens, tf.tile(tf.expand_dims(q_sentinel, axis=0), [batch_size, 1, 1])], 1) # shape b x M+1 x 2h
            C = tf.concat([context_hiddens, tf.tile(tf.expand_dims(c_sentinel, axis=0), [batch_size, 1, 1])], 1) # shape b x N+1 x 2h

            w_proj = tf.get_variable("w_proj", shape=[M+1, M+1], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[1, M + 1], initializer=tf.contrib.layers.xavier_initializer())
            Q_t = tf.transpose(Q, perm=[0, 2, 1])   # shape b x 2h x M+1
            Q_t = tf.reshape(Q_t, [-1, M+1])
            wq = tf.reshape(tf.matmul(Q_t, w_proj) + b, [-1, d, M+1])
            Q_proj = tf.transpose(tf.tanh(wq), [0, 2, 1])    # shape b x M+1 x 2h

            L = tf.matmul(C, tf.transpose(Q, perm=[0, 2, 1]))    # shape b x N+1 x M+1

            # Calculate Context-to-Question Attention
            # softmax across rows ie. for each context word, take a softmax of masked question words
            _, alpha = masked_softmax(L, tf.expand_dims(question_mask, axis=1), dim=2) # shape b x N+1 x M+1
            # A = tf.matmul(alpha, Q_proj)     # shape b x N+1 x 2h

            # Calculate Question-to-Context Attention
            # softmax across cols ie. for each question word, take a softmax of masked context words
            _, beta = masked_softmax(L, tf.expand_dims(context_mask, axis=2), dim=1)   # shape b x N+1 x M+1
            B = tf.matmul(tf.transpose(beta, perm=[0, 2, 1]), C)  # shape b x M+1 x 2h

            # Calculate 2nd level attention
            # From original paper this can be paralleized with [Q_proj;B]alpha
            # S = tf.matmul(alpha, B) # shape b x N+1 x 2h
            # S_n_A = tf.concat([S, A], axis=2)   # shape b x N+1 x 4h
            S_n_A = tf.matmul(alpha, tf.concat([Q_proj, B], axis=2))    # shape b x N+1 x 4h

            # Send to a biLSTM
            biLSTM = RNNEncoder(d, self.keep_prob, 'lstm')
            # Already applied dropout in RNNEncoder.build_graph
            output = biLSTM.build_graph(tf.concat([C, S_n_A], axis=2), context_mask, "BiLSTM")

            return None, output[:,:-1,:]
