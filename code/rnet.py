import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

from modules import masked_softmax, RNNEncoder

class SelfAttention(object):
    """
    Module for rnet self-attention.
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

    def build_graph(self, question_hiddens, question_mask, context_hiddens, context_mask, scope="self_attention"):
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
        with vs.variable_scope(scope):

            hidden_size = 75
            N = context_hiddens.shape.as_list()[1]
            M = question_hiddens.shape.as_list()[1]
            d = question_hiddens.shape.as_list()[2]
            assert(d == context_hiddens.shape.as_list()[2])
            batch_size = tf.shape(question_hiddens)[0]

            temp1 = tf.nn.relu(self.one_layer(question_hiddens, hidden_size, 'question'))
            #temp1 = tf.expand_dims(temp1, axis=1)   # shape = b x 1 x M x hidden_size

            temp2 = tf.nn.relu(self.one_layer(context_hiddens, hidden_size, 'context'))
            #temp2 = tf.expand_dims(temp2, axis=2)   # shape = b x N x 1 x hidden_size

	    E = tf.matmul(temp2, tf.transpose(temp1, [0, 2, 1]))

            #tanh = tf.tanh(tf.add(temp1, temp2))    # shape = b x N x M x hidden_size
            #v = tf.get_variable("v", shape=[hidden_size, 1])
            #E = tf.reshape(tf.matmul(tf.reshape(tanh, [-1, hidden_size]), v), [-1, N, M]) # shape = b x N x M

            _, alpha = masked_softmax(E, tf.expand_dims(question_mask, axis=1), dim=2)  # shape = b x N x M
            A = tf.matmul(alpha, question_hiddens)  # shape = b x N x 2h

        with vs.variable_scope(scope+'_gate'):
            concat = tf.concat([context_hiddens, A], axis=2)    # shape = b x N x 4h
            concat_d = tf.nn.dropout(concat, self.keep_prob)
            G = tf.nn.sigmoid(self.one_layer(concat_d, concat.shape.as_list()[-1], 'concat'))   # shape = b x N x 4h
            gated = tf.multiply(G, concat)

        with vs.variable_scope(scope+'_lstm'):
            # Send to a biLSTM
            biLSTM = RNNEncoder(hidden_size, self.keep_prob, 'lstm')
            # Already applied dropout in RNNEncoder.build_graph
            output = biLSTM.build_graph(gated, context_mask, "BiLSTM")

        return None, output


    def one_layer(self, input, hidden_size, scope):
        with vs.variable_scope(scope):
            d = input.shape.as_list()[-1]
            original_size = input.shape.as_list()[1]
            W = tf.get_variable("W", shape=[d, hidden_size])
            input_ = tf.reshape(input, [-1, d])
            return tf.reshape(tf.matmul(input_, W), [-1, original_size, hidden_size])   # shape = b x original_size x hidden_size
