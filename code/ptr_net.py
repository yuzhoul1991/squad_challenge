import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

class PointerNet(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def build_graph(self, source_hidden_states, mask, **kwargs):

        labels = tf.unstack(tf.cast(kwargs['labels'], tf.float32), axis=1)
        seq_len = tf.reduce_sum(mask, reduction_indices=1) # shape (batch_size)

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            source_hidden_states.shape.as_list()[-1],
            source_hidden_states,
            memory_sequence_length=seq_len
        )
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            rnn_cell.BasicLSTMCell(self.hidden_size),
            attention_mechanism,
            cell_input_fn=lambda input, context: context
        )

        import pdb;pdb.set_trace()

        logits, _ = tf.nn.static_rnn(attention_cell, labels, dtype=tf.float32)

        import pdb;pdb.set_trace()

        return logits_start, logits_end, probdist_start, probdist_end
