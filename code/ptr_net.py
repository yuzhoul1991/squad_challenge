import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

from modules import masked_softmax

class PointerNet(object):
    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size

    def build_graph(self, source_hidden_states, context_mask, labels, question_hiddens, question_mask):
        def get_query_pooled(query_states, query_lengths, scope="pool_query"):
            with vs.variable_scope(scope):
                attn_size = query_states.get_shape()[2].value
                batch_size_m = tf.shape(query_lengths)
                print("Query Pooled attention size = %d" %attn_size)
                dummy_inputs =  [tf.zeros(batch_size_m, dtype=tf.int32)]
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                            attn_size,
                                            query_states,
                                            memory_sequence_length=query_lengths)
                input_fn = lambda curr_input, atts : atts  # independent of inputs
                cell = tf.contrib.seq2seq.AttentionWrapper(
                                        tf.nn.rnn_cell.BasicLSTMCell(attn_size),
                                        attention_mechanism,
                                        attention_layer_size = attn_size,
                                        cell_input_fn = input_fn)
                atts, _ = tf.nn.static_rnn(cell, dummy_inputs, dtype = tf.float32)
            return atts[0]


        labels = tf.unstack(labels, axis=1)
        context_len = tf.reduce_sum(context_mask, reduction_indices=1) # shape (batch_size)
        question_len = tf.reduce_sum(question_mask, reduction_indices=1) # shape (batch_size)

        atten_size = source_hidden_states.shape.as_list()[1]

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            atten_size,
            source_hidden_states,
            memory_sequence_length=context_len
        )
        init_state = get_query_pooled(question_hiddens, question_len)

        answer_ptr_input_fn = lambda curr_input, atts : tf.concat([atts, init_state], -1)

        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            rnn_cell.BasicLSTMCell(atten_size),
            attention_mechanism,
            attention_layer_size=atten_size,
            cell_input_fn=answer_ptr_input_fn
        )

        logits, _ = tf.nn.static_rnn(attention_cell, labels, dtype=tf.float32)

        logits_start, logits_end = logits
        masked_logits_start, probdist_start = masked_softmax(logits_start, context_mask, 1)
        masked_logits_end, probdist_end = masked_softmax(logits_end, context_mask, 1)


        return masked_logits_start, masked_logits_end, probdist_start, probdist_end
