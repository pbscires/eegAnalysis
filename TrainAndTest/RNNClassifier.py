import numpy as np
import tensorflow as tf

class LSTMModel(object):
    def __init__(self, input, is_training, num_steps, num_channels, dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = 512
        self.num_channels = num_channels
        self.hidden_size = 100
        self.num_layers = 1
        self.num_steps = num_steps
        
        if self.is_training and dropout < 1:
            self.input_obj = tf.nn.dropout(self.input_obj, dropout)
        
        self.init_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.hidden_size])

        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
            for idx in range(self.num_layers)]
        )

        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0)
        if self.is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if self.num_layers>1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.num_layers)], state_is_tuple=True)
        
        output, self.state = tf.nn.dynamic_rnn(cell, self.input_obj, dtype=tf.float32, initial_state=rnn_tuple_state)
        output = tf.reshape(output, [-1, self.hidden_size])

        softmax_w = tf.Variable(tf.random_uniform([self.hidden_size, self.num_channels], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([self.num_channels], -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.num_channels])

        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch = False
        )
        self.cost = tf.reduce_sum(loss)

        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, self.num_channels]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not self.is_training:
            return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr:lr_value})

def train(train_data, num_channels, num_layers, num_epochs, batch_size, model_save_name, learning_rate=1.0, max_lr_epoch = 10, lr_decay=0.93):
    m = LSTMModel(train_data, num_steps = 21, is_training=True, num_channels=num_channels)
    init_op = tf.global_variables_initializer()
    
    orig_decay = lr_decay
    with tf.Session() as sess:
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()

        for epoch in range(num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
            for step in range()