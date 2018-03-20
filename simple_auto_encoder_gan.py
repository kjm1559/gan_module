
import numpy as np

import pandas as pd

import tensorflow as tf




class gan_ae:

    def __init__(self, batch_size = 256, num_input = 35, learning_rate = 0.0001):

    

        self.batch_size = batch_size

        self.num_input = num_input

        self.learing_rate = learning_rate




        tf.reset_default_graph()




        config = tf.ConfigProto() 

        config.gpu_options.allow_growth = True 




        #self.y_data = tf.placeholder(tf.float32, [None, num_input], name='y_data')

        #self.x_data = tf.placeholder(tf.float32, [None, num_input], name='x_data')

        #self.z_data = tf.placeholder(tf.float32, [None, num_input], name='z_data')

        self.x_data = tf.placeholder(tf.float32, [None, num_input], name='x_data')




        self.filter = {

            'conv_1': tf.zeros([int(int(self.x_data.shape[1]) / 2), 1, 8]),

            'conv_2': tf.zeros([int(int(self.x_data.shape[1]) / 4), 8, 16])

        }

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob") 




        self.generator_x, self.g_params = self.generator(self.x_data)

        self.discriminator_y, self.discriminator_y_generated, self.d_params = self.discriminator(self.x_data, self.generator_x, self.keep_prob)




        

        with tf.name_scope('d_xent'):

            self.d_loss = -(tf.log(self.discriminator_y) + tf.log(1 - self.discriminator_y_generated))

            tf.summary.histogram('d_ent', self.d_loss)

            tf.summary.scalar('d_xent', tf.reduce_mean(self.discriminator_y))

        

        with tf.name_scope('g_xent'):

            self.g_loss = -tf.log(self.discriminator_y_generated) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.generator_x, labels=self.x_data))#(tf.reduce_sum(tf.abs(self.generator_x - self.x_data)) / 35)

            tf.summary.histogram('g_ent', self.g_loss)

            tf.summary.scalar('g_xent', tf.reduce_mean(self.discriminator_y_generated))

        

#         self.y_pred = self.decoder_op

#         self.y_true = self.x




#         self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_true))

        # loss = tf.reduce_mean(tf.pow(y_pred-y_true, 2))

        #self.optimizer = tf.train.AdamOptimizer(self.learing_rate).minimize(self.loss)

        with tf.name_scope('train'):

            self.optimizer = tf.train.AdamOptimizer(self.learing_rate)




            self.d_trainer = self.optimizer.minimize(self.d_loss, var_list = self.d_params)

            self.g_trainer = self.optimizer.minimize(self.g_loss, var_list = self.g_params)

        

        self.merged_summary = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter('./gan_summary/')

        

        self.h1_size = 150

        self.h2_size = 300




    def generator(self, z, name='ge'):

        with tf.name_scope(name):

            # Fully Connected Layer 1 (100 (latent-vector) -> 150 (h1_size))

            w1 = tf.Variable(tf.truncated_normal([self.num_input, 16], stddev=0.1), name="g_w1", dtype=tf.float32)

            b1 = tf.Variable(tf.zeros([16]), name="g_b1", dtype=tf.float32)

            h1 = tf.nn.relu(tf.matmul(z, w1) + b1)




            # Fully Connected Layer 2 (150 (h1_size) -> 300 (h2_size))

            w2 = tf.Variable(tf.truncated_normal([16, 8], stddev=0.1), name="g_w2", dtype=tf.float32)

            b2 = tf.Variable(tf.zeros([8]), name="g_b2", dtype=tf.float32)

            h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

            

            w3 = tf.Variable(tf.truncated_normal([8, 16], stddev=0.1), name="g_w3", dtype=tf.float32)

            b3 = tf.Variable(tf.zeros([16]), name="g_b3", dtype=tf.float32)

            h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)




            # Fully Connected Layer 3 (300 (h2_size) -> input_height * input_width (img_size))

            w4 = tf.Variable(tf.truncated_normal([16, self.num_input], stddev=0.1), name="g_w4", dtype=tf.float32)

            b4 = tf.Variable(tf.zeros([self.num_input]), name="g_b4", dtype=tf.float32)

            h4 = tf.matmul(h3, w4) + b4

            x_generate = tf.nn.tanh(h4)




#             x_in = tf.reshape(z, shape=[self.batch_size, int(z.shape[1]), 1])

            

#             # Convolution Layer1

#             c1 = tf.Variable(tf.nn.conv1d(x_in, self.filter['conv_1'], stride = 2, padding='VALID'), name='g_conv1', dtype=tf.float32)




#             # Convolution Layer2

#             c2 = tf.Variable(tf.nn.conv1d(c1, self.filter['conv_2'], stride = 2, padding='VALID'), name='g_conv2', dtype=tf.float32)

            

#             # Fully Connected Layer1

#             fc1 = tf.contrib.layers.flatten(c2)

#             w1 = tf.Variable(tf.truncated_normal([int(fc1.shape[1]), int(z.shape[1])], stddev=0.1), name='g_w1', dtype=tf.float32)

#             b1 = tf.Variable(tf.zeros([z.shape[1]]), name='g_b1', dtype=tf.float32)

#             h1 = tf.nn.relu(tf.matmul(fc1, w1) + b1)

#             x_generate = tf.nn.tanh(h1)

#             layer_1 = tf.nn.relu(tf.matmul(x, self.weights['encoder_h1']))

#             layer_2 = tf.nn.relu(tf.matmul(layer_1, self.weights['encoder_h2']))

#             layer_3 = tf.nn.relu(tf.matmul(layer_2, self.weights['encoder_h3']))

#             layer_4 = tf.nn.relu(tf.matmul(layer_3, self.weights['encoder_h4']))

#             layer_5 = tf.nn.relu(tf.matmul(layer_4, self.weights['encoder_h5']))

#             layer_6 = tf.nn.relu(tf.matmul(layer_5, self.weights['encoder_h6']))

            g_params = [w1, b1, w2, b2, w3, b3, w4, b4]

            return x_generate, g_params




    def discriminator(self, x, x_generated, keep_prob, name = 'dc'):

        with tf.name_scope(name):

            x_in = tf.concat([x, x_generated], 0)

            

            #w1 = tf.Variable(tf.truncated_normal([self.num_input, self.h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)

            w1 = tf.Variable(tf.truncated_normal([self.num_input, 32], stddev=0.1), name="d_w1", dtype=tf.float32)

            #b1 = tf.Variable(tf.zeros([self.h2_size]), name="d_b1", dtype=tf.float32)

            b1 = tf.Variable(tf.zeros([32]), name="d_b1", dtype=tf.float32)

            h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)




            #w2 = tf.Variable(tf.truncated_normal([self.h2_size, self.h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)

            w2 = tf.Variable(tf.truncated_normal([32, 16], stddev=0.1), name="d_w2", dtype=tf.float32)

            #b2 = tf.Variable(tf.zeros([self.h1_size]), name="d_b2", dtype=tf.float32)

            b2 = tf.Variable(tf.zeros([16]), name="d_b2", dtype=tf.float32)

            h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)




            #w3 = tf.Variable(tf.truncated_normal([self.h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)

            w3 = tf.Variable(tf.truncated_normal([16, 1], stddev=0.1), name="d_w3", dtype=tf.float32)

            b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)

            h3 = tf.matmul(h2, w3) + b3

            

            y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [self.batch_size, -1], name=None))

            y_generated = tf.nn.sigmoid(tf.slice(h3, [self.batch_size, 0], [-1, -1], name=None))




            d_params = [w1, b1, w2, b2, w3, b3]




            return y_data, y_generated, d_params

#             layer_1 = tf.nn.relu(tf.matmul(x, self.weights['decoder_h1']))

#             layer_2 = tf.nn.relu(tf.matmul(layer_1, self.weights['decoder_h2']))

#             layer_3 = tf.nn.relu(tf.matmul(layer_2, self.weights['decoder_h3']))

#             layer_4 = tf.nn.relu(tf.matmul(layer_3, self.weights['decoder_h4']))

#             layer_5 = tf.nn.relu(tf.matmul(layer_4, self.weights['decoder_h5']))

#             layer_6 = tf.matmul(layer_5, self.weights['decoder_h6'])

#             return layer_6

        

    def train(self, tmp_train_list, model_name):

        #saver = tf.train.Saver(self.weights)

        saver = tf.train.Saver()

        

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)




        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        sess.run(init, feed_dict={self.x_data: np.zeros((self.batch_size, int(self.x_data.shape[1]))), self.keep_prob: np.sum(0.5).astype(np.float32)})




        self.writer.add_graph(sess.graph)

        

        train_len = len(tmp_train_list)

        total_batch = int(train_len / self.batch_size)

        for epoch in range(100):

            total_cost = 0




            for i in range(total_batch):

                batch_xs = np.array(tmp_train_list[i*self.batch_size:(i+1)*self.batch_size]).astype(np.float32)

                #batch_zs = np.array(tmp_train_list[i*self.batch_size:(i+1)*self.batch_size]).astype(np.float32)

                #try:

                if i % 5:

                    summary = sess.run(self.merged_summary, feed_dict={self.x_data: batch_xs, self.keep_prob: np.sum(0.5).astype(np.float32)})

                    self.writer.add_summary(summary, epoch)

                #except:

                #    print(sys.exc_info)

                

                

                #if i % 10 == 0:

                sess.run(self.d_trainer, feed_dict={self.x_data: batch_xs, self.keep_prob: np.sum(0.5).astype(np.float32)})

                

                sess.run(self.g_trainer, feed_dict={self.x_data: batch_xs, self.keep_prob: np.sum(0.5).astype(np.float32)})

                

                #_, loss_val = sess.run([self.optimizer, self.loss], feed_dict={self.x: batch_xs})

                #total_cost += loss_val

                

            saver.save(sess, './' + model_name + '_model.ckpt')

            #print('Epoch:', '%04d' % (epoch + 1), 'generator loss = ', '{:.6f}'.format(self.g_loss), 'discriminator loss = ', '{:.6f}'.format(self.d_loss))

            print('Epoch:', '%04d' % (epoch + 1))

        sess.close()

            

    def test(self, no_vec_data, model_name):

        normal_loss = []




        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        saver = tf.train.Saver()

        saver.restore(sess, './' + model_name + '_model.ckpt')




        for tmp_x in no_vec_data:

            x_gen_val = sess.run(self.generator_x, feed_dict={self.x_data: [tmp_x]})

            #normal_loss.append(tf.reduce_mean(tf.squared_difference(tmp_x - x_gen_val)))

            normal_loss.append(x_gen_val)




        sess.close()

        return np.array(normal_loss)