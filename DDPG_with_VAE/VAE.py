import numpy as np
import tensorflow as tf
import time
import json

class VAE(object):
    def __init__(self, z_size= 512, batch_size= 100,
                 learning_rate= 0.0001, kl_tolerance= 0.5,
                 is_training= True, reuse= False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.kl_t = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse
        with tf.variable_scope('Conv_VAE',reuse = self.reuse):
            tf.logging.info("Start with GPU mode")
            self._build_graph()
        self._init_sess()

        # tf.summary.FileWriter("logs/", graph= self.sess.graph)

    def _build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, 80, 160, 3], name='X')
            print("X_shape\t", self.x.shape)

            # Encoder
            h = tf.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name="encoder_conv1")
            print("H_shape\t", h.shape)
            h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="encoder_conv2")
            print("H_shape\t", h.shape)
            h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="encoder_conv3")
            h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="encoder_conv4")
            print("H_shape\t", h.shape)
            h = tf.reshape(h, [-1, 3*8*256])
            print("H_shape\t", h.shape)

            # VAE
            self.mu = tf.layers.dense(h, self.z_size, name='enc_fc_mu')
            self.logvar = tf.layers.dense(h, self.z_size, name='enc_fc_logvar')
            self.sigma = tf.exp(0.5 * self.logvar)
            self.epsilon = tf.random_normal([self.batch_size, self.z_size])
            self.z = self.mu + self.sigma * self.epsilon

            print("Z_shape\t", self.z.shape)

            #Decoder
            h = tf.layers.dense(self.z, 3*8*256, name='decode_fc')
            # print("H_shape\t", h.shape)
            h = tf.reshape(h, [-1, 3, 8, 256])
            # print("H_shape\t", h.shape)
            h = tf.layers.conv2d_transpose(h, 128, 4, strides=2, activation=tf.nn.relu, name="decoder_conv2")
            h = tf.layers.conv2d_transpose(h, 64, 4, strides=2, activation=tf.nn.relu, name="decoder_conv3")
            h = tf.layers.conv2d_transpose(h, 32, 5, strides=2, activation=tf.nn.relu, name="decoder_conv4")
            self.y = tf.layers.conv2d_transpose(h, 3, 4, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")

            print("y_shape\t", self.y.shape)

            # train
            if self.is_training:
                # self.global_step = tf.Variable(0, trainable=True, name='Global_Step')
                # eps = 1e-6

                # reconstruction loss
                # self.rc_loss = tf.reduce_sum(tf.square(self.x - self.y), reduction_indices=[1,2,3]) # differences
                # print("self.rc_loss", self.rc_loss.shape)
                # self.rc_loss = tf.reduce_sum(self.rc_loss, reduction_indices=[1])
                self.rc_loss = tf.reduce_sum(tf.square(self.x - self.y))
                self.rc_loss = tf.reduce_mean(self.rc_loss)

                # kl loss
                self.kl_loss = -0.5 * tf.reduce_sum(1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar))
                self.kl_loss = tf.maximum(self.kl_loss, self.kl_t * self.z_size)
                self.kl_loss = tf.reduce_sum(self.kl_loss)

                self.loss = self.kl_loss + self.rc_loss

                # training
                self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss) # differences
                # This part is different from the original one

                # init
                self.init = tf.global_variables_initializer()

    def _init_sess(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def _close_sess(self):
        self.sess.close()

    def encode(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def decode(self, Z):
        return self.sess.run(self.y, feed_dict={self.z: Z})

    def get_model_params(self):
        '''
        get trainable model parameters
        :return:
        '''
        model_names = []
        model_shapes = []
        model_params = []
        with self.graph.as_default():
            t_vars = tf.trainable_variables()
            # print("t_vars\t", self.sess.run(t_vars))
            assert 1
            for var in t_vars:
                param_name = var.name  # 返回每个参量的name
                p = self.sess.run(var)  # eval
                model_names.append(param_name)
                params = np.round(p * 10000).astype(np.int).tolist() # 以整型形式存储，保留四位小数
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names
    def set_model_params(self, param):
        with self.graph.as_default():
            t_vars = tf.trainable_variables()
            i = 0
            for var in t_vars:
                pshape = self.sess.run(var).shape
                p = np.array(param[i])
                assert pshape == p.shape
                assign_op = var.assign(p.astype(np.float)/10000.)
                self.sess.run(assign_op)
                i += 1





    def load_json(self, jsonfile):
        with open(jsonfile) as f:
            params = json.load(f)
        self.set_model_params(params)

    def export_json(self, jsonfile):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []

        for p in model_params:
            qparams.append(p)

        with open(jsonfile, 'wt') as f:
            json.dump(qparams, f, sort_keys=True, indent=0, separators=(',', ': '))


if __name__ == '__main__':
    x = np.random.random(size=[100, 80, 160, 3])
    # print(x)
    RL_brain = VAE()
    Z = RL_brain.encode(X=x)
    y = RL_brain.decode(Z=Z)
    print(Z.shape)
    print(y.shape)
    # RL_brain.get_model_params()
    RL_brain.export_json("new.json")
    RL_brain.load_json("new.json")

