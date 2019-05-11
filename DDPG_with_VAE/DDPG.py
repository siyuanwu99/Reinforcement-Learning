import gym
import time
import random
import numpy as np
import pickle
import tensorflow as tf
from collections import deque

# 在MountainCar-v0 上面试验，效果不太好
# 暂时不清楚错误在哪里，貌似算法实践没有错误

### Hyper-param

MAX_EPISODE = 100
MAX_EP_STEP = 200
LR_A = 0.001  # learning rate of actor
LR_C = 0.001  # learning rate of critic
GAMMA = 0.9  # reward discount
TAU = 0.01
MEMORY_SIZE = 200
BATCH_SIZE = 64
ENV_NAME = "MountainCarContinuous-v0"

###


class DDPG(object):
    def __init__(self, dim_a, dim_s, action_bound):
        self.dim_s, self.dim_a, self.action_bound = dim_s, dim_a, action_bound

        self.pointer = 0
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.history_loss = []
        self.history_TD_error = []
        self.history_reward = []

        self.sess = tf.Session()

        # variable
        self.state = tf.placeholder(tf.float32, [None, dim_s], name="state")
        self.state_ = tf.placeholder(tf.float32, [None, dim_s], name="state_")
        # self.action = tf.placeholder(tf.float32, [None, 1], name="action")
        self.reward = tf.placeholder(tf.float32, [None, 1], name="reward")

        # parameters
        self.ae_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor/eval")
        self.at_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor/target")
        self.ce_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="critic/eval")
        self.ct_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="critic/target")

        #
        with tf.variable_scope("actor"):
            self.action = self._build_actor(self.state, scope="eval", trainable=True)
            self.action_ = self._build_actor(self.state_, scope="target", trainable=False)

        with tf.variable_scope("critic"):
            self.q_value = self._build_critic(self.state, self.action, scope="eval", trainable=True)
            self.q_value_ = self._build_critic(self.state_, self.action_, scope="target", trainable=False)

        # loss
        with tf.name_scope('loss'):
            q_target = self.reward + GAMMA * self.q_value_
            self.TD_error = tf.losses.mean_squared_error(labels=self.q_value, predictions=q_target)
            self.loss = - tf.reduce_mean(self.q_value)

        with tf.variable_scope("summaries"):
            tf.summary.scalar('TD', self.TD_error)
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Reward', self.history_reward)

        # train actor
        with tf.name_scope('train'):
            self.actor_train = tf.train.AdamOptimizer(LR_A).minimize(self.loss)
            self.critic_train = tf.train.AdamOptimizer(LR_C).minimize(self.TD_error)

        self.sess.run(tf.global_variables_initializer())

        # tensorboard
        # merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/", self.sess.graph)
        print("Wrote Tensorboard Logs")
        writer.close()

    def _build_actor(self, state, scope, trainable):
        with tf.variable_scope(scope):
            n_layer1 = 2 * self.dim_s
            n_layer2 = self.dim_s
            W1 = tf.Variable(tf.truncated_normal(shape=[self.dim_s, n_layer1], stddev=1), name="W1", trainable=trainable)
            b1 = tf.Variable(tf.constant(0., shape=[n_layer1]), name="b1", trainable=trainable)
            net = tf.nn.relu(tf.matmul(state, W1) + b1)

            W2 = tf.Variable(tf.truncated_normal(shape=[n_layer1, n_layer2], stddev=1), name="W2", trainable=trainable)
            b2 = tf.Variable(tf.constant(0., shape=[n_layer2]), name="b2", trainable=trainable)
            net = tf.nn.relu(tf.matmul(net, W2) + b2)

            W3 = tf.Variable(tf.truncated_normal(shape=[n_layer2, 1], stddev=1), name="W3", trainable=trainable)
            b3 = tf.Variable(tf.constant(0., shape=[1]), name="b3", trainable=trainable)
            action = tf.nn.relu(tf.matmul(net, W3) + b3)

            return tf.multiply(action, self.action_bound)

    def _build_critic(self, state, action, scope, trainable):
        with tf.variable_scope(scope):
            n_layer1 = 2 * self.dim_s + self.dim_a
            n_layer2 = round(n_layer1 / 3)
            W1 = tf.Variable(tf.truncated_normal(shape=[self.dim_s, n_layer1], stddev=1), name="W1", trainable=trainable)
            b1 = tf.Variable(tf.constant(0., shape=[n_layer1]), name="b1", trainable=trainable)
            W2 = tf.Variable(tf.truncated_normal(shape=[self.dim_a, n_layer1], stddev=1), name="W2", trainable=trainable)
            b2 = tf.Variable(tf.constant(0., shape=[n_layer1]), name="b2", trainable=trainable)
            W3 = tf.Variable(tf.truncated_normal(shape=[n_layer1, n_layer2], stddev=1), name="W3", trainable=trainable)
            b3 = tf.Variable(tf.constant(0., shape=[n_layer2]), name="b3", trainable=trainable)
            W4 = tf.Variable(tf.truncated_normal(shape=[n_layer2, 1], stddev=1), name="W4", trainable=trainable)
            b4 = tf.Variable(tf.constant(0., shape=[1]), name="b4", trainable=trainable)

            net = tf.nn.relu(tf.matmul(state, W1) + b1) + tf.nn.relu(tf.matmul(action, W2) + b2)
            net = tf.nn.relu(tf.matmul(net, W3) + b3)
            net = tf.nn.relu(tf.matmul(net, W4) + b4)

        return net

    def choose_actions(self, state):
        return self.sess.run(self.action, feed_dict={self.state: np.transpose(state)})

    def store_transition(self, state, action, reward, state_):
        trans = [state, action, reward, state_]
        self.memory.append(trans)
        # if self.pointer < MEMORY_SIZE:
        #     self.memory.append(trans)
        # else:
        #     i = self.pointer % MEMORY_SIZE
        #     self.memory[i] = trans

        self.pointer += 1

    def learn(self):
        # print(self.memory)
        indices = random.choices(self.memory, k=BATCH_SIZE)
        # indices = np.random.choice(self.memory[:], BATCH_SIZE)
        s, a, r, s_ = ([row[index] for row in indices] for index in range(4))
        # print(np.shape(s))

        s = np.array(s)[:,:,0]
        s_ = np.array(s_)[:,:,0]
        a = np.array(a)[:,:,0]
        r = np.array(r)[:, np.newaxis]

        self.sess.run([self.actor_train, self.critic_train],
                      feed_dict={self.state: s,
                                 self.reward: r,
                                 self.action: a,
                                 self.state_: s_})

        self._update()
        self.history_loss.append(self.loss)
        self.history_TD_error.append(self.history_TD_error)


    def _update(self):

        self.at_param = [tf.assign(t, (1 - TAU) * t + TAU * e)
                         for t, e in zip(self.at_param, self.ae_param)]
        self.ct_param = [tf.assign(t, (1 - TAU) * t + TAU * e)
                         for t, e in zip(self.ct_param, self.ce_param)]

        # self.at_param = tf.assign(self.at_param,
        #                 TAU * self.ae_param + (1 - TAU) * self.at_param)
        # self.ct_param = tf.assign(self.ct_param,
        #                 TAU * self.ce_param + (1 - TAU) * self.ct_param)
    def graph(self):
        import matplotlib.pyplot as plt
        # plt.subplot(1, 3, 1)
        # plt.figure(1)
        # plt.plot(np.arange(len(self.history_reward)), self.history_reward)
        # plt.ylabel('Cost')
        # plt.xlabel('training steps')
        # plt.show()

        # plt.subplot(1, 3, 2)
        # plt.plot(np.arange(len(self.history_TD_error)), self.history_TD_error)
        # plt.ylabel('TD')
        # plt.xlabel('training steps')
        # plt.show()

        # plt.subplot(1, 3, 3)
        plt.plot(np.arange(len(self.history_loss)), self.history_loss)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.show()


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    # env = env.unwrapped
    # env.seed(1)

    dim_s = env.observation_space.shape[0]
    dim_a = env.action_space.shape[0]
    a_h = env.action_space.high
    # a_l = env.action_space

    brain = DDPG(dim_a, dim_s, a_h)

    # env.reset()
    # state, _, _,_ = env.step(2)
    # print(state)
    # assert 0

    var = 3
    t_0 = time.time()
    for i in range(MAX_EPISODE):
        s = env.reset()
        s = s[:, np.newaxis]

        ep_reward = 0
        for j in range(MAX_EP_STEP):

            env.render()

            a = brain.choose_actions(s)
            a = np.clip(np.random.normal(a, var), 0, a_h)

            s_, r, done, info = env.step(a)

            # print(np.shape(s_))
            # s_ = np.squeeze(s_, axis=0)

            brain.store_transition(s, a, r, s_)
            # print('\n', brain.pointer)

            if brain.pointer > MEMORY_SIZE :
                # print(brain.pointer) % MEMORY_SIZE
                var *= 0.995
                brain.learn()

            s = s_

            # print('action:', a, 'next_state', s_)
            ep_reward += r
            print('Step:', brain.pointer, '\tReward', r,
                  '\tExplore: %.2f' % var)

        brain.history_reward.append(ep_reward)
        print('\tEpisode:', i, '\tReward: %i' % int(ep_reward),
              'Explore: %.2f' % var)
    brain.graph()
    print("Running time: ", time.time() - t_0)





















