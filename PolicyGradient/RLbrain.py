import numpy as np 
import tensorflow as tf 
class PolicyGradient:
    def __init__(self, n_actions, n_states, learning_rate, reward_decay, output_graph=False):
        
        """
        n_actions   the shape of actions
        n_states:   the shape of states
        learning_rate:  learning rate
        reward_decay:   gamma, in learning process, to decay later reward
        output_graph:   False as custom
        """

        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = reward_decay

        self.list_state, self.list_action, self.list_reward = [], [], []
        
        self._build_net()

        self.sess = tf.Session()
        if output_graph:    
            tf.summary.FileWriter("PGlogs/", self.sess.graph())

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope("Inputs"):
            self.tf_state = tf.placeholder(tf.float32, [None, self.n_states], name="observations")
            self.tf_action = tf.placeholder(tf.int32, [None, ], name="action_choosed")
            self.tf_reward = tf.placeholder(tf.float32, [None, ], name="action_reward")

        n_layer1 = 10
        n_layer2 = self.n_actions

        with tf.name_scope("Layer1"):
            W1 = tf.get_variable(name='w1', dtype=tf.float32, shape=[self.n_states, n_layer1], 
                            initializer= tf.initializers.random_normal(mean=0, stddev=0.3))
            b1 = tf.get_variable(name='b1', dtype=tf.float32, shape=[1, n_layer1], 
                            initializer= tf.initializers.constant(value=1))

            l1 = tf.nn.relu(tf.matmul(self.tf_state, W1) + b1)

        with tf.name_scope("Layer2"):
            W2 = tf.get_variable(name='w2', dtype=tf.float32, shape=[n_layer1, n_layer2], 
                            initializer= tf.initializers.random_normal(mean= 0, stddev=0.3))
            b2 = tf.get_variable(name='b2', dtype=tf.float32, shape=[1, n_layer2],
                            initializer= tf.initializers.constant(value=1))

            l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)
        
        self.all_act_prob = tf.nn.softmax(l2, name="all_act_prob")

        with tf.name_scope("Loss"):
            neg_action_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= l2, labels=self.tf_action)
            loss = tf.reduce_mean(neg_action_prob * self.tf_reward)

        with tf.name_scope("Train"):
            self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        
    def learn(self):
        discount_reward = np.zeros_like(self.list_reward)
        running_add = 0
        for t in reversed(range(0, len(self.list_reward))):
            running_add = running_add * self.gamma + self.list_reward[t] # 后面的乘系数加到前面去
            discount_reward[t] = running_add # 怎么理解，前面的比后面的高？

        discount_reward -= np.mean(discount_reward)
        discount_reward /= np.std(discount_reward)

        self.sess.run(
            self.train,
            feed_dict={
                self.tf_action : np.array(self.list_action),
                self.tf_state : np.vstack(self.list_state), 
                self.tf_reward : discount_reward
            }
        )
        self.list_action, self.list_reward, self.list_state = [], [], []
    
    
    def choose(self, observation):
        prob_weight = self.sess.run(
            self.all_act_prob,
            feed_dict={
                    self.tf_state: observation[np.newaxis,:]
            }
        )
        action = np.random.choice(range(prob_weight.shape[1]), p=prob_weight.ravel())
        return action


    def transition(self, state, action, reward):
        """
        transition of state action and reward
        """
        self.list_action.append(action)
        self.list_reward.append(reward)
        self.list_state.append(state)



    