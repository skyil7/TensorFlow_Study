import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQN:
    REPLAY_MEMORY = 10000       # 학습에 사용할 플레이 결과를 얼마나 많이 저장해서 사용할지
    BATCH_SIZE = 100            # 1회 학습에 사용할 기억의 갯수
    GAMMA = 0.99                # 오래된 상태의 가중치를 줄이기 위한 값
    STATE_LEN = 4               # 한번에 볼 프레임의 총 수(얼마나 과거까지 고려할 것인지)

    def __init__(self, session, width, height, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        self.memory = deque()   # 게임 결과를 저장할 메모리 생성
        self.state = None
        self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN])    # 게임 상태 입력
        self.input_A = tf.placeholder(tf.int64, [None])                                     # 각 상태를 만들어낸 액션의 값
        self.input_Y = tf.placeholder(tf.float32, [None])                                  # 손실값 계산에 사용할 값 입력(보상 + 목표신경망으로 구한 다음 상태의  Q값)

        self.Q = self._build_network('main')
        self.cost, self.train_op = self._build_op()

        self.target_Q = self._build_network('target')

    def _build_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4,4], padding='SAME', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2,2], padding='SAME', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)

            Q = tf.layers.dense(model, self.n_action, activation=None)

        return Q

    def _build_op(self):
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main")
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict={self.input_X:[self.state]})
        action = np.argmax(Q_value[0])

        return action

    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()
        target_Q_value = self.session.run(self.target_Q, feed_dict={self.input_X:next_state})
        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))

        self.session.run(self.train_op, feed_dict={self.input_X:state, self.input_A:action, self.input_Y:Y})

    def init_state(self, state):
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.stack(state, axis=2)

    def remember(self, state, action, reward, terminal):
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)

        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal