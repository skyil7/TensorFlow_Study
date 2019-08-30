import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN

tf.app.flags.DEFINE_boolean('train', False, "학습모드, 게임을 화면에 표시하지 않음")
FLAGS = tf.app.flags.FLAGS

#Hyperparameters
MAX_EPISODE = 10000             # 최대로 학습할 게임의 판수
TARGET_UPDATE_INTERVAL = 1000   # 목표 신경망 갱신 주기
TRAIN_INTERVAL = 4              # 4 프레임당 1회 학습
OBSERVE = 100                   # 100 프레임이 지난 후부터 학습 개시

#취할 수 있는 액션
NUM_ACTION = 3  # 0:좌, 1:유지, 2:우
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10

################################################################
### 학습 구간
################################################################

def train():
    print('학습')
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.',tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    brain.update_target_network()
    epsilon = 1.0               #행동에서 DQN의 값을 사용할 시점. 초반에는 높다가 후반으로 가며 떨어져야 한다.(초반에는 한가지 행동만 반복할 확률이 높다.)

    time_step = 0
    total_reward_list = []

    for episode in range(MAX_EPISODE):
        terminal = False        # 게임의 상태(False는 진행중, True면 게임 끝)
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        while not terminal:     # 게임중이면
            if np.random.rand() < epsilon:
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()

            if episode > OBSERVE:
                epsilon -= 1/1000

            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                brain.train()

            if time_step % TARGET_UPDATE_INTERVAL == 0:
                brain.update_target_network()

            time_step += 1

        print('게임횟수: %d 점수: %d' %(episode+1, total_reward))

        total_reward_list.append(total_reward)

        if episode % 10 == 0:
            summary = sess.run(summary_merged, {rewards:total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode % 100 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)

################################################################
### 게임 구간
################################################################

def replay():
    print('게임')
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game = True)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(MAX_EPISODE):
        terminate = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        while not terminate:
            action = brain.get_action()
            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            #게임 프레임
            time.sleep(0.3)

        print('게임횟수: %d 점수: %d' %(episode+1, total_reward))

def main(_):
    if FLAGS.train:
        train()
    else:
        replay()

if __name__ == '__main__':
    tf.app.run()