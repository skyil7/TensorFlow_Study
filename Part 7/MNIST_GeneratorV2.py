import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# hyperparameters
total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28 * 28
n_noise = 128
n_class = 10    #one_hot 레이블이 0~10 이므로

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])  # hint
Z = tf.placeholder(tf.float32, [None, n_noise])  # noise

def generator(noise, labels):
    with tf.variable_scope('generator'):
        inputs = tf.concat([noise, labels], 1)  #noise 값에 label 정보를 달아서 생성
        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, n_input, activation=tf.nn.sigmoid)

    return output

def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:   #변수 재활용, 아래 discriminator 생성 부분보면 이해 딱 됨
        if reuse:
            scope.reuse_variables()
        inputs = tf.concat([inputs, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, 1, activation=None)

    return output

def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=[batch_size, n_noise])

G = generator(Z, Y)
D_real = discriminator(X,Y)
D_gene = discriminator(G,Y,True)

loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))   #1 쪽으로
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene)))  #0 쪽으로
loss_D = loss_D_real + loss_D_gene

loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))        #만든것도 1해라

#스코프로부터 변수들 가져오기
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

train_D = tf.train.AdamOptimizer().minimize(loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer().minimize(loss_G, var_list=vars_G)

################################################################
#학습 ㄱㄱㄱㄱㄱ
################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G =0,0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _,loss_val_D = sess.run([train_D, loss_D], {X:batch_xs, Y:batch_ys, Z:noise})
        _,loss_val_G = sess.run([train_G, loss_G], {Y:batch_ys, Z:noise})

    print('Epoch:','%04d' %(epoch+1)+'/'+str(total_epoch), 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))

    #생성된 가짜 이미지 저장
    if epoch == 0 or (epoch+1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Y:mnist.test.labels[:sample_size], Z:noise})

        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
            ax[1][i].imshow(np.reshape(samples[i], (28,28)))

        plt.savefig('samples2/{}.png'.format(str(epoch+1).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료')