import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#Hyperparameters
total_epoch = 100
batch_size = 100
learning_rate = 0.1
n_hidden = 256
n_input = [None, 28, 28, 1] #28 x 28 회색조 이미지를 Vector로 N장
n_class = [None, 10]    #one_hot 0~9
n_noise = [None, 12, 12, 1] #12 x 12 회색조 이미지 형식 노이즈

X = tf.placeholder(tf.float32, n_input, name='X')
Y = tf.placeholder(tf.float32, n_class, name='Y')  # hint
Z = tf.placeholder(tf.float32, n_noise, name='Z')  # noise

def generator(noise):
    with tf.variable_scope('generator'):
        L1 = tf.layers.conv2d(noise, 32, [3, 3], activation=tf.nn.relu, padding='SAME')
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2], padding='SAME')
        L2 = tf.layers.conv2d(L1, 64, [3, 3], activation=tf.nn.relu, padding='SAME')
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2], padding='SAME')
        L3 = tf.contrib.layers.flatten(L2)
        L3 = tf.layers.dense(L3, 28*28, activation=tf.nn.relu)
    output = tf.reshape(L3, [-1, 28, 28, 1])
    return output

def discriminator(inputs, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        L1 = tf.layers.conv2d(inputs, 32, [3, 3], activation=tf.nn.relu, padding='SAME')
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2], padding='SAME')
        L2 = tf.layers.conv2d(L1, 64, [3, 3], activation=tf.nn.relu, padding='SAME')
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2], padding='SAME')
        L3 = tf.contrib.layers.flatten(L2)
        L3 = tf.layers.dense(L3, 1, activation=tf.nn.relu)

    return L3

def get_noise(batch_size, n_noise):
    bs = n_noise
    bs[0] = batch_size
    return np.random.uniform(-1., 1., size=bs)

#학습
G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X, True)

#Loss Functions
#두 손실값이 모두 최대가 되면 좋겠지만, 서로 대립 관계이므로 그럴일은 없다.
loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))   #1 쪽으로
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene)))  #0 쪽으로
loss_D = loss_D_real + loss_D_gene

loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))        #만든것도 1해라

#스코프로부터 변수들 가져오기
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_G, var_list=vars_G)

#Training Start
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], {X:batch_xs, Z:noise})
        _, loss_val_G = sess.run([train_G, loss_G], {Z:noise})

    print('Epoch:','%04d' %(epoch+1)+'/'+str(total_epoch), 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))

    # 생성된 가짜 이미지 저장
    if epoch == 0 or (epoch+1)%10 ==0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, {Z:noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28,28)))

        plt.savefig('samples2/{}.png'.format(str(epoch+1).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료')