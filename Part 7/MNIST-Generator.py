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

X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])  # noise

#생성자 신경망 변수
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

#구분자 신경망 변수
#구분자는 진짜와 얼마나 가까운지를 0~1 사이의 실수로 출력, output은 한개의 스칼라값
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

#생성자 신경망
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1)+G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2)+G_b2)
    return output

#구분자 신경망
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1)+D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2)+D_b2)
    return output

#노이즈 생성 함수
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size,n_noise))

#학습
G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

#Loss Functions
#두 손실값이 모두 최대가 되면 좋겠지만, 서로 대립 관계이므로 그럴일은 없다.
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

#각각의 신경망을 학습시킬 때, 다른 신경망의 변수를 건드리면 안된다.
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

#Optimizer Setting
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

#Training Start
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
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

        plt.savefig('samples/{}.png'.format(str(epoch+1).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료')