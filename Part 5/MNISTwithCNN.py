import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#Hyperparam
learning_rate=0.01
batch_size = 500
epoch = 30

X = tf.placeholder(tf.float32, [None, 28, 28, 1])#28*28 이미지, 1은 특징의 개수로, MNIST에서는 회색조 이미지라 색상이 하나 뿐이므로 1
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))#3*3 짜리 가중치
    L1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding='SAME')#2d Convolution
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
    #28*28 이미지를 가지는 뉴런이 32개 -> 풀링으로 반감 14*14*32

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01)) #이번엔 64개로 뉴런을 늘리자
    L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #14*14 뉴런을 64개로 늘리고 다시 풀링 -> 7*7*64

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_normal([7*7*64, 256],stddev=0.01))#출력을 위해 차원을 줄이고, 256개의 뉴런에 풀커넥션
    L3 = tf.reshape(L2, [-1, 7*7*64])
    L3 = tf.matmul(L3,W3)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob)

with tf.name_scope('layer4'):
    W4 = tf.Variable(tf.random_normal([256,10],stddev=0.01))
    model = tf.matmul(L3,W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#나중에 아래 옵티마이저로도 해보자.
#optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

#####
#학습
#####

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

for e in range(epoch):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1,28,28,1) #MNIST 데이터를 28*28 형태로 재구성

        _,cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.7})

        total_cost+=cost_val

    print('Epoch:','%04d/%04d' %(e+1,epoch), 'Avg.cost =','{:3f}'.format(total_cost/total_batch))

print('Optimize Complete!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

print('정확도 :',sess.run(accuracy, feed_dict={X:mnist.test.images.reshape(-1,28,28,1), Y:mnist.test.labels, keep_prob:1}))

sess.close()