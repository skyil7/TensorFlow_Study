import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
#원 핫 인코딩 아시죠?

global_step = tf.Variable(0, trainable=False, name='global_step') #학습횟수 count용 변수

X = tf.placeholder(tf.float32, [None, 784])#28*28 아시죠?
#위의 None에 한번에 학습시킬 사이즈를 지정하면 된다.
#지정 안하면 TensorFlow가 알아서 계산해 준다.
Y = tf.placeholder(tf.float32, [None, 10])

#신경망 설계를 해보자.
#784 -> 256 -> 256 -> 10 으로 하자.

keep_prob = tf.placeholder(tf.float32)#학습시에만 드롭아웃이 일어나도록 해준다.

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([784,256], stddev=0.01))#표준편차 0.01 정규분포
    L1 = tf.nn.relu(tf.matmul(X,W1))
    #L1 = tf.nn.dropout(L1, keep_prob)
    tf.layers.batch_normalization(L1, training=(keep_prob!=1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([256,256], stddev=0.01))
    L2 = tf.nn.relu(tf.matmul(L1,W2))
    #L2 = tf.nn.dropout(L2, keep_prob)
    tf.layers.batch_normalization(L2, training=(keep_prob!=1))

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_normal([256,10], stddev=0.01))
    model = tf.matmul(L2,W3) #출력층에는 활성화 함수를 사용안함

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step)

# 손실값 추적을 위한 cost 추적 코드
tf.summary.scalar('cost', cost)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

#앞서 지정한 텐서들을 수집하고 저장하는 부분
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs',sess.graph)

batch_size = 5500#한번에 학습시킬 배치 크기
total_batch = int(mnist.train.num_examples/batch_size)#배치로 나눈 전체 데이터 크기 (55000/5500)

for epoch in range(15):
    total_cost=0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.8})
        total_cost+=cost_val
    summary = sess.run(merged, feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.8})
    writer.add_summary(summary, global_step=sess.run(global_step))
    print('Epoch:', '%04d' %(epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost/total_batch))

print('Optimize Complete!')

saver.save(sess, './model/dnn.ckpt', global_step=global_step)

labels = sess.run(model, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1})
fig = plt.figure()

for i in range(10,20):
    #2열 5행짜리 그래프 생성 후 i+1 숫자 출력
    subplot = fig.add_subplot(2,5,i-9)
    #눈금 끄기
    subplot.set_xticks([])
    subplot.set_yticks([])
    #예측값
    subplot.set_title('%d' %np.argmax(labels[i]))
    #이미지
    subplot.imshow(mnist.test.images[i].reshape((28,28)),cmap=plt.cm.gray_r)

plt.show()

sess.close()