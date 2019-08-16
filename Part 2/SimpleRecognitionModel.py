import tensorflow as tf
import numpy as np

#털, 날개가 있냐를 기반으로 조류를 구분
#[털, 날개]
x_data = np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])
#결과값은 원-핫 인코딩으로 저장
y_data = np.array([
    [1,0,0],    #기타
    [0,1,0],    #포유류
    [0,0,1],    #조류
    [1,0,0],
    [1,0,0],
    [0,0,1]
])

X=tf.placeholder(tf.float32, name='X')
Y=tf.placeholder(tf.float32, name='Y')

W=tf.Variable(tf.random_uniform([2,3],-1,1))#[입력층,출력층] 균등분포
b=tf.Variable(tf.zeros([3]))#출력층과 같은 3개의 요소를 가진 편향

y=tf.add(tf.matmul(X,W),b)#합성곱하고 bias 더하는 식
L=tf.nn.relu(y)#relu를 activation func로 갖는 신경망
#신경망 구현끝

model = tf.nn.softmax(L)
#softmax 함수로 출력을 확률로

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model),axis=1))
#교차 엔트로피 함수
#실측값*예측값의 로그값을 한 후, 행별로 값을 모두 더해 평균을 냄

#이제 학습 ㄱㄱ
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})

    if (step+1)%10==0:
        print(step+1,sess.run(cost,feed_dict={X:x_data, Y:y_data}))

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:',sess.run(prediction, feed_dict={X:x_data}))
print('실제값:',sess.run(target,feed_dict={Y:y_data}))

is_correct = tf.equal(prediction,target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' %sess.run(accuracy*100, feed_dict={X:x_data, Y:y_data}))

sess.close()