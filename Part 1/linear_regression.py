import tensorflow as tf
import random
#이제 간단한 선형회귀 모델을 구현해보자. 선형회귀에 대한 설명은 README.md에서 하겠다.

x_data=[1,2,3]
y_data=[3,5,7]#y=2x+1

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
#여기서 이름을 지정하면 X를 그냥 프린트했을 때 이름이 나와서, 디버그 할 때 편하다.

hypothesis = W*X+b
#선형관계 분석을 위한 수식, W와 X가 이번엔 스칼라 값이므로 행렬곱을 하지 않아도 괜찮다.

cost = tf.reduce_mean(tf.square(hypothesis - Y))
#손실 함수(loss function)으로 스퀘어 함수를 사용한 비용 계산

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)
#경사하강법(gradient descent) 최적화 함수를 통해 cost를 최소화하는 학습 수행
#learning_rate는 학습을 얼마나 정확하게 할 것인지를 정함.
#너무 크면 최적값을 못 찾고 지나치고, 너무 작으면 정확하지만 너무 느려짐
#이런 학습에 영향을 주는 변수를 하이퍼파라미터라고 한다.

with tf.Session() as sess:#끝나면 알아서 세션 닫음
    sess.run(tf.global_variables_initializer())

    for step in range(4000):#학습 4000번 하면서 감소하는 cost 출력
        _, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y:y_data})
        print(step, cost_val, sess.run(W), sess.run(b))

    print('===TEST===')
    print('X:4, Y:',sess.run(hypothesis, feed_dict={X:4}))
    print('X:2.5, Y:',sess.run(hypothesis, feed_dict={X:2.5}))