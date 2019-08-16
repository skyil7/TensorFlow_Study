import tensorflow as tf
#플레이스홀더는 그래프에 사용할 입력값을 나중에 받기 위해 사용하는 매개변수이다.
#변수는 그래프를 최적화하는 용도로 텐서플로의 학습 함수들이 학습한 결과를 갱신하기 위해 사용하는 변수들이다.

#None은 크기가 정해지지 않음을 뜻함
X = tf.placeholder(tf.float32, [None, 3])
print(X)    #Tensor("Placeholder:0", shape=(?, 3), dtype=float32)
#값을 3개씩 가지는 배열들로 X를 채울 수 있을 것이다.
x_data = [[1,2,3],[4,5,6]]  #이따가 이걸로 채우자.

W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))
#각각 [3,2], [2,1]형태의 행렬로 초기화 한다.
#random_normal 함수는 변수를 정규분포의 무작위 값으로 초기화한다.
#W = tf.Variable([[0.1,0.1],[0.2,0.2],[0.3,0.3]]) 이렇게 수동으로 초기화도 가능하다.

expr = tf.matmul(X,W) + b
#행렬곱 계산하고 편향을 더해주는 기본 신경망 공식이다.

#이제 세션에서 돌려보자.

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#앞에 정의한 변수들의 초기화를 실행한다.

print("===X_data===")
print(x_data)
print('=== W ===')
print(sess.run(W))
print('=== b ===')
print(sess.run(b))
print('===expr===')
print(sess.run(expr, feed_dict={X:x_data}))
#feed_dict로 사용할 입력값을 지정한다.

sess.close()