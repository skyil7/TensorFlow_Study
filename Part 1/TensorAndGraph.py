import tensorflow as tf

hello = tf.constant('Hello World')#tf.constant로 상수를 Tensor형 변수에 저장
print(hello)    #Tensor("Const:0", shape=(), dtype=string)
#랭크랑 셰이프 아시죠? 모르면 절.대.선.대.해.

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a,b)
print(c)    #Tensor("Add:0", shape=(), dtype=int32)

#아까부터 출력이 add니 const니 연산을 출력한다.
#이는 연산을 예약해놓고 특정 시점에 시행하는 텐서플로의 특성 지연 실행(lazy evaluation) 때문이다.

#즉, 위에서는 그래프를 생성한 것이고, 이제 그래프를 실행해서 실제 계산을 수행하고 결과를 보면 된다.

#그래프 실행은 세션안에서 이뤄저야 한다.
sess = tf.Session()

print(sess.run(hello))
print(sess.run([a,b,c]))

sess.close()