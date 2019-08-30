import tensorflow as tf
import numpy as np

char_arr = [chr(i) for i in range(ord('a'),ord('z')+1)]
num_dic = {n:i for i, n in enumerate(char_arr)}
#{'a':0, 'b':1, 'c':2 ....}
dic_len = len(num_dic)

seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'like', 'link', 'kiss', 'kind', 'kill', 'pool', 'pull']

def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]  #'wor', 'woo', 'dee'....의 인덱스
        target = num_dic[seq[-1]]   #'d', 'd', 'p',.... 의 인덱스
        input_batch.append(np.eye(dic_len)[input])  #이걸 원-핫 인코딩으로
        target_batch.append(target)

    return input_batch, target_batch

#hyperparameters
learning_rate = 0.01
n_hidden = 128
total_epoch = 30

n_step = 3  #단어 앞글자 3글자
n_input = n_class = dic_len #알파벳의 갯수

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])    #바로 알파벳으로 낼 거라서 int

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

#셀 생성
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

#outputs : [batch_size, n_step, n_hidden]
# -> [n_step, batch_size, n_hidden]
outputs = tf.transpose(outputs, [1,0,2])
# -> [batch_size, n_hidden]
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], {X:input_batch, Y:target_batch})

    print('Epoch:','%04d' %(epoch+1)+'/'+str(total_epoch), 'Cost: {:.6}'.format(loss))

print('Done!')

prediction = tf.cast(tf.argmax(model,1), tf.int32)
prediction_check = tf.equal(prediction, Y)

accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy], {X:input_batch, Y:target_batch})

predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n=== Prediction Results ===')
print('입력값:', [w[:3]+' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)