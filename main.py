import numpy as np
import tensorflow as tf
from model import char_rnn
from utils import build_dataset
from flask import Flask,jsonify,render_template,request

app=Flask(__name__)


def int_to_word(predict, vocabs):
    predict = predict[0]
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample > len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]



# 基本设置
start_token = 'B'
end_token = 'E'
result_dir = 'result'
corpus_file = 'data/poems.txt'
lr = 0.0002

print("正在从%s加载数据...." % corpus_file)
poems_vector,word_to_int,vocabularies=build_dataset(corpus_file)

# 初始化
print("正在从%s加载训练结果" % result_dir)
batch_size = 1
input_data = tf.placeholder(tf.int32, [batch_size, None])
end_points = char_rnn(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=lr)
saver = tf.train.Saver(tf.global_variables())
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess=tf.Session()
sess.run(init_op)

checkpoint = tf.train.latest_checkpoint(result_dir)
saver.restore(sess, checkpoint)


@app.route('/write_poem',methods=['POST','GET'])
def write():

    x = np.array([list(map(word_to_int.get, start_token))])
    [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                     feed_dict={input_data: x})
    begin_word = request.args.get('begin_word', 0, type=str)
    if begin_word:
        word = begin_word.strip()
    else:
        word = int_to_word(predict, vocabularies)
    poem_ = ''

    i = 0
    while word != end_token:
        poem_ += word
        i += 1
        if i >= 24:
            break
        x = np.zeros((1, 1))
        x[0, 0] = word_to_int[word]
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x, end_points['initial_state']: last_state})
        word = int_to_word(predict, vocabularies)

    return jsonify(result=poem_)
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True,port=5000)