import tensorflow as tf
from model import char_rnn
from utils import build_name_dataset
import numpy as np

start_token = 'B'
end_token = 'E'
model_dir = 'result/name'
corpus_file = 'data/names.txt'

lr = 0.0002


def to_word(predict, vocabs):
    predict = predict[0]
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample > len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]


def gen_name(begin_word):
    batch_size = 1
    print('## loading corpus from %s' % model_dir)
    names_vector, word_int_map, vocabularies = build_name_dataset(corpus_file)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = char_rnn(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=128, learning_rate=lr)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        name_ = ''

        i = 0
        while word != end_token:
            name_ += word
            i += 1
            if i >= 24:
                break
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)

        return name_


def pretty_print_name(name_):
    name= "".join([word for word in name_ if word!='B'])
    print(name)
    

if __name__ == '__main__':
    begin_char = input('## 请输入您的姓氏:')
    name = gen_name(begin_char)
    pretty_print_name(name_=name)