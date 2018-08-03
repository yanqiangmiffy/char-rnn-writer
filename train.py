import os
import numpy as np
import tensorflow as tf
from model import char_rnn
from utils import build_dataset,build_name_dataset,generate_batch

# 参数设置
tf.app.flags.DEFINE_integer('batch_size',64,'batch size.')
tf.app.flags.DEFINE_float('learning_rate',0.01,'learning rate.')
tf.app.flags.DEFINE_string('result_dir','result/poem','trained model save path.')
tf.app.flags.DEFINE_string('file_path','data/poems.txt','file of poems dataset.')
tf.app.flags.DEFINE_string('model_prefix','poems','model save prefix.')
tf.app.flags.DEFINE_integer('epochs',50,'train how many epochs.')

FLAGS=tf.app.flags.FLAGS


def train():

    # 创建结果保存的路径
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
    if FLAGS.model_prefix=='poems':
        poems_vector,word_to_int,vocabularies=build_dataset(FLAGS.file_path)
    elif FLAGS.model_prefix=='names':
        poems_vector,word_to_int,vocabularies=build_name_dataset(FLAGS.file_path)

    batches_inputs,batches_outputs=generate_batch(FLAGS.batch_size,poems_vector,word_to_int)

    input_data=tf.placeholder(tf.int32,[FLAGS.batch_size,None])
    output_targets=tf.placeholder(tf.int32,[FLAGS.batch_size,None])

    end_points=char_rnn(model='lstm',
        input_data=input_data,
        output_data=output_targets,
        vocab_size=len(vocabularies),
        rnn_size=128,
        num_layers=2,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate)

    saver=tf.train.Saver(tf.global_variables())
    init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        start_epoch=0
        checkpoint=tf.train.latest_checkpoint(FLAGS.result_dir)
        if checkpoint: # 从上次结束的地方继续训练
            saver.restore(sess,checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        try:
            for epoch in range(start_epoch,FLAGS.epochs):
                n=0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))
                if epoch % 10 == 0:
                    saver.save(sess, os.path.join(FLAGS.result_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.result_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))

def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()