#MLP:共三层，隐藏层1000神经元，Accuracy: 0.9638
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

#建立layer函数
def layer(output_dim, input_dim, inputs, activation = None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs

#建立输入层x
x = tf.placeholder('float', [None, 784])
#建立隐藏层h1
h1 = layer(output_dim = 200, input_dim = 784,
           inputs = x, activation = tf.nn.relu
          )
#建立输出层y
y_predict = layer(output_dim = 10, input_dim = 200,
                  inputs = h1, activation = None
                 )

y_label = tf.placeholder('float', [None, 10])

#定义损失函数（交叉熵cross_entropy）
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_predict, labels = y_label))
#定义优化器
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_function)

#计算每一项数据是否预测成功
correct_prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))
#预测正确结果的平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

#定义训练参数
trainEpochs = 15
batchSize = 100
totalBatchs = 550
loss_list = []
epoch_list = []
accuracy_list = []
from time import time
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#进行训练
for epoch in range(0, trainEpochs):
    for i in range(0, totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict = {x: batch_x, y_label: batch_y})
    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict = {x: mnist.validation.images, y_label: mnist.validation.labels}
                        )
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)
    print('Train Epoch:', '%02d' % (epoch + 1), 'Loss=', '{:.9f}'.format(loss), 'Accuracy=', acc)
print('训练结束')

#模型保存和读取
saver = tf.train.Saver()
saver.save(sess, "MLP_Model/model.ckpt")
saver.restore(sess, "./MLP_Model/model.ckpt")

#评估模型准确率
print('Accuracy:', sess.run(accuracy,
                            feed_dict = {x: mnist.test.images,
                                         y_label: mnist.test.labels}
                            ))
