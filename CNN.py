#Accuracy: 0.9898
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

#查看多项训练数据的images与labels
def plot_images_labels_prediction(images, labels,
                                 prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    #不超过25个
    if num>25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[idx].reshape(28, 28), cmap = 'binary')
        title = 'label=' + str(np.argmax(labels[idx]))
        if len(prediction)>0:
            title += ',predict='+str(prediction[idx])
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

#plot_images_labels_prediction(mnist.train.images, mnist.train.labels, [], 0, 10)

#建立共享函数
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1],
                          padding = 'SAME'
                          )
#建立输入层x
with tf.name_scope('Input_Layer'):
    x = tf.placeholder('float', shape = [None, 784], name = 'x')
    x_image = tf.reshape(x, [-1, 28, 28, 1])

#建立卷积层1
with tf.name_scope('C1_Conv'):
    W1 = weight([5, 5, 1, 16])
    b1 = bias([16])
    Conv1 = conv2d(x_image, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)
#建立池化层1
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)
#建立卷积层2
with tf.name_scope('C2_Conv'):
    W2 = weight([5, 5, 16, 36])
    b2 = bias([36])
    Conv2 = conv2d(C1_Pool, W2) + b2
    C2_Conv = tf.nn.relu(Conv2)
#建立池化层2
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv)
#建立平坦层
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_Pool, [-1, 1764])
#建立隐藏层
with tf.name_scope('D_Hidden_Layer'):
    W3 = weight([1764, 128])
    b3 = bias([128])
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3) + b3)
    D_Hidden_Dropout = tf.nn.dropout(D_Hidden, keep_prob = 0.8)

#建立输出层
with tf.name_scope('Output_Layer'):
    W4 = weight([128, 10])
    b4 = bias([10])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout, W4) + b4)

#定义训练方式
with tf.name_scope('optimizer'):
    y_label = tf.placeholder('float', [None, 10], name='y_label')
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_predict, labels = y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_function)
#定义评估方式
with tf.name_scope('evaluate_model'):
    correct_prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

#定义训练参数
trainEpochs = 30
batchSize = 100
totalBatchs = int(mnist.train.num_examples/batchSize)
loss_list = []
epoch_list = []
accuracy_list = []
from time import time

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#进行训练
startTime = time()
print('开始训练：')
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
    endTime = time()
    print('用时' + str(endTime - startTime) + '秒')
endTime = time()
print('训练完成！')
print('训练时间=' + str(endTime - startTime) + '秒')

#模型保存和读取
saver = tf.train.Saver()
saver.save(sess, "CNN_Model/model.ckpt")
saver.restore(sess, "./CNN_Model/model.ckpt")

#评估模型准确率
print('Accuracy:', sess.run(accuracy,
                            feed_dict = {x: mnist.test.images,
                                         y_label: mnist.test.labels}
                            ))

