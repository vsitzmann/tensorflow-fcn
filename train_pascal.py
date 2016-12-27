import numpy as np

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from scipy.misc import imread
import tensorflow as tf

import fcn8_vgg

import utils
import scipy as scp

import matplotlib.pyplot as plt

def load_pascal_train():
    with open('/data_4/pascal_pure.msg','r') as msg_file:
        pascal_train = msgpack.load(msg_file)

    with open('/data_4/pascal_pure_trainval.msg','r') as msg_file:
        pascal_trainval = msgpack.load(msg_file)

    whole_dataset = {}

    for key in pascal_train:
        whole_dataset[key] = np.concatenate([pascal_train[key], pascal_trainval[key]], axis=0)

    return whole_dataset


def get_batch_iterator(batch_size, dataset, endless=True):
    total_batch_counter = 0
    while True:
        batch_counter = 0
        tot_data_count = len(dataset['image'])

        # shuffle dataset:
        idx_range = np.arange(tot_data_count)
        idx_shuffle = np.random.shuffle(idx_range)

        images = np.squeeze(dataset['image'][idx_shuffle,:,:,:])
        gts = np.squeeze(dataset['gt'][idx_shuffle,:,:])
        image_name = dataset['img_name'][idx_shuffle,:]

        while True:
            lower_idx = batch_counter * batch_size
            upper_idx = lower_idx + batch_size

            if upper_idx > tot_data_count - 1: break

            yield images[lower_idx:upper_idx,:, :, :].astype(np.float32), gts[lower_idx:upper_idx,:,:].astype(np.int32), image_name[lower_idx:upper_idx]

            batch_counter += 1
            total_batch_counter += 1

        if not endless: break


def _get_train_op(loss, learning_rate=1e-6):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables.

    Args:
    total_loss: Total loss from loss().
    lr: Learning rate.
    Returns:
    train_op: op for training.
    """
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    return train_step

def loss(logits, labels, num_classes):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, [-1, num_classes])
        epsilon = tf.constant(value=1e-4)
        logits = logits + epsilon
        labels = tf.reshape(labels, [-1])

        # Object borders are class 21 - this class will be dropped by the tf.one_hot function.
        labels_one_hot = tf.one_hot(labels, on_value=1, off_value=0, depth=20)
        labels_one_hot = tf.reshape(labels_one_hot, [-1, 20])

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels_one_hot, dim=-1)
        loss=tf.reduce_mean(cross_entropy

        # labels = tf.to_float(labels)
        #
        #softmax = tf.nn.softmax(logits)
        # cross_entropy = tf.contrib.losses.sparse_softmax_cross_entropy(logits, labels)

        #cross_entropy = -tf.reduce_sum(labels_one_hot * tf.log(softmax), reduction_indices=[1])
        # cross_entropy_mean = tf.reduce_mean(cross_entropy,
        #                                     name='xentropy_mean')
        # tf.add_to_collection('losses', cross_entropy_mean)

        # loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


def iou(predictions, gts, out_path=None):
    # true positive / (truepositive + false positive + false negative)
    accuracies = np.zeros(20)
    for i in xrange(20):
        true_pos = np.sum(np.logical_and(predictions==i, gts==i))
        all_pred_pos = np.sum(predictions==i)
        all_gt_pos = np.sum(gts==i)
        accuracies[i] = 100*true_pos/(all_pred_pos + all_gt_pos - true_pos)

    print(accuracies)
    print("Mean iou: %f"%accuracies)

    if out_path is not None:
        with open(out_path, 'wb') as np_file:
            np.savetxt(np_file, accuracies)


def train(num_batches=500, batch_size=50):
    dataset = load_pascal_train()
    batch_iterator = get_batch_iterator(batch_size, dataset, endless=True)

    # test_img1 = np.repeat(imread("./test_data/tabby_cat.png"), axis=0)

    images = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
    labels = tf.placeholder(tf.int32, shape=(batch_size, 224, 224))

    vgg_fcn = fcn8_vgg.FCN8VGG()
    with tf.name_scope("content_vgg"):
        # TODO dropout
        vgg_fcn.build(images, train=True, debug=True, random_init_fc8=False, num_classes=20)

    train_loss = loss(vgg_fcn.upscore32, labels, 20)
    train_step = _get_train_op(train_loss)
    init = tf.global_variables_initializer()

    print('Finished building Network.')

    with tf.Session() as sess:
        sess.run(init)

        for batch_no in xrange(num_batches):
            batch_imgs, batch_gts, _ = batch_iterator.next()
            train_step.run(feed_dict={images: batch_imgs,
                                      labels: batch_gts},
                           session=sess)

            if not batch_no % 50:
                train_loss_ex = train_loss.eval(feed_dict={images: batch_imgs,
                                                        labels: batch_gts})
                test_pred = sess.run([vgg_fcn.pred_up],
                                     feed_dict={images:batch_imgs})[0]
                print(train_loss_ex)
                with open('test.txt', 'wb') as f:
                    np.savetxt(f, test_pred[0].astype(float))
                up_color = utils.color_image(test_pred[0])

                #plt.imshow(up_color[:, :, :3])
                #plt.show()

                scp.misc.imsave('test.png', up_color)


