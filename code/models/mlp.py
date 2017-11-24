import math
import tensorflow as tf

def build_model(image_data, image_size, hsize):
    """
    single hidden layer for now
    input: images tensor - [batch_size, image_size]
    output: outputs tensor - [batch_size, 1]
    """

    with tf.name_scope('hidden'):
        weights = tf.Variable(
            tf.truncated_normal([image_size, hsize],
                                stddev=1.0 / math.sqrt(float(image_size))),
                              name='weights')
        biases = tf.Variable(tf.zeros([hsize]),
                             name='biases')
        hidden = tf.nn.relu(tf.matmul(image_data, weights) + biases)

    with tf.name_scope('output'):
        weights = tf.Variable(
            tf.truncated_normal([hsize, 1],
                                stddev=1.0 / math.sqrt(float(image_size))),
                              name='weights')
        biases = tf.Variable(tf.zeros([1]),
                             name='biases')
        output = tf.matmul(hidden, weights) + biases

    return output


def loss(outputs, labels):
    """
    Args:
        outputs: tensor - [batch_size, 1]
        labels:  tensor - [batch_size, 1]

    Returns:
        loss: float (MSE, etc.)
    """
    return tf.losses.mean_squared_error(labels, outputs)

def run_training(loss, lr):
    """ Train the network by gradient descent.

    Args: loss (float), learning_rate (float)

    Returns: train_op
    """
    tf.summary.scalar('loss', loss)
    # optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = tf.train.AdamOptimizer(lr)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluate(outputs, labels):
    # evaluate using MSE for now
    return loss(outputs, labels)

