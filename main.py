import os
import argparse
import warnings
import tensorflow as tf
from utils import get_batches, download_pretrained_vgg ,save_inference_samples
from distutils.version import LooseVersion
from os.path import join, expanduser
from improcess import perform_augmentation

# usual training session
def train_nn(sess, training_epochs, batch_size, get_batches, train_op, cross_entropy_loss,
             image_input, labels, keep_prob, learning_rate):

    # Variable initialization
    sess.run(tf.global_variables_initializer())

    # use learning rate in argument or default 1e-4
    lr = args.learning_rate

    for e in range(0, training_epochs):

        loss_this_epoch = 0.0

        for i in range(0, args.b_per_ep):

            # Load a batch of examples
            batch_x, batch_y = next(get_batches(batch_size))
            if preprocess:
                # if defined use preprocessing to expand training batches
                batch_x, batch_y = perform_augmentation(batch_x, batch_y)

               # calculate loss for current epoch
            _, cur_loss = sess.run(fetches=[train_op, cross_entropy_loss],
                                   feed_dict={image_input: batch_x, labels: batch_y, keep_prob: 0.25,
                                              learning_rate: lr})

            loss_this_epoch += cur_loss

        print('Epoch: {:02d}  -  Loss: {:.03f}'.format(e, loss_this_epoch / args.b_per_ep))

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, n_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    For reference: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    Fully convolutional networks can efficiently learn to make dense predictions for per-pixel 
    tasks like semantic segmentation.
    (At the last fully connected layer, the network overloads GPU memory under 5GB, making training unefficient...)
    
    Uses 3 pretrained layers from VGG16 network, that are inputs to convolutional layers between to produce additional class predictions.
    Use 2x upsampling at 'deconvolutional' layers.
    Layer predictions are combined in the fusion layers to combine coarse, high layer information with fine, low layer information.
    Predictions are combined with deconvolutional layers in fusion streams.
    (Additional information on the architecture is avalable in Section 4.2 of the original paper.)
    """
    # choose a more drastic l2 (feedforward) regularization of 0.01 to avoid overfitting
    # Dropout maybe?
    kernel_regularizer = tf.contrib.layers.l2_regularizer(0.01)

    # Compute probabilities
    layer3_logits = tf.layers.conv2d(vgg_layer3_out, n_classes, kernel_size=[1, 1],
                                     padding='same', kernel_regularizer=kernel_regularizer)
    layer4_logits = tf.layers.conv2d(vgg_layer4_out, n_classes, kernel_size=[1, 1],
                                     padding='same', kernel_regularizer=kernel_regularizer)
    layer7_logits = tf.layers.conv2d(vgg_layer7_out, n_classes, kernel_size=[1, 1],
                                     padding='same', kernel_regularizer=kernel_regularizer)

    # Add skip connection before 4th and 7th layer
    layer7_logits_upsample = tf.image.resize_images(layer7_logits, size=[10, 36])
    layer_4_7_fusion = tf.add(layer7_logits_upsample, layer4_logits)

    # Add skip connection before (4+7)th and 3rd layer
    layer_4_7_upsample = tf.image.resize_images(layer_4_7_fusion, size=[20, 72])
    layer_3_4_7_fusion = tf.add(layer3_logits, layer_4_7_upsample)

    # resize to original size
    layer_3_4_7_upsample = tf.image.resize_images(layer_3_4_7_fusion, size=[160, 576])
    layer_3_4_7_upsample = tf.layers.conv2d(layer_3_4_7_upsample, n_classes, kernel_size=[15, 15],
                                      padding='same', kernel_regularizer=kernel_regularizer)

    return layer_3_4_7_upsample

def load_vgg(sess, vgg_path):

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3_out = graph.get_tensor_by_name('layer3_out:0')
    layer4_out = graph.get_tensor_by_name('layer4_out:0')
    layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

def run():
    n_classes = 2

    # set up shape of input to first layer
    image_h, image_w = (160, 576)

    with tf.Session() as sess:

        # set up path to vgg network model
        vgg_path = join(data_dir, 'vgg')

        # load training batches generator
        batches = get_batches(join(data_dir, 'data_road/training'),(image_h, image_w))
        
        # Load pretrained VGG model
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # Create Fully convolutional layers (fusion, upsampling)
        output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, n_classes)

        # Define placeholders labels and learning rate
        labels = tf.placeholder(tf.float32, shape=[None, image_h, image_w, n_classes])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        # training pipeline as usual
        logits_flat = tf.reshape(output, (-1, n_classes)) # flatten to 1-D
        labels_flat = tf.reshape(labels, (-1, n_classes)) # flatten to 1-D
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_flat, logits=logits_flat)
        loss_function = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = optimizer.minimize(loss_function)

        # Train FCN
        train_nn(sess, args.ep, args.b_size, batches, image_input, loss_function,
                 image_input, labels, keep_prob, learning_rate)

        save_inference_samples(runs_dir, data_dir, sess, (image_h, image_w), logits_flat, keep_prob, image_input)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # decrease batch size to reduce allocated memory usage
    parser.add_argument('--b_size', type=int, default=8, help='Define size of training batches.', metavar='')
    parser.add_argument('--b_per_ep', type=int, default=100, help='Define batches per training iterations.', metavar='')
    parser.add_argument('--ep', type=int, default=30, help='Define amount of training iterations.', metavar='')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Define learning rate.', metavar='')
    parser.add_argument('--augmentation', type=bool, default=True, help='Perform preprocessing in training.', metavar='')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use', metavar='')

    return parser.parse_args()

if __name__ == '__main__':

    data_dir = join(os.path.dirname(__file__), 'data')
    runs_dir = join(os.path.dirname(__file__), 'prediction')

    args = parse_args()
    preprocess = False
    download_pretrained_vgg(data_dir)
    preprocess = args.augmentation
    run()
