#import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.python.framework import tensor_shape

def initialize_generator(image_input, tag_vector):
#    generator_input = tf.keras.Input(input_shape + num_tags)
    # Flatten input image tensor and concatenate with tag vector
    flattened_image = tf.keras.layers.Flatten()(image_input)
    x = tf.keras.layers.Concatenate(axis=-1)([flattened_image, tag_vector])
    x = tf.keras.layers.Dense(64 * 16 * 16,
                              kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                              bias_initializer=RandomNormal(mean=0, stddev=0.02),
                              )(x)
    x = tf.keras.layers.Reshape([128, 128, 1])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = resblock_input = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    # 16 residual blocks
    for _ in range(16):
        x = ResBlock(3, 64)(x)

    # Batch normalization and elementwise summation
    # with original input into 16-Resblock section
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.add(x, resblock_input)

    # Pixel shuffle layers
    for _ in range(3):
        x = tf.keras.layers.Conv2D(256, 3,
                                   strides=1,
                                   kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                                   bias_initializer=RandomNormal(mean=0, stddev=0.02),
                                   )(x)
        print("x_shape: ", x.shape)
#        x = tf.keras.layers.Lambda(pixel_shuffle_x2_layer)(x)
        x = PixelShuffleX2(256)(x)
        print("x_shape: ", x.shape)
#        x.set_shape([None, 508, 508, 64])
#        x.set_shape()
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Conv2D(9, 3,
                               strides=1,
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)

    return tf.keras.layers.Activation(tf.keras.activations.tanh)(x)

class ResBlock(layers.Layer):
    '''
    class defining ResBlock layer in Dragan generator.
    Includes batchnormalization layers
    '''
    def __init__(self,
                 filter_shape,
                 num_filters):
        super(ResBlock, self).__init__()
        self.filter_shape = filter_shape
        self.num_filters = num_filters

    def build(self, layer_input):

        # first convolutional layer
        self.c1 = tf.keras.layers.Conv2D(self.num_filters, self.filter_shape,
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                                         bias_initializer=RandomNormal(mean=0, stddev=0.02)
                                         )
        self.bn1 = tf.keras.layers.BatchNormalization()
#        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

        # second convolutional layer
        self.c2 = tf.keras.layers.Conv2D(self.num_filters, self.filter_shape,
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                                         bias_initializer=RandomNormal(mean=0, stddev=0.02)
                                         )
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Return elementwise summation of resblock input and output
#        return tf.add(x, layer_input)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.bn1(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

        x = self.c2(x)
        x = self.bn2 (x)

        return tf.add(x, inputs)

class PixelShuffleX2(tf.keras.layers.Layer):

    def __init__(self, input_channels):
        super(PixelShuffleX2, self).__init__()
        self.input_channels = input_channels

    def build(self, input_shape):
        self.build = True

    def call(self, input_fm):
        """
        Applies pixel shuffling to upsampled feature map.
        For an input of x256 channels, new feature maps will be composed using the
        next x4 channels in iteration.

        Function documented in the paper:
            "Real-Time Single Image and Video Super-Resolution Using an Efficient
             Sub-Pixel Convolutional Neural Network" -- Shi W. (2016)

        :param input_fm: input tensor of shape -- (batch_size, fm_x, fm_y, 256)

        :return out: output tensor of shape -- (batch_size, 2 * fm_x, 2 * fm_y, 64)
        """
        # Flatten input_fm along channel dimension
        #input_channels = tf.shape(input_fm)[3]
        #r = tf.math.sqrt(tf.cast(input_channels, dtype=tf.float32))
        fm_x = tf.shape(input_fm)[1]
        fm_y = tf.shape(input_fm)[2]

        # Reshape tensor to combine 2x channels along x-dim
        pix_shuffle_xdim = tf.reshape(input_fm, [1, 2 * fm_x, fm_y, -1])

        # Perform transpose and reshape tensor to combine the 2x remaining channels along
        # the y-dim
        pix_shuffle_x2_output = tf.reshape(tf.transpose(pix_shuffle_xdim, perm=[0, 2, 1, 3]),
                                           [1, 2 * fm_x, 2 * fm_y, -1])
        pix_shuffle_x2_output.set_shape(self.compute_output_shape(input_fm.shape))

        return pix_shuffle_x2_output

    def compute_output_shape(self, input_shape):
#        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return (input_shape[0], 2 * input_shape[1], 2 * input_shape[2],
                self.input_channels // 4)

#def pixel_shuffle_x2_layer(input_fm):
#    """
#    Applies pixel shuffling to upsampled feature map.
#    For an input of x256 channels, new feature maps will be composed using the
#    next x4 channels in iteration.
#
#    Function documented in the paper:
#        "Real-Time Single Image and Video Super-Resolution Using an Efficient
#         Sub-Pixel Convolutional Neural Network" -- Shi W. (2016)
#
#    :param input_fm: input tensor of shape -- (batch_size, fm_x, fm_y, 256)
#
#    :return out: output tensor of shape -- (batch_size, 2 * fm_x, 2 * fm_y, 64)
#    """
#    # Flatten input_fm along channel dimension
#    #input_channels = tf.shape(input_fm)[3]
#    #r = tf.math.sqrt(tf.cast(input_channels, dtype=tf.float32))
#    fm_x = tf.shape(input_fm)[1]
#    fm_y = tf.shape(input_fm)[2]
#
#    # Reshape tensor to combine 2x channels along x-dim
#    pix_shuffle_xdim = tf.reshape(input_fm, [1, 2 * fm_x, fm_y, -1])
#
#    # Perform transpose and reshape tensor to combine the 2x remaining channels along
#    # the y-dim
#    pix_shuffle_x2_output = tf.reshape(tf.transpose(pix_shuffle_xdim, perm=[0, 2, 1, 3]),
#                                       [1, 2 * fm_x, 2 * fm_y, -1])
#
#    return pix_shuffle_x2_output
