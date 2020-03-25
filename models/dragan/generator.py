import tensorflow as tf
from tf.keras.layers import (Layer, Conv2D, Dense, Input,
                             BatchNormalization, LeakyRelu,
                             Lambda)
from tf.keras.models import Model

def build_generator(num_classes):
    input_tensor = Input((num_classes, ))

    # Fully Connected Layer
    x = Dense(64 * 16 * 16)(input_tensor)
    x = BatchNormalization()(x)
    resblock_input = tf.nn.relu(x)

    # 16 x Resblock Layers
    x = ResBlock(64, 3, strides=(1, 1))(resblock_input)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)
    x = ResBlock(64, 3, strides=(1, 1))(x)

    x = BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.add(x, resblock_input)

    # 2 x Pixel-shuffle layer
    x = Conv2D(256, 3, strides=(1, 1), padding='valid')(x)
    x = Lambda(pixel_shuffle_x2_layer)(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu(x)

    # 2 x Pixel-shuffle layer
    x = Conv2D(256, 3, strides=(1, 1), padding='valid')(x)
    x = Lambda(pixel_shuffle_x2_layer)(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu(x)

    # 2 x Pixel-shuffle layer
    x = Conv2D(256, 3, strides=(1, 1), padding='valid')(x)
    x = Lambda(pixel_shuffle_x2_layer)(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = Conv2D(3, 9, strides=(1, 1), padding='valid')
    output = tf.math.tanh(x)

    return Model(inputs=input_tensor, outputs=output)

class ResBlock(Layer):

    def __init__(self, no_filters, filter_shape,
                 strides, **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        assert len(no_filters) == 1 or len(no_filters ==2)
        if len(no_filters) == 1: no_filters_a = no_filters_b = no_filters
        else: no_filters_a, no_filters_b = no_filters

        assert len(filter_shape) == 1 or len(filter_shape == 2)
        if len(filter_shape) == 1: filter_shape_a = filter_shape_b = filter_shape
        else: filter_shape_a, filter_shape_b = filter_shape

        assert len(strides) == 1 or len(strides == 2)
        if len(strides) == 1: strides_a = strides_b = strides
        else: strides_a, strides_b = strides

        self.conv2d_a = Conv2D(no_filters_a, kernel_size=filter_shape_a,
                          strides=strides_a, padding='valid')
        self.batch_norm_a = BatchNormalization()

        self.conv2d_b = Conv2D(no_filters_b, kernel_size=filter_shape_b,
                          strides=strides_b, padding='valid')
        self.batch_norm_b = BatchNormalization()

    def call(self, input_tensor, training=False):

        x = self.conv2d_a(input_tensor)
        x = self.batch_norm_a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2d_b(x)
        x = self.batch_norm_b(x, training=training)
        x = tf.keras.layers.add(x, input_tensor)

        return x


def pixel_shuffle_x2_layer(x):
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
    fm_x = tf.shape(x)[1]
    fm_y = tf.shape(x)[2]

    # Reshape tensor to combine 2x channels along x-dim
    pix_shuffle_xdim = tf.reshape(x, [1, 2 * fm_x, fm_y, -1])

    # Perform transpose and reshape tensor to combine the 2x remaining channels along
    # the y-dim
    pix_shuffle_x2_output = tf.reshape(tf.transpose(pix_shuffle_xdim, perm=[0, 2, 1, 3]),
                                       [1, 2 * fm_x, 2 * fm_y, -1])

    return pix_shuffle_x2_output
