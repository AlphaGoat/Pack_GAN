import tensorflow as tf
from tf.keras.layers import Input, Conv2D, Dense, Layer
from tf.keras.layers import LeakyRelu
from tf.keras.models import Model

def build_discriminator(num_classes):
    input_tensor = Input(shape=(128,))

    # 1st conv layer (32 x channel)
    x = Conv2D(32, 4, strides=(2, 2), padding='valid')(input_tensor)
    x = LeakyRelu(alpha=0.1)(x)

    # 32 x channel Resblock
    x = ResBlock(32, 3, strides=(1, 1))(x)

    # 32 x Resblock
    x = ResBlock(32, 3, strides=(1, 1))(x)

    # Conv layer (64 x channel)
    x = Conv2D(64, 4, strides=(2, 2))(x)
    x = LeakyRelu(alpha=0.1)(x)

    # 64 x channel Resblock
    x = ResBlock(64, 3, strides=(1, 1))(x)

    # 64 x channel Resblock
    x = ResBlock(64, 3, strides=(1, 1))(x)

    # Conv layer (128 x channel)
    x = Conv2D(128, 4, strides=(2, 2), padding='valid')(x)
    x = LeakyRelu(alpha=0.1)(x)

    # 128 x channel Resblock
    x = ResBlock(128, 3, strides=(1, 1))

    # 128 x channel Resblock
    x = ResBlock(128, 3, strides=(1, 1))

    # Conv layer (256 x channel)
    x = Conv2D(256, 3, strides=(2, 2), padding='valid')
    x = LeakyRelu(alpha=0.1)(x)

    # 256 x channel Resblock
    x = ResBlock(256, 3, strides=(1, 1))

    # 256 x channel Resblock
    x = ResBlock(256, 3, strides=(1, 1))

    # Conv layer (512 x channel)
    x = Conv2D(512, 3, strides=(2, 2), padding='valid')
    x = LeakyRelu(alpha=0.1)(x)

    # 512 x channel Resblock
    x = ResBlock(512, 3, strides=(2, 2))

    # 512 x channel Resblock
    x = ResBlock(512, 3, strides=(2, 2))

    # Output layers
    forgery_score = Dense(1)(x)
    forgery_score = tf.keras.activations.sigmoid(forgery_score)

    class_scores = Dense(num_classes)(x)
    class_scores = tf.keras.activations.sigmoid(class_scores)

    return Model(inputs=input_tensor, outputs=[forgery_score, class_scores])

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

        self.conv2d_b = Conv2D(no_filters_b, kernel_size=filter_shape_b,
                          strides=strides_b, padding='valid')

    def call(self, input_tensor, training=False):

        x = self.conv2d_a(input_tensor)
        x = tf.nn.leaky_relu(x)

        x = self.conv2d_b(x)
        x = tf.keras.layers.add(x, input_tensor)
        x = tf.nn.leaky_relu(x)

        return x











