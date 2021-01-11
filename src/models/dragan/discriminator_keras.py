import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

def initialize_discriminator(discriminator_input, num_tags):
#    discriminator_input = tf.keras.Input(shape=(input_shape))
    x = tf.keras.layers.Conv2d(32, 4, strides=2,
                               activation="leaky_relu",
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(discriminator_input)
    x = ResBlock(3, 32)(x)
    x = ResBlock(3, 32)(x)
    x = tf.keras.layers.Conv2d(64, 4, strides=2,
                               activation="leaky_relu",
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)
    x = ResBlock(3, 64)(x)
    x = ResBlock(3, 64)(x)
    x = tf.keras.layers.Conv2d(128, 4, strides=2,
                               activation="leaky_relu",
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)
    x = ResBlock(3, 128)(x)
    x = ResBlock(3, 128)(x)
    x = tf.keras.layers.Conv2d(256, 3, strides=2,
                               activation="leaky_relu",
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)
    x = ResBlock(3, 256)(x)
    x = ResBlock(3, 256)(x)
    x = tf.keras.layers.Conv2d(512, 3, strides=2,
                               activation="leaky_relu",
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)
    x = ResBlock(3, 512)(x)
    x = ResBlock(3, 512)(x)
    x = tf.keras.layers.Conv2d(1024, 3, strides=2,
                               activation="leaky_relu",
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)

    forgery_score = tf.keras.layers.Dense(1, activation="sigmoid",
                                          kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                                          bias_initializer=RandomNormal(mean=0, stddev=0.02),
                                          )(x)

    tag_scores = tf.keras.layers.Dense(num_tags, activation="signmoid",
                                       kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                                       bias_initializer=RandomNormal(mean=0, stddev=0.02),
                                       )(x)

    return forgery_score, tag_scores

class ResBlock(layers.Layer):
    def __init__(self,
                 filter_shape,
                 num_filters=32):
        super(ResBlock, self).__init__()
        self.filter_shape = filter_shape
        self.num_filters = num_filters

    def build(self, input_shape):
        self.conv_layer1 = tf.keras.layers.Conv2d(self.num_filters, self.filter_shape,
                                                  strides=1,
                                                  activation="leaky_relu",
                                                  kernel_initializer=RandomNormal(mean=0,stddev=0.02),
                                                  bias_initializer=RandomNormal(mean=0,stddev=0.02),
                                                  )
        self.conv_layer2 = tf.keras.layers.Conv2d(self.num_filters, self.filter_shape,
                                                  strides=1,
                                                  kernel_initializer=RandomNormal(mean=0,stddev=0.02),
                                                  bias_initializer=RandomNormal(mean=0,stddev=0.02),
                                                  )

    def call(self, inputs):
        x = self.conv_layer1(inputs)
        x = self.conv_layer2(x)
        x = tf.add(x, inputs)
        return tf.keras.layers.LeakyReLU(alpha=0.01)(x)






