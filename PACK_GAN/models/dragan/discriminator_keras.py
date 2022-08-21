import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

def initialize_discriminator(discriminator_input, num_tags):
#    discriminator_input = tf.keras.Input(shape=(input_shape))
    print("(pjt) discriminator_input: ", discriminator_input.get_shape())
    x = tf.keras.layers.Conv2D(32, 4, strides=2,
                               padding='same',
                               activation=tf.keras.layers.LeakyReLU(),
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(discriminator_input)
    print("(pjt): 35", x.get_shape())
    x = ResBlock(3, 32)(x)
    x = ResBlock(3, 32)(x)
    x = tf.keras.layers.Conv2D(64, 4, strides=2,
                               padding='same',
                               activation=tf.keras.layers.LeakyReLU(),
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)
    print("(pjt): 35", x.get_shape())
    x = ResBlock(3, 64)(x)
    x = ResBlock(3, 64)(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2,
                               padding='same',
                               activation=tf.keras.layers.LeakyReLU(),
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)
    print("(pjt): 35", x.get_shape())
    x = ResBlock(3, 128)(x)
    x = ResBlock(3, 128)(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=2,
                               padding='same',
                               activation=tf.keras.layers.LeakyReLU(),
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)
    print("(pjt): 35", x.get_shape())
    x = ResBlock(3, 256)(x)
    x = ResBlock(3, 256)(x)
    x = tf.keras.layers.Conv2D(512, 3, strides=2,
                               padding='same',
                               activation=tf.keras.layers.LeakyReLU(),
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)
    print("(pjt): 43", x.get_shape())
    x = ResBlock(3, 512)(x)
    x = ResBlock(3, 512)(x)
    x = tf.keras.layers.Conv2D(1024, 3, strides=2,
                               padding='same',
                               activation=tf.keras.layers.LeakyReLU(),
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                               bias_initializer=RandomNormal(mean=0, stddev=0.02),
                               )(x)
    print("(pjt): 51", x.get_shape())

    forgery_score = tf.keras.layers.Dense(1, activation="sigmoid",
                                          kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                                          bias_initializer=RandomNormal(mean=0, stddev=0.02),
                                          )(x)

    tag_scores = tf.keras.layers.Dense(num_tags, activation="sigmoid",
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
        self.conv_layer1 = tf.keras.layers.Conv2D(self.num_filters, self.filter_shape,
                                                  strides=1,
                                                  padding='same',
                                                  activation=tf.keras.layers.LeakyReLU(),
                                                  kernel_initializer=RandomNormal(mean=0,stddev=0.02),
                                                  bias_initializer=RandomNormal(mean=0,stddev=0.02),
                                                  )
        self.conv_layer2 = tf.keras.layers.Conv2D(self.num_filters, self.filter_shape,
                                                  strides=1,
                                                  padding='same',
                                                  kernel_initializer=RandomNormal(mean=0,stddev=0.02),
                                                  bias_initializer=RandomNormal(mean=0,stddev=0.02),
                                                  )

    def call(self, inputs):
        x = self.conv_layer1(inputs)
        x = self.conv_layer2(x)
        x = tf.add(x, inputs)
        return tf.keras.layers.LeakyReLU(alpha=0.01)(x)






