"""
Implements modified SRResNet generator architecture, as detailed in
'Towards the Automatic Anime Characters Creation with Generative
Adversarial Neural Networks', Yanghua et. al 2017
--------------------------
[INSERT ARXIV LINK HERE]
-------------------------

Peter J. Thomas
"""

import tensorflow as tf
from layers import WeightVariable, BiasVariable


class Generator(object):

    def __init__(self,
                 image_width,
                 image_height,
                 image_channels):

        # Provide dims of image we would like to generate
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels

    def residual_block(self):
        """
        resblock adopted from SRResNet.
        The generator uses 16 of these
        blocks in total
        """

        #TODO: Conv Block
        #TODO: Batch Normalization Layer
        #TODO: ReLU Activation
        #TODO: Conv Block
        #TODO: Batch Normalization Layer
        #TODO: Elementwise Sum layer

    def sub_pixel_cnn(self):
        pass

    def forward_pass(self, x):
        """
        Params:
            x: input tensor of noise with shape:
                    (batch_size, image_width, image_height, channels)
                    where image_width, image_height are dims of desired image output
        Return:
            out: output tensor of same shape as input
        """
        # Initial dense layer
        with tf.name_scope('fully_connected1'):

            # Weight variable
            # shape: (batch_size, width * height * channels, 64 * 16 * 16)
            wfc1_shape = (self.image_height * self.image_width * self.image_channels, 64 * 16 * 16)
            W_fc1 = WeightVariable(shape=wfc1_shape,
                                   name='W_fc1',
                                   model_scope=self.model_scope,
                                   initializer=tf.initializer.TruncatedNormal(mean=0.0,
                                                                              stddev=0.02)
                                   )
            # Initialize bias variable
            # Shape: (batch_size, 64 * 16 * 16)
            b_fc1 = BiasVariable(shape=(64 * 16* 16,),
                                 name='b_fc1',
                                 model_scope=self.model_scope,
                                 initializer=tf.initializer.TruncatedNormal(mean=0.0,
                                                                            stddev=0.02)
                                 )
            # Implement dense layer:
            out_fc1 = tf.bias_add(tf.matmul(x, W_fc1), b_fc1)

            # Batch Normalization
            batch_norm_out_fc1 = tf.nn.batch_normalization(out_fc1)

            # Activation of output
            act_out_fc1 = tf.nn.reul(batch_norm_out_fc1)

        # Reshape output of dense layer to feed into resblocks
        # NOTE: the output of this first dense layer will be saved and used
        #       at the end of the following residual blocks as the last
        #       input
        # Shape: (batch_size, 16, 16, 64)
        resdiual_input = re_out_fc1 = tf.reshape(act_out_fc1, [-1, 16, 16, 64])

        # Initialize 16 Residual blocks
        for i in range(1, 17):
            with tf.name_scope('residual_block{}'.format(i)):

                # first 2-D Convolutional Layer:
                # Initializing first filter
                kernel_1 = WeightVariable(shape=[3, 3, 64, 64],
                                          name='Filter1_resblock{}'.format(i),
                                          model_scope=self.model_scope,
                                          initializer=tf.initializer.TruncatedNormal(mean=0.0,
                                                                                     stddev=0.02)
                                         )
                # output shape: (batch_size, 16, 16, 64)
                feature_map_1 = tf.nn.conv2d(residual_input,
                                             kernel_1,
                                             strides=[1, 1, 1, 1],
                                             padding='SAME'
                                             )
                # batch_normalization
                # output shape: (batch_size, 16, 16, 64)
                batch_norm_feature_map_1 = tf.nn.batch_normalization(feature_map_1)

                # first activation
                # output shape: (batch_Size, 16, 16, 64)
                act_feature_map_1 = tf.nn.relu(batch_norm_feature_map_1)

                # 2nd 2-D Convolutional Layer:
                # initializing second filter
                kernel_2 = WeightVariable(shape=[3, 3, 64, 64],
                                          name='Filter2_resblock{}'.format(i),
                                          model_scope=self.model_scope,
                                          initializer=tf.initializer.TruncatedNormal(mean=0.02,
                                                                                     stddev=0.02)
                                          )
                # output shape: (batch_size, 16, 16, 64)
                feature_map_2 = tf.nn.conv2d(act_feature_map_1,
                                             kernel_2,
                                             strides=[1, 1, 1, 1],
                                             padding='SAME'
                                             )
                # batch_normalization
                # output shape: (batch_size, 16, 16, 64)
                batch_norm_feature_map_2 = tf.nn.batch_normalization(feature_map_2)

                # element-wise sum of output of second convolution with
                # original residual input
                # output shape: (batch_size, 16, 16, 64)
                residual_sum = tf.add(batch_norm_feature_map_2, residual_input)

                # Final activation
                # NOTE: May remove this activation, not included in actual paper
                residual_input = residual_block_output = tf.nn.relu(residual_sum)

        # Final residual sum
        with tf.name_scope('final_residual_output'):

            # Batch Normalization
            # Shape: (batch_Size, 16, 16, 64)
            batch_norm_residual_block_output = tf.nn.batch_normalization(residual_block_output)

            # ReLU Activation
            # SHape: (batch_size, 16, 16, 64)
            act_residual_block_output = tf.nn.relu(batch_norm_residual_block_output)

            # Final element-wise sum, with original res block input (i.e., the output of the
            # first fully-connected layer
            residual_output = tf.add(act_residual_block_output, re_out_fc1)























