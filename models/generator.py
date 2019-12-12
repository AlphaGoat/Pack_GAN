"""
Implements modified SRResNet generator architecture, as detailed in
'Towards the Automatic Anime Characters Creation with Generative
Adversarial Neural Networks', Yanghua et. al 2017
--------------------------
[INSERT ARXIV LINK HERE]
-------------------------

Peter J. Thomas
09 December 2019
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
        residual_input = re_out_fc1 = tf.reshape(act_out_fc1, [-1, 16, 16, 64])

        ####################################################
        ##################  RESIDUAL LAYERS ################
        ####################################################

        # Initialize 16 Residual blocks
        for i in range(1, 17):
            with tf.name_scope('residual_block{}'.format(i)):

                # first 2-D Convolutional Layer:
                # Initializing filter for first conv layer
                kernel1_res = WeightVariable(shape=[3, 3, 64, 64],
                                             name='Filter1_resblock{}'.format(i),
                                             model_scope=self.model_scope,
                                             initializer=tf.initializer.TruncatedNormal(mean=0.0,
                                                                                        stddev=0.02)
                                            )
                # Initializing bias parameters for first conv layer
                bias1_res = BiasVariable(shape=(64,),
                                         name='bias1_resblock{}'.format(i),
                                         model_scope=self.model_scope,
                                         initializer=tf.initializer.TruncatedNormal(mean=0.02,
                                                                                    stddev=0.02),
                                         )

                # output shape: (batch_size, 16, 16, 64)
                feature_map1_res = tf.nn.conv2d(residual_input,
                                                kernel1_res,
                                                strides=[1, 1, 1, 1],
                                                padding='SAME'
                                                )
                # batch_normalization
                # output shape: (batch_size, 16, 16, 64)
                batch_norm_feature_map_1 = tf.nn.batch_normalization(feature_map1_res)

                # first activation
                # output shape: (batch_Size, 16, 16, 64)
                act_feature_map_1 = tf.nn.relu(batch_norm_feature_map_1)

                # 2nd 2-D Convolutional Layer:
                # initializing second filter
                kernel2_res = WeightVariable(shape=[3, 3, 64, 64],
                                             name='Filter2_resblock{}'.format(i),
                                             model_scope=self.model_scope,
                                             initializer=tf.initializer.TruncatedNormal(mean=0.02,
                                                                                        stddev=0.02)
                                             )
                bias2_res = BiasVariable(shape=(64,),
                                         name='bias2_resblock{}'.format(i),
                                         model_scope=self.model_scope,
                                         initializer=tf.initializer.TruncatedNormal(mean=0.02,
                                                                                    stddev=0.02)
                                         )
                # output shape: (batch_size, 16, 16, 64)
                feature_map2_res = tf.nn.conv2d(act_feature_map_1,
                                                kernel2_res,
                                                strides=[1, 1, 1, 1],
                                                padding='SAME'
                                                )
                # Add bias tensor
                bias_feature_map2_res = tf.bias_add(feature_map2_res, bias2_res)

                # batch_normalization
                # output shape: (batch_size, 16, 16, 64)
                batch_norm_feature_map2_res = tf.nn.batch_normalization(feature_map2_res)

                # element-wise sum of output of second convolution with
                # original residual input
                # output shape: (batch_size, 16, 16, 64)
                residual_sum = tf.add(batch_norm_feature_map2_res, residual_input)

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
            # Shape: (batch_size, 16, 16, 64)
            residual_output = tf.add(act_residual_block_output, re_out_fc1)

        ############################################################################
        ######################### END RESIDUAL LAYERS ##############################
        ############################################################################

        # Upsampling sub-pixel convolution: scales the input tensor by 2 in both the
        #           x and y dimension and randomly shuffles pixels in those dimensions
        for i in range(1, 4):
            with tf.name_scope('upsampling_subpixel_convolution{}'.format(i)):

                # Initializer filter and bias for upsampling convolution
                upscale_kernel = WeightVariable(shape=[3, 3, 64, 256],
                                                name='Filter_PixelShuffle1',
                                                model_scope=self.model_scope,
                                                initializer=tf.initializer.TruncatedNormal(mean=0.02,
                                                                                           stddev=0.02)
                                                )
                upscale_bias = BiasVariable(shape=(256,),
                                            name='bias_PixelShuffle{}'.format(i),
                                            model_scope=self.model_scope,
                                            initializer=tf.initializer.TruncatedNormal(mean=0.02,
                                                                                       stddev=0.02)
                                            )

                # Output shape: (batch_size, 16, 16, 256)
                upscale_feature_map = tf.nn.conv2d(residual_output,
                                                   kernel_subpix_1,
                                                   strides=[1, 1, 1, 1],
                                                   padding='SAME'
                                                   )
        # Pixel shuffling operation
        # Output shape: (batch_Size, 16, 16,


#        for i in range(1, 4):
#            with tf.name_scope('upsampling_sub-pixel_convolution{}'.format(i)):
#
#                # 2-D conv layer with filter 3x3
#                kernel = WeightVariable(shape=[3, 3,

    def pixel_shuffle_x2_layer(self, input_fm):
        """
        Applies pixel shuffling to upsampled feature map.
        For an input of x256 channels, new feature maps will be composed using the
        next x4 channels in iteration.

        :param input_fm: input tensor of shape -- (batch_size, fm_x, fm_y, 256)

        :return out: output tensor of shape -- (batch_size, 2 * fm_x, 2 * fm_y, 64)
        """


        def apply_pix_shuffling_to_batch(batch):

            # Compose a shuffled pixel map with every 4xgroup of input feature
            # maps by iterating over the channel dim in intervals of 4:
            i = tf.constant(0)
            cond = lambda i: tf.less(i, 64)

            def gather_shuffled_pixels():

                # Compose indices that will be needed to perform pixel shuffling operation
                x_dim, y_dim, channel_dim = tf.meshgrid(tf.range(tf.shape(input_fm)[0]),
                                                        tf.range(tf.shape(input_fm)[1]),
                                                        tf.range(4*i, 4*(i+1))
                                                        )

                coords = tf.stack([x_dim, y_dim, channel_dim], axis=-1)

                # Gather the pixels from a 4x group of feature maps
                # shape: (2 * fm_x, 2* fm_y, 1)
                upsampled_shuffled_feature_map = tf.gather_nd(batch, coords)

                # update iterator
                i = tf.add(i, 1)

                return upsampled_shuffled_feature_map

            # Perform pixel shuffling operation
            # input shape: (fm_x, fm_y, 256)
            # output shape: (2 * fm_x, 2 * fm_y, 64)
            pixel_shuffled_batch = tf.while_loop(cond, gather_shuffled_pixels(batch), i)

            return pixel_shuffled_batch

        # Maps pixel shuffling operation to a single batch.
        # Input shape: (batch_size, fm_x, fm_y, 256)
        # Output shape: (batch_size, 2 * fm_x, 2 * fm_y, 64)
        pix_shuffled_feature_maps = tf.map_fn(apply_pix_shuffling_to_batch, input_fm)

        return pix_shuffled_feature_maps




























