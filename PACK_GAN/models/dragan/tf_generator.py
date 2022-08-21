"""
Implements modified SRResNet generator architecture, as detailed in
'Towards the Automatic Anime Characters Creation with Generative
Adversarial Neural Networks', Yanghua et. al 2017
--------------------------
https://arxiv.org/pdf/1708.05509.pdf
-------------------------

Peter J. Thomas
09 December 2019
"""

import tensorflow as tf

from models.layers import WeightVariable, BiasVariable, BatchNormalization

class SRResNet(object):

    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 latent_space_vector_dim,
                 num_tags,
                 variable_summary_update_freq=10,
                 model_scope="SRResNet_Generator"):

        # Provide dims of image we would like to generate
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels

        # Provide dims of input
        self.latent_space_vector_dim = latent_space_vector_dim
        self.num_tags = num_tags

        # variable scope to be used for the model
        self.model_scope = model_scope

        # frequency with which we'll update tensorboard with
        # statistics about model parameters
        self.variable_summary_update_freq = variable_summary_update_freq

    def forward_pass(self, x, step=0):
        """
        Params:
            x: input tensor of noise with shape:
                    (batch_size, image_width, image_height, channels)
                    where image_width, image_height are dims of desired image output

            step: What step are we at in the training loop (Note: necessary to generate
                  variable summaries for weights and biases

        Return:
            out: output tensor of same shape as input
        """
        # Initial dense layer
        with tf.name_scope(self.model_scope):

            with tf.name_scope('fully_connected1') as layer_scope:

                #print("layer_scope as presented in generator: ", layer_scope)
                print("We are currently on step: ", step)

                # Weight variable
                # shape: (batch_size, width * height * channels, 64 * 16 * 16)
                wfc1_shape = (self.latent_space_vector_dim + self.num_tags, 64 * 16 * 16)
                W_fc1 = WeightVariable(shape=wfc1_shape,
                                       name='W_fc1',
                                       #model_scope=self.model_scope,
                                       layer_scope=layer_scope,
                                       initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                  stddev=0.02),
                                       summary_update_freq=self.variable_summary_update_freq,
                                       )(step)

                # Initialize bias variable
                # Shape: (batch_size, 64 * 16 * 16)
                b_fc1 = BiasVariable(shape=(64 * 16* 16,),
                                     name='b_fc1',
                                     #model_scope=self.model_scope,
                                     layer_scope=layer_scope,
                                     initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                stddev=0.02),
                                     summary_update_freq=self.variable_summary_update_freq,
                                     )(step)

                # Implement dense layer:
                out_fc1 = tf.nn.bias_add(tf.matmul(x, W_fc1), b_fc1)

                # Batch Normalization
                #batch_norm_out_fc1 = tf.nn.batch_normalization(out_fc1, training=
                batch_norm_out_fc1 = BatchNormalization(
                                                name='BatchNorm_fc1',
                                                #model_scope=self.model_scope,
                                                layer_scope=layer_scope,
                                                summary_update_freq=self.variable_summary_update_freq,
                                                )(out_fc1, step)

                # Activation of output
                act_out_fc1 = tf.nn.relu(batch_norm_out_fc1)

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
                with tf.name_scope('residual_block{}'.format(i)) as layer_scope:

                    # first 2-D Convolutional Layer:
                    # Initializing filter for first conv layer
                    kernel1_res = WeightVariable(shape=[3, 3, 64, 64],
                                                 name='Filter1_resblock{}'.format(i),
                                                 #model_scope=self.model_scope,
                                                 layer_scope=layer_scope,
                                                 initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                            stddev=0.02),
                                                 summary_update_freq=self.variable_summary_update_freq,
                                                )(step)
                    # Initializing bias parameters for first conv layer
                    bias1_res = BiasVariable(shape=(64,),
                                             name='bias1_resblock{}'.format(i),
                                             #model_scope=self.model_scope,
                                             layer_scope=layer_scope,
                                             initializer=tf.initializers.TruncatedNormal(mean=0.02,
                                                                                        stddev=0.02),
                                             summary_update_freq=self.variable_summary_update_freq,
                                             )(step)

                    # output shape: (batch_size, 16, 16, 64)
                    feature_map1_res = tf.nn.conv2d(residual_input,
                                                    kernel1_res,
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME'
                                                    )
                    # batch_normalization
                    # output shape: (batch_size, 16, 16, 64)
                    batch_norm_feature_map_1 = tf.nn.batch_normalization(feature_map1_res,
                                                                         mean=0.0,
                                                                         variance=1.0,
                                                                         offset=None,
                                                                         scale=None,
                                                                         variance_epsilon=0.001)

                    # first activation
                    # output shape: (batch_Size, 16, 16, 64)
                    act_feature_map_1 = tf.nn.relu(batch_norm_feature_map_1)

                    # 2nd 2-D Convolutional Layer:
                    # initializing second filter
                    kernel2_res = WeightVariable(shape=[3, 3, 64, 64],
                                                 name='Filter2_resblock{}'.format(i),
                                                 #model_scope=self.model_scope,
                                                 layer_scope=layer_scope,
                                                 initializer=tf.initializers.TruncatedNormal(mean=0.02,
                                                                                            stddev=0.02)
                                                 )(step)
                    bias2_res = BiasVariable(shape=(64,),
                                             name='bias2_resblock{}'.format(i),
                                             #model_scope=self.model_scope,
                                             layer_scope=layer_scope,
                                             initializer=tf.initializers.TruncatedNormal(mean=0.02,
                                                                                        stddev=0.02)
                                             )(step)
                    # output shape: (batch_size, 16, 16, 64)
                    feature_map2_res = tf.nn.conv2d(act_feature_map_1,
                                                    kernel2_res,
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME'
                                                    )
                    # Add bias tensor
                    bias_feature_map2_res = tf.nn.bias_add(feature_map2_res, bias2_res)

                    # batch_normalization
                    # output shape: (batch_size, 16, 16, 64)
                    batch_norm_feature_map2_res = tf.nn.batch_normalization(feature_map2_res,
                                                                            mean=0.0,
                                                                            variance=1.0,
                                                                            offset=None,
                                                                            scale=None,
                                                                            variance_epsilon=0.001)

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
                batch_norm_residual_block_output = tf.nn.batch_normalization(residual_block_output,
                                                                             mean=0.0,
                                                                             variance=1.0,
                                                                             offset=None,
                                                                             scale=None,
                                                                             variance_epsilon=0.001)

                # ReLU Activation
                # SHape: (batch_size, 16, 16, 64)
                act_residual_block_output = tf.nn.relu(batch_norm_residual_block_output)

                # Final element-wise sum, with original res block input (i.e., the output of the
                # first fully-connected layer
                # Shape: (batch_size, 16, 16, 64)
                fm_intermediate = tf.add(act_residual_block_output, re_out_fc1)

            ############################################################################
            ######################### END RESIDUAL LAYERS ##############################
            ############################################################################

            # Upsampling sub-pixel convolution: scales the input tensor by 2 in both the
            #           x and y dimension and randomly shuffles pixels in those dimensions
            for i in range(1, 4):
                with tf.name_scope('upsampling_subpixel_convolution{}'.format(i)) as layer_scope:

                    # Initializer filter and bias for upsampling convolution
                    upscale_kernel = WeightVariable(shape=[3, 3, 64, 256],
                                                    name='Filter1_PixelShuffle',
                                                    #model_scope=self.model_scope,
                                                    layer_scope=layer_scope,
                                                    initializer=tf.initializers.TruncatedNormal(mean=0.02,
                                                                                               stddev=0.02)
                                                    )(step)
                    upscale_bias = BiasVariable(shape=(256,),
                                                name='bias_PixelShuffle{}'.format(i),
                                                #model_scope=self.model_scope,
                                                layer_scope=layer_scope,
                                                initializer=tf.initializers.TruncatedNormal(mean=0.02,
                                                                                           stddev=0.02)
                                                )(step)

                    # Output shape: (batch_size, 16, 16, 256)
                    upscale_feature_map = tf.nn.conv2d(fm_intermediate,
                                                       upscale_kernel,
                                                       strides=[1, 1, 1, 1],
                                                       padding='SAME'
                                                       )

                    # Perform pixel shuffling operation
                    pix_shuffled_fm = self.pixel_shuffle_x2_layer(upscale_feature_map)

                    # batch normalization and final activation
                    bn_pix_shuffled_fm = tf.nn.batch_normalization(pix_shuffled_fm,
                                                                   mean=0.0,
                                                                   variance=1.0,
                                                                   offset=None,
                                                                   scale=None,
                                                                   variance_epsilon=0.001)
                    fm_intermediate = tf.nn.relu(bn_pix_shuffled_fm)

            # Shape of last intermediate feature map output by the upsampling sub-pixel
            # convolution layers
            # (batch_size, 128, 128, 64)

            # Final convolution layer
            with tf.name_scope('final_convolution') as layer_scope:

                final_conv_kernel = WeightVariable(shape=[9, 9, 64, 3],
                                                   name='Filter_final',
                                                   #model_scope=self.model_scope,
                                                   layer_scope=layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.02,
                                                                                              stddev=0.02)
                                                   )(step)
                final_conv_bias = BiasVariable(shape=(3,),
                                               name='bias_final',
                                               #model_scope=self.model_scope,
                                               layer_scope=layer_scope,
                                               initializer=tf.initializers.TruncatedNormal(mean=0.02,
                                                                                          stddev=0.02)
                                               )(step)

                final_conv_fm = tf.nn.conv2d(fm_intermediate,
                                             final_conv_kernel,
                                             strides=[1, 1, 1, 1],
                                             padding='SAME'
                                             )

                bias_final_conv_fm = tf.nn.bias_add(final_conv_fm, final_conv_bias)

                # Shape: (batch_size, 128, 128, 3)
                output = tf.nn.sigmoid(final_conv_fm)

            return output

    def pixel_shuffle_x2_layer(self, input_fm):
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

        return pix_shuffle_x2_output

    def __call__(self, x, step=0):
        """
        When model is called like a function, initiate forward pass
        """
        return self.forward_pass(x, step=step)

#    @tf.function
#    def pixel_shuffle_x2_layer(self, input_fm):
#        """
#        Applies pixel shuffling to upsampled feature map.
#        For an input of x256 channels, new feature maps will be composed using the
#        next x4 channels in iteration.
#
#        :param input_fm: input tensor of shape -- (batch_size, fm_x, fm_y, 256)
#
#        :return out: output tensor of shape -- (batch_size, 2 * fm_x, 2 * fm_y, 64)
#        """
#        fm_shape = tf.shape(input_fm)
#        num_channels = fm_shape[3]
#
#        def apply_pix_shuffling_to_batch(batch):
#
#            # Compose a shuffled pixel map with every 4xgroup of input feature
#            # maps by iterating over the channel dim in intervals of 4:
#            i = tf.constant(0)
#            cond = lambda i, iter_batch: tf.less(i, num_channels // 4)
#
#            def gather_shuffled_pixels(iterable, iter_batch):
#
#                # Compose indices that will be needed to perform pixel shuffling operation
#                x_dim, y_dim, channel_dim = tf.meshgrid(tf.range(tf.shape(input_fm)[0]),
#                                                        tf.range(tf.shape(input_fm)[1]),
#                                                        tf.range(4*iterable, 4*(iterable+1))
#                                                        )
#
#                coords = tf.stack([x_dim, y_dim, channel_dim], axis=-1)
#
#                # Gather the pixels from a 4x group of feature maps
#                # shape: (2 * fm_x, 2* fm_y, 1)
#                upsampled_shuffled_feature_map = tf.gather_nd(iter_batch, coords)
#
#                # update iterator
#                iterable = tf.add(iterable, 1)
#                tf.print("(pjt) iterable: ", iterable)
#
#                return tf.add(iterable, 1), upsampled_shuffled_feature_map
#
#            # Perform pixel shuffling operation
#            # input shape: (fm_x, fm_y, 256)
#            # output shape: (2 * fm_x, 2 * fm_y, 64)
#            pixel_shuffled_batch = tf.while_loop(cond, gather_shuffled_pixels, i)[1]
#
#            return pixel_shuffled_batch
#
#        # Maps pixel shuffling operation to a single batch.
#        # Input shape: (batch_size, fm_x, fm_y, 256)
#        # Output shape: (batch_size, 2 * fm_x, 2 * fm_y, 64)
#        pix_shuffled_feature_maps = tf.map_fn(apply_pix_shuffling_to_batch, input_fm)
#
#        return pix_shuffled_feature_maps































