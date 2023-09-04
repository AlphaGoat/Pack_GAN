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

from PACK_GAN.models.layers import WeightVariable, BiasVariable, BatchNormalization

class SRResNet(object):

    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 latent_space_vector_dim,
                 num_tags,
                 batch_size=1,
                 variable_summary_update_freq=10,
                 model_scope="SRResNet_Generator"):

        # Provide dims of image we would like to generate
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels

        # Provide dims of input
        self.latent_space_vector_dim = latent_space_vector_dim
        self.num_tags = num_tags
        self.batch_size = batch_size

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
        # Weight variable
        # shape: (batch_size, width * height * channels, 64 * 16 * 16)
        input_shape = self.latent_space_vector_dim + self.num_tags
        x = self.fully_connected_layer(x,
                                       input_shape=input_shape,
                                       output_shape=(64 * 16 * 16),
                                       layer_name="fully_connected",
                                       step=step)

        # Batch Normalization
        batch_norm_out_fc1 = BatchNormalization(
                                        name='BatchNorm_fc1',
#                                        layer_scope=layer_scope,
                                        summary_update_freq=self.variable_summary_update_freq,
                                        )(x, step)

        # Activation of output
        act_out_fc1 = tf.nn.relu(batch_norm_out_fc1)

        # Reshape output of dense layer to feed into resblocks
        # NOTE: the output of this first dense layer will be saved and used
        #       at the end of the following residual blocks as the last
        #       input
        # Shape: (batch_size, 16, 16, 64)
        x = residual_input = tf.reshape(act_out_fc1, [-1, 16, 16, 64])
        import pdb; pdb.set_trace()

        # Initialize 16 Residual blocks
        for i in range(1, 17):
            with tf.name_scope('residual_block{}'.format(i)) as layer_scope:

                x = self.resblock(x,
                                  filter_dims=[3, 3],
                                  num_filters=64,
                                  input_channels=64,
                                  strides=[1, 1, 1, 1],
                                  layer_scope=layer_scope
                                  )
         # Shape: (batch_Size, 16, 16, 64)
        import pdb; pdb.set_trace()
        x = tf.nn.batch_normalization(x,
                                      mean=0.0,
                                      variance=1.0,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=0.001)
        x = tf.nn.relu(x)
        x = tf.add(x, residual_input)
        import pdb; pdb.set_trace()

        # Upsampling sub-pixel convolution: scales the input tensor by 2 in both the
        #           x and y dimension and randomly shuffles pixels in those dimensions
        for i in range(1, 4):
            with tf.name_scope('upsampling_subpixel_convolution{}'.format(i)) as layer_scope:

                # Initializer filter and bias for upsampling convolution
                x = self.pixel_shuffle_block(x, layer_scope=layer_scope, step=step)
        import pdb; pdb.set_trace()

        # Shape of last intermediate feature map output by the upsampling sub-pixel
        # convolution layers
        # (batch_size, 128, 128, 64)
        # Final convolution layer
        with tf.name_scope('final_convolution') as layer_scope:

            x = self.conv2d(x,
                            filter_dims=[9, 9],
                            num_filters=3,
                            input_channels=64,
                            strides=[1, 1, 1, 1],
                            name="conv1_1",
                            layer_scope=layer_scope)

            # Shape: (batch_size, 128, 128, 3)
            output = tf.nn.sigmoid(x)
        import pdb; pdb.set_trace()
        return output

    def fully_connected_layer(self,
                              x,
                              input_shape,
                              output_shape,
                              layer_name,
                              step=0):
        # Weight variable
        weight_shape = (input_shape, output_shape)
        W_fc1 = WeightVariable(shape=weight_shape,
                               variable_name='W_fc1',
                               #model_scope=self.model_scope,
                               layer_name=layer_name,
                               initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                          stddev=0.02),
                               summary_update_freq=self.variable_summary_update_freq,
                               )(step)

        # Initialize bias variable
        # Shape: (batch_size, 64 * 16 * 16)
        b_fc1 = BiasVariable(shape=(output_shape,),
                             variable_name='b_fc1',
                             #model_scope=self.model_scope,
                             layer_name=layer_name,
                             initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                        stddev=0.02),
                             summary_update_freq=self.variable_summary_update_freq,
                             )(step)

        # Implement dense layer:
        out_fc1 = tf.nn.bias_add(tf.matmul(x, W_fc1), b_fc1)
        return out_fc1

    def conv2d(self,
               x,
               filter_dims,
               num_filters,
               input_channels,
               name,
               strides=[1, 1, 1, 1],
               layer_scope=None,
               step=0):

        assert len(filter_dims) == 1 or len(filter_dims) == 2
        if len(filter_dims) == 2: dim_x, dim_y = filter_dims
        else: dim_x = dim_y = filter_dims

        weight_initializer = tf.initializers.TruncatedNormal(mean=0.0,
                                                             stddev=0.02)

        kernel = WeightVariable(shape=[dim_x, dim_y, input_channels, num_filters],
                                variable_name='kernel',
                                layer_name=name,
                                scope=layer_scope,
                                initializer=weight_initializer,
                                )(step)

        bias = BiasVariable(shape=(num_filters,),
                            variable_name='bias',
                            layer_name=name,
                            scope=layer_scope,
                            initializer=weight_initializer)(step)

        fm = tf.nn.conv2d(x, kernel, strides=strides, padding='SAME')
        fm = tf.nn.bias_add(fm, bias)

        return fm

    def resblock(self,
                 x,
                 filter_dims,
                 num_filters,
                 input_channels,
                 layer_scope,
#                 name,
                 strides=[1, 1, 1, 1],
#                 layer_scope=None,
                 step=0):

        # Store input as residual
        res_input = x

        # First Convolution
        x = self.conv2d(x, filter_dims, num_filters, input_channels,
                        name="conv1_1", strides=strides, layer_scope=layer_scope,
                        step=step)
        x = BatchNormalization(
            name=layer_scope + "_batch_norm1_1",
            summary_update_freq=self.variable_summary_update_freq,
        )(x, step)

        x = tf.nn.relu(x)

        # Second Convolution
        x = self.conv2d(x, filter_dims, num_filters, input_channels,
                        name="conv1_2", strides=strides, layer_scope=layer_scope,
                        step=step)

        x = BatchNormalization(
            name=layer_scope + "_batch_norm1_2",
            summary_update_freq=self.variable_summary_update_freq,
        )(x, step)
        residual_sum = tf.add(res_input, x)
        output = tf.nn.relu(residual_sum)

        return output

    def pixel_shuffle_block(self,
                            x,
                            layer_scope=None,
                            step=0):
        x = self.conv2d(x,
                        filter_dims=[3, 3],
                        num_filters=256,
                        input_channels=64,
                        name="conv1_1",
                        strides=[1, 1, 1, 1],
                        layer_scope=layer_scope)
        x = self.pixel_shuffle_x2_layer(x)
        x = BatchNormalization(
            name=layer_scope + "batch_norm1_1",
            summary_update_freq=self.variable_summary_update_freq,
        )(x, step=step)

        output = tf.nn.relu(x)

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
        pix_shuffle_xdim = tf.reshape(input_fm, [self.batch_size, 2 * fm_x, fm_y, -1])

        # Perform transpose and reshape tensor to combine the 2x remaining channels along
        # the y-dim
        pix_shuffle_x2_output = tf.reshape(tf.transpose(pix_shuffle_xdim, perm=[0, 2, 1, 3]),
                                           [self.batch_size, 2 * fm_x, 2 * fm_y, -1])

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































