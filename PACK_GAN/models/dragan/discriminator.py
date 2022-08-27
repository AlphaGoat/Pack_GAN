"""
Implements ResNet generator architecture, as detailed in
'Towards the Automatic Anime Characters Creation with Generative
Adversarial Neural Networks', Yanghua et. al 2017
--------------------------
[INSERT ARXIV LINK HERE]
-------------------------

Peter J. Thomas
12 December 2019
"""
import tensorflow as tf

from PACK_GAN.models.layers import WeightVariable, BiasVariable

class Discriminator(object):

    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 num_tags,
                 variable_summary_update_freq=10):

        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels

        self.num_tags = num_tags

        # how often do we want to see tensorboard summaries for model parameters?
        self.variable_summary_update_freq = variable_summary_update_freq

    def forward_pass(self, x, step):
        """
        :param x: input image batch (either from real dataset or generator)

                  shape: (batch_size, image_width, image_height, image_channels)

        :return truth_score: scalar value detailing confidence the discriminator
                             places on the image being real, where 1 is utmost
                             confidence that the image is real, whereas 0 is utmost
                             confidence that the image is generated

                             shape: (batch_size,)

        :return tags_score: vector detailing confidence that the discriminator places
                            on a certain tag detailing an image
        """
        # initial convolution
        with tf.name_scope('initial_conv') as layer_scope:

            initial_kernel = WeightVariable(shape=[4, 4, 3, 32],
                                            variable_name='Filter_initial',
                                            #model_scope=self.model_scope,
                                            layer_name=layer_scope,
                                            scope=layer_scope,
                                            initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                       stddev=0.02)
                                            )(step)

            initial_bias = BiasVariable(shape=(32,),
                                        variable_name='bias_initial',
                                        #model_scope=self.model_scope,
                                        layer_name=layer_scope,
                                        scope=layer_scope,
                                        initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                   stddev=0.02)
                                        )(step)

            feature_map = tf.nn.conv2d(x, initial_kernel, strides=[1, 2, 2, 1], padding='SAME')

            bias_feature_map = tf.bias_add(feature_map, initial_bias)

            residual_input = act_fm = tf.nn.leaky_relu(bias_feature_map)

        # First series of ResBlocks
        for i in range(2):
            with tf.name_scope('(k3n32s1) ResBlock1_pass{}'.format(i)) as layer_scope:

                # First convolutional layer in ResBlock 1
                res_kernel1 = WeightVariable(shape=[3, 3, 32, 32],
                                              variable_name='res1_filter1',
                                              layer_name=layer_scope,
                                              scope=layer_scope,
                                              initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                         stddev=0.02)
                                              )

                res_bias1 = BiasVariable(shape=(32,),
                                          variable_name='res1_bias1',
                                          layer_name=layer_scope,
                                          scope=layer_scope,
                                          initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                     stddev=0.02)
                                          )

                res_fm1 = tf.nn.conv2d(residual_input, res_kernel1, strides=[1, 1, 1, 1], padding='SAME')

                bias_res_fm1 = tf.bias_add(res_fm1, res_bias1)

                act_res1_fm1 = tf.nn.leaky_relu(bias_res_fm1)

                # Second convolutional layer in ResBlock
                res_kernel2 = WeightVariable(shape=[3, 3, 32, 32],
                                              variable_name='res1_filter2',
                                              layer_name=layer_scope,
                                              scope=layer_scope,
                                              initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                         stddev=0.02)
                                              )

                res_bias2 = BiasVariable(shape=(32,),
                                          variable_name='res1_bias2',
                                          layer_name=layer_scope,
                                          scope=layer_scope,
                                          initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                     stddev=0.02)
                                          )

                res_fm2 = tf.nn.conv2d(act_fm, res_kernel2, strides=[1, 1, 1, 1], padding='SAME')

                bias_res_fm2 = tf.bias_add(res_fm2, res_bias2)

                # Elementwise sum with residual input
                residual_sum = tf.add(bias_res_fm2, residual_input)

                # Final ResBlock activation
                residual_input = residual_output = tf.nn.leaky_relu(residual_sum)

        # Convolutional layer 'bridging gap' between first and second set of ResBlocks. This
        # conv layer has the effect of blowing up the number of channels x2 as well
        with tf.name_scope('bridge_conv_layer1') as layer_scope:
            bridge1_kernel = WeightVariable(shape=[4, 4, 32, 64],
                                            variable_name='bridge1_kernel',
                                            layer_name=layer_scope,
                                            scope=layer_scope,
                                            initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                       stddev=0.02)
                                            )

            bridge1_bias = BiasVariable(shape=(64,),
                                       variable_name='bridge_bias',
                                       layer_name=layer_scope,
                                       scope=layer_scope,
                                       initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                  stddev=0.02)
                                       )

            bridge1_fm = tf.nn.conv2d(residual_output, bridge1_kernel, strides=[2, 2, 2, 2], padding='SAME')

            bias_bridge1_fm = tf.bias_add(bridge1_fm, bridge1_bias)

            residual_input = bridge1_output = tf.nn.leaky_relu(bias_bridge1_fm)

        # Second ResBlock (filter: (3x3), stride: (1, 1, 1, 1), num_filters: 64)
        for i in range(2):
            with tf.name_scope('(k3n64s1) ResBlock2_pass{}'.format(i)) as layer_scope:
                # 1st conv layer in ResBlock
                resblock2_kernel1 = WeightVariable(shape=[3, 3, 64, 64],
                                                   variable_name='resblock2_kernel1_pass{}'.format(i),
                                                   layer_name=layer_scope,
                                                   scope=layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                              stddev=0.02)
                                                   )

                resblock2_bias1 = BiasVariable(shape=(64,),
                                               variable_name='resblock2_bias1_pass{}'.format(i),
                                               layer_name=layer_scope,
                                               scope=layer_scope,
                                               initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                          stddev=0.02)
                                               )

                resblock2_fm1 = tf.nn.conv2d(residual_input, resblock2_kernel1,
                                             strides=[1, 1, 1, 1], padding='SAME')

                bias_resblock2_fm1 = tf.bias_add(resblock2_fm1, resblock2_bias1)

                act_resblock2_fm1 = tf.nn.leaky_relu(bias_resblock2_fm1)

                # 2nd conv layer in ResBlock
                resblock2_kernel2 = WeightVariable(shape=[3, 3, 64, 64],
                                                   variable_name='resblock2_kernel2_pass{}'.format(i),
                                                   layer_name=layer_scope,
                                                   scope=layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                              stddev=0.02)
                                                   )

                resblock2_bias2 = BiasVariable(Shape=(64,),
                                               variable_name='resblock2_bias2_pass{}'.format(i),
                                               layer_name=layer_scope,
                                               scope=layer_scope,
                                               initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                          stddev=0.02)
                                               )

                resblock2_fm2 = tf.nn.conv2d(act_resblock2_fm1, resblock2_kernel2,
                                             strides=[1, 1, 1, 1], padding='SAME')

                # Elementwise sum with residual input (the feature map input at the beginning of
                # the ResBlock
                elementwise_sum_resblock2 = tf.add(resblock2_fm2, residual_input)

                # Final activation before feeding into next ResBlock/ bridge conv layer
                residual_output = residual_input = tf.nn.leaky_relu(elementwise_sum_resblock2)

        # Second bridge conv layer between two ResBlocks
        with tf.name_scope('bridge_conv_layer2') as layer_scope:
            bridge2_kernel = WeightVariable(shape=[4, 4, 64, 128],
                                            variable_name='bridge2_kernel',
                                            layer_name=layer_scope,
                                            scope=layer_scope,
                                            initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                       stddev=0.02)
                                            )

            bridge2_bias = BiasVariable(shape=(128,),
                                        variable_name='bridge2_bias',
                                        layer_name=layer_scope,
                                        scope=layer_scope,
                                        initializer=tf.initialzier.TruncatedNormal(mean=0.0,
                                                                                   stddev=0.02)
                                        )

            bridge2_fm = tf.nn.conv2d(residual_output, bridge2_kernel,
                                      strides=[2, 2, 1, 1], padding='SAME')

            residual_input = bridge2_output = tf.nn.leaky_relu(bridge2_fm)

        # Initialize 3rd ResBlock
        with tf.name_scope('(k3n128s1) ResBlock3') as block_scope:
            for i in range(1,3):
                with tf.name_scope('(k3n128s1) ResBlock3_pass{}'.format(i)) as layer_scope:

                    # First conv layer
                    resblock3_kernel1 = WeightVariable(shape=[3, 3, 128, 128],
                                                       variable_name='resblock3_kernel1_pass{}'.format(i),
                                                       layer_name=layer_scope,
                                                       scope=block_scope + layer_scope,
                                                       initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                                  stddev=0.02)
                                                       )

                    resblock3_bias1 = BiasVariable(shape=(128,),
                                                   variable_name='resblock3_bias1_pass{}'.format(i),
                                                   layer_name=layer_scope,
                                                   scope=block_scope +layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                              stddev=0.02)
                                                   )

                    resblock3_fm1 = tf.nn.conv2d(residual_input, resblock3_kernel1,
                                                 strides=[1, 1, 1, 1], padding='SAME')

                    bias_resblock3_fm1 = tf.bias_add(resblock3_fm1, resblock3_bias1)

                    act_resblock3_fm1 = tf.nn.leaky_relu(resblock3_fm1)

                    # Second conv layer
                    resblock3_kernel2 = WeightVariable(shape=[3, 3, 128, 128],
                                                       variable_name='resblock3_kernel2_pass{}'.format(i),
                                                       layer_name=layer_scope,
                                                       scope=block_scope + layer_scope,
                                                       initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                                  stddev=0.02)
                                                       )

                    resblock3_bias2 = BiasVariable(shape=(128,),
                                                   variable_name='resblock3_bias2_pass{}'.format(i),
                                                   layer_name=layer_scope,
                                                   scope=block_scope + layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                              stddev=0.02)
                                                   )

                    resblock3_fm2 = tf.nn.conv2d(act_resblock3_fm1, resblock3_kernel2,
                                                 strides=[1, 1, 1, 1], padding='SAME')

                    bias_resblock3_fm2 = tf.bias_add(resblock3_fm2, resblock3_bias2)

                    # Element-wise summation of input to ResBlock and the final feature map
                    # produced by the second conv layer in the ResBlock
                    elementwise_sum_resblock3 = tf.add(bias_resblock3_fm2, residual_input)

                    # Final ResBlock activation
                    residual_output = residual_input = tf.nn.leaky_relu(elementwise_sum_resblock3)

        # Third bridge convolutional layer
        with tf.name_scope('bridge_conv_layer3') as layer_scope:
            bridge3_kernel = WeightVariable(shape=[3, 3, 128, 256],
                                            variable_name='bridge3_kernel',
                                            layer_scope=layer_scope,
                                            scope=layer_scope,
                                            initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                       stddev=0.02)
                                            )

            bridge3_bias = BiasVariable(shape=(256,),
                                        variable_name='bridge3_bias',
                                        layer_name=layer_scope,
                                        scope=layer_scope,
                                        initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                   stddev=0.02)
                                        )

            bridge3_fm = tf.nn.conv2d(residual_output, bridge3_kernel,
                                      strides=[2, 2, 1, 1], padding='SAME')

            residual_input = act_bridge3_fm = tf.nn.leaky_relu(bridge3_fm)

        # Initialize 4th ResBlock
        with tf.name_scope('(k3n256s1) ResBlock4') as block_scope:
            for i in range(1,3):
                with tf.name_scope('ResBlock4 pass{}'.format(i)) as layer_scope:

                    # 1st Conv Layer
                    resblock4_kernel1 = WeightVariable(shape=[3, 3, 256, 256],
                                                       variable_name='resblock4_kernel1_pass{}'.format(i),
                                                       layer_name=layer_scope,
                                                       scope=block_scope + layer_scope,
                                                       initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                                  stddev=0.02)
                                                       )

                    resblock4_bias1 = BiasVariable(shape=(256,),
                                                   variable_name='resblock4_bias1_pass{}'.format(i),
                                                   layer_name=layer_scope,
                                                   scope=block_scope + layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                              stddev=0.02)
                                                   )


                    resblock4_fm1 = tf.nn.conv2d(residual_input, resblock4_kernel1,
                                                 strides=[1, 1, 1, 1], padding='SAME')

                    bias_resblock4_fm1 = tf.bias_add(resblock4_fm1, resblock4_bias1)

                    # 2nd Conv layer
                    resblock4_kernel2 = WeightVariable(shape=[3, 3, 256, 256],
                                                       variable_name='resblock4_kernel2_pass{}'.format(i),
                                                       layer_name=layer_scope,
                                                       scope=block_scope + layer_scope,
                                                       initializer=tf.initiializer.TruncatedNormal(mean=0.0,
                                                                                                   stddev=0.02)
                                                       )

                    resblock4_bias2 = BiasVariable(shape=(256,),
                                                   variable_name='resblock4_bias2_pass{}'.format(i),
                                                   layer_name=layer_scope,
                                                   scope=block_scope + layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                              stddev=0.02)
                                                   )

                    resblock4_fm2 = tf.nn.conv2d(bias_resblock4_fm1, resblock4_kernel2,
                                                 strides=[1, 1, 1, 1], padding='SAME')


                    # Perform elementwise sum with input to ResBlock (i.e., output to
                    # bridge conv layer or previous ResBlock)
                    elementwise_sum_resblock4 = tf.add(resblock4_fm2, residual_input)

                    # Final activation in ResBlock
                    residual_output = residual_input = tf.nn.leaky_relu(elementwise_sum_resblock4)


        # ..."Hey, look at that... 4th bridge layer
        with tf.name_scope('bridge_conv_layer4') as layer_scope:
            bridge4_kernel = WeightVariable(shape=[3, 3, 256, 512],
                                            variable_name='bridge_conv_layer4_kernel',
                                            layer_name=layer_scope,
                                            scope=layer_scope,
                                            model_scope=self.model_scope,
                                            initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                       stddev=0.02)
                                            )

            bridge4_bias = BiasVariable(shape=(512,),
                                        variable_name='bridge_conv_layer4_bias',
                                        layer_name=layer_scope,
                                        scope=layer_scope,
                                        initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                   stddev=0.02)
                                        )

            bridge4_fm = tf.nn.conv2d(residual_output, bridge4_kernel,
                                      strides=[2, 2, 1, 1], padding='SAME')

            act_bridge4_fm = residual_input = tf.nn.leaky_relu(bridge4_fm)

        # And a wild 5th ResBlock (the last one, I promise) appears
        with tf.name_scope('(k3n512s1) ResBlock5') as block_scope:
            for i in range(1, 3):
                with tf.name_scope('(k3n512s1) ResBlock5_pass{}.'.format(i)) as layer_scope:
                    # 1st Conv layer
                    resblock5_kernel1 = WeightVariable(shape=[3, 3, 512, 512],
                                                       variable_name='resblock5_kernel1_pass{}'.format(i),
                                                       layer_name=layer_scope,
                                                       scope=block_scope + layer_scope,
                                                       initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                                  stddev=0.02)
                                                       )

                    resblock5_bias1 = BiasVariable(shape=(512,),
                                                   variable_name='resblock5_bias1_pass{}'.format(i),
                                                   layer_name=layer_scope,
                                                   scope=block_scope + layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                              stddev=0.02)
                                                   )

                    resblock5_fm1 = tf.nn.conv2d(residual_input, resblock5_kernel1,
                                                 strides=[1, 1, 1, 1], padding='SAME')

                    resblock5_bias_fm1 = tf.bias_add(resblock5_fm1, resblock5_bias1)

                    resblock5_act_fm1 = tf.nn.leaky_relu(resblock5_bias_fm1)

                    # 2nd Conv layer
                    resblock5_kernel2 = WeightVariable(shape=[3, 3, 512, 512],
                                                       variable_name='resblock5_kernel1_pass{}'.format(i),
                                                       layer_name=layer_scope,
                                                       scope=block_scope + layer_scope,
                                                       initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                                  stddev=0.02)
                                                       )

                    resblock5_bias2 = BiasVariable(shape=(512,),
                                                   variable_name='resblock5_bias2_pass{}'.format(i),
                                                   layer_name=layer_scope,
                                                   scope=block_scope + layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                              stddev=0.02)
                                                   )

                    resblock5_fm2 = tf.nn.conv2d(resblock5_act_fm1, resblock5_kernel2,
                                                 strides=[1, 1, 1, 1], padding='SAME')

                    resblock5_bias_fm2 = tf.bias_add(resblock5_fm2, resblock5_bias2)

                    # Elementwise summation with input to residual block
                    resblock5_elementwise_sum = tf.add(resblock5_fm2, residual_input)

                    # Final activation
                    residual5_output = residual_input = tf.nn.leaky_relu(resblock5_elementwise_sum)

        # Initialize final conv layer
        with tf.name_scope('(k3n1024s2) final_conv_layer') as layer_scope:
            final_kernel = WeightVariable(shape=[3, 3, 512, 1024],
                                          variable_name='final_conv_layer_filter',
                                          layer_name=layer_scope,
                                          scope=block_scope + layer_scope,
                                          initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                     stddev=0.02)
                                          )

            final_bias = BiasVariable(shape=(1024,),
                                      variable_name='final_conv_layer_bias',
                                      layer_name=layer_scope,
                                      scope=block_scope + layer_scope,
                                      initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                 stddev=0.02)
                                     )

            final_fm = tf.nn.conv2d(residual5_output, final_kernel,
                                    strides=[2, 2, 1, 1], padding='SAME')

            final_bias_fm = tf.bias_add(final_fm, final_bias)

            final_act_fm = tf.nn.leaky_relu(final_bias_fm)

        # Flatten feature map for read in to final fully-connected layers
        flattened_shape = self.image_width * self.image_height * 1024
        flattened_final_fm = tf.reshape(final_act_fm, [-1, flattened_shape])

        # Final output layer for truth_score
        with tf.name_scope('forgery_score_output_layer') as layer_scope:
            forgery_score_weights = WeightVariable(shape=[flattened_shape, 1],
                                                   variable_name='forgery_score_weights',
                                                   layer_name=layer_scope,
                                                   scope=layer_scope,
                                                   initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                            stddev=0.02)
                                                 )

            forgery_score_bias = BiasVariable(shape=(1,),
                                            variable_name='forgery_score_bias',
                                            layer_name=layer_scope,
                                            scope=layer_scope,
                                            initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                       stddev=0.02)
                                            )

            unactivated_forgery_score = tf.bias_add(tf.matmul(flattened_final_fm, forgery_score_weights),
                                                    forgery_score_bias)

            # output shape: (batch_size,)
            forgery_score = tf.nn.sigmoid(unactivated_forgery_score)

        # Final output layer for tags to assign to input image
        with tf.name_scope('tag_confidence_output_layer') as layer_scope:
            tag_confidence_weights = WeightVariable(shape=[flattened_shape, self.num_tags],
                                                    variable_name='tag_confidence_weights',
                                                    layer_name=layer_scope,
                                                    scope=layer_scope,
                                                    initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                               stddev=0.02)
                                                    )

            tag_confidence_bias = BiasVariable(shape=(self.num_tags,),
                                               variable_name='tag_confidence_bias',
                                               layer_name=layer_scope,
                                               scope=layer_scope,
                                               initializer=tf.initializers.TruncatedNormal(mean=0.0,
                                                                                          stddev=0.02)
                                               )

            unactivated_tag_confidences = tf.bias_add(tf.matmul(flattened_final_fm, tag_confidence_weights),
                                                      tag_confidence_bias)

            # output shape: (batch_size, num_tags)
            tag_confidences = tf.nn.sigmoid(unactivated_tag_confidences)

        # Outputs
        # forgery_scores: (batch_size,)
        # tag_confidences: (batch_size, num_tags)
        return forgery_score, tag_confidences

    def __call__(self, x, step=0):
        """
        When model is called like a function, initiate forward pass
        """
        return self.forward_pass(x, step=step)

