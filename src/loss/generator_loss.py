import tensorflow as tf

class GeneratorLoss(object):
    """
    Class encompassing the loss functions used to train both the
    discriminator and the generator. The disciminator seeks to
    maximize the log probability for real images as well as the inverse
    probability of generated images. In turn, the generator wants to
    minimize the log of the inverse probability the discriminator
    assigns to generated imagery. We also incorporate class losses
    to train the discriminator to correctly tag imagery as well as the
    generator to produce convincing fakes displaying attributes of the
    tags we assign to the noise matrix input
    """
    def __init__(self,
                 adv_balance_factor=34,
                 gp_balance_factor=0.05,
                 ):

        # adversarial training balance factor (note: in the paper Zhout et. al 2017,
        # it is detailed that the best value to choose for this hyperparameter is
        # the number of tags (attributes) that we are assigning to the problem space
        self.adv_lambda = adv_balance_factor

        # Gradient penalty factor applied to discrinator loss
        self.gp_lambda = gp_balance_factor

    def adversarial_generator_loss(self,
                                   y_pred_generated,
                                   offset=1e-8,
                                   ):
        """
        Calculates log loss for generator, modified to maintain stronger gradient early
        in training loop
        """
        # Retrieve the batch_size of the sample
        batch_size = tf.shape(y_pred_generated)[0]
        log_loss = -(1 / batch_size) * tf.math.reduce_sum(tf.math.log(y_pred_generated + offset))

        return log_loss

    def class_loss(self,
                   pred_cls_real,
                   pred_cls_gen,
                   truth_cls_real,
                   assigned_cls_gen,
                   offset=1e-8,
                   ):
        """
        Classification loss for discriminator

        :param pred_cls_real: Class labels discriminator assigned to batch of real imagery

        :param pred_cls_gen: Class labels discriminator assigned to batch of generated imagery

        :param truth_cls_real: Truth class labels for batch of real imagery

        :param assigned_cls_gen: labels assigned to noise input to generator
        """
        real_component =  (truth_cls_real) * tf.math.log(pred_cls_real + offset) + \
            (1 - truth_cls_real) * tf.math.log(1 - pred_cls_real + offset)

        gen_component =  -(assigned_cls_gen) * tf.math.log(pred_cls_gen + offset) + \
                                          (1 - assigned_cls_gen) * tf.math.log(pred_cls_gen + offset)

        batch_size = tf.shape(real_component)[0]

        # Ensure that the batch size of the real sample is the same as the generated sample
        assert batch_size == tf.shape(gen_component)[0]

        real_component = (1/batch_size) * real_component
        gen_component = (1/batch_size) * gen_component

        return real_component, gen_component

    def gradient_penalty(self, x_real, gen_images, y_real, y_gen, offset=1e-8):
        """
        Implements gradient penalty term for DRAGAN
        """
        # Get the batch size and check that it is the same between the real data distribution
        # and the generated distribution
        batch_size = tf.shape(x_real)[0]
        assert batch_size == tf.shape(gen_images)[0]

        # randomly sample members of the 'real' data distribution and noise (generated images)
#        flat_x = tf.reshape(x_real, [batch_size, -1])
#        flat_gen_images = tf.reshape(gen_images, [batch_size, -1])
        combined_distribution = tf.concat([x_real, gen_images], 0)
        combined_outputs = tf.concat([y_real, y_gen], 0)

        # Craft probability distribution for sampling real and generated inistance
        # distributions
        probability_dist = tf.math.log(tf.ones(2 * batch_size)/(2 * batch_size))

        # sample from this distribution. Use the returned indices to gather elements
        # from the combined real and generated distribution
        sampled_indices = tf.random.categorical(probability_dist, batch_size)
        sampled_distribution = tf.gather(combined_distribution, sampled_indices)

        # Gather discriminator outputs for the sampled input distribution
        sampled_outputs = tf.gather(combined_outputs, sampled_indices)

        # Take the gradient of this distribution with respect to the original input
        grad_sampled_dist = tf.gradients(sampled_distribution, sampled_outputs)

        # Take euclidean norm of calculated gradients
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_sampled_dist), axis=[1, 2, 3]) + offset)

        # Calculate gradient penalty term
        grad_penalty = tf.reduce_mean(tf.square((grad_norm - 1)))

        return grad_penalty

    def loss(self,
             x_real,
             gen_images,
             y_real,
             y_gen,
             class_confidences_real,
             class_confidences_gen,
             truth_classes_real,
             assigned_classes_gen
             ):

        # Calculating the discriminator loss
        adv_gen_loss = self.adversarial_generator_loss(y_gen)
        cls_gen_loss =  self.class_loss(class_confidences_real,
                                                       class_confidences_gen,
                                                       truth_classes_real,
                                                       assigned_classes_gen)

        # calculate gradient penalty term
        grad_penalty = self.gradient_penalty(x_real, gen_images, y_real, y_gen)

        # calculate full loss terms for the generator and discriminator
        generator_loss = cls_gen_loss + (self.adv_lambda * adv_gen_loss)

        return generator_loss

    def __call__(self,
                 x_real,
                 x_gen,
                 y_real,
                 y_gen,
                 class_confidences_real,
                 class_confidences_gen,
                 truth_classes_real,
                 assigned_classes_gen
                 ):
        """
        When DRAGANLoss object is called, return loss values for discriminator and generator
        """
        generator_loss = self.loss(x_real, x_gen, y_real, y_gen,
                                                       class_confidences_real, class_confidences_gen,
                                                       truth_classes_real, assigned_classes_gen)

        return generator_loss



