"""
Test script to run full DRAGAN architecture

31 Dec 2019

Peter J. Thomas
"""
import tensorflow as tf

import os
import argparse

from loss.loss import DRAGANLoss
from models.generator import Generator
from models.discriminator import Discriminator

def main(flags):

    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    with tf.device('/cpu:0'):
        train_tfrecord_name = os.path.abspath(flags.train_tfrecord)
        valid_tfrecord_name = os.path.abspath(flags.valid_tfrecord)
        test_tfrecord_name = os.path.abspath(flags.test_tfrecord)

    # Initialize data generators
    # data generators for real imagery:

    num_train_images = flags.num_train_images
    num_valid_images = flags.num_valid_images
    num_test_images = flags.num_test_images

    train_batch_size = flags.train_batch_size
    valid_batch_size = flags.valid_batch_size
    test_batch_size = flags.test_batch_size

    train_real_data_generator = DatasetGenerator(train_tfrecord_name,
                                                 num_train_images,
                                                 batch_size=train_batch_size,
                                                 label_smoothing=flags.label_smoothing,
                                                 num_threads=1,
                                                 buffer=30,
                                                 return_filename=False)
    train_iterator = train_real_data_generator.get_iterator()

    valid_real_data_generator = DatasetGenerator(valid_tfrecord_name,
                                                 num_valid_images,
                                                 batch_size=valid_batch_size,
                                                 num_threads=1,
                                                 buffer=30,
                                                 return_filename=False)
    valid_iterator = valid_real_data_generator.get_iterator()

    test_real_data_generator = DatasetGenerator(test_tfrecord_name,
                                                 num_test_images,
                                                 batch_size=test_batch_size,
                                                 num_threads=1,
                                                 buffer=30,
                                                 return_filename=False)
    test_iterator = test_real_data_generator.get_iterator()

    # Initialize Discriminator and Generator models

    # variable to store how many often we would like to see tensorboard
    # summaries generated for the weights of the respective models
    parameter_summary_update_freq = flags.parameter_summary_update_freq

    # First grab the resolutions of the images we would like to generate
    image_width = flags.generated_image_resolution[0]
    image_height = flags.generated_image_resolution[1]
    image_channels = flags.generated_image_resolution[2]

    # If a latent space vector dim for input noise for Generator was provided, use
    # that. Otherwise, just use the image width from the provided image resolution
    if flags.latent_space_vector_dim:
        latent_space_vector_dim = flags.latent_space_vector_dim

    else:
        latent_space_vector_dim = image_width

    # fetch the number of tags we will use to label/generate imagery
    num_tags = flags.num_tags

    dragan_generator = Generator(image_width,
                                 image_height,
                                 image_channels,
                                 latent_space_vector_dim,
                                 num_tags,
                                 variable_summary_update_freq=parameter_summary_update_freq,
                                 )

    dragan_discriminator = Discriminator(image_width,
                                         image_height,
                                         image_channels,
                                         num_tags,
                                         variable_summary_update_freq=parameter_summary_update_freq,
                                         )

    # Unpack balance factors for loss function
    adversarial_balance_factor = flags.adversarial_balance_factor
    gradient_penalty_balance_factor = flags.gradient_penalty_balance_factor

    # Initialize Loss function
    dragan_loss = DRAGANLoss(adv_balance_factor=adversarial_balance_factor,
                             gp_balance_factor=gradient_penalty_balance_factor)

    # Initialize optimization function for Generator
    generator_optimizer = tf.optimizers.SGD(learning_rate=flags.gan_learning_rate,
                                            momentum=flags.gan_momentum,
                                            nesterov=flags.gan_nesterov)

    # Initialize optimization function for Discriminator
    discriminator_optimizer = tf.optimizers.SGD(learning_rate=flags.discriminator_learning_rate,
                                                momentum=flags.discriminator_momentum,
                                                nesterov=flags.discriminator_nesterov)

    # Initialize summary writer
    logdir = flags.logdir
    writer = tf.summary.create_file_writer(flags.logdir)

    with writer.as_default():

        # Initiate training loop
        for epoch in range(flags.num_epochs):

            for step in range(len(train_real_data_generator)):

                # Retrieve next batch of imagery
                train_batch = train_iterator.get_next()

                images, tags = train_batch[0], train_batch[1]












