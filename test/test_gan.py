"""
Test script to run full DRAGAN architecture

31 Dec 2019

Peter J. Thomas
"""
import tensorflow as tf

import os
import argparse

from loss.loss import DRAGANLoss
from models.dragan.generator import SRResNet
from models.discriminator import Discriminator
from dataset.real_dataset_generator import DatasetGenerator
from dataset.noise_generator import NoiseGenerator
from utils.tensorboard_plotting import plot_images

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

    train_real_data_generator = DatasetGenerator(train_tfrecord_name,
                                                 num_train_images,
                                                 batch_size=flags.train_batch_size,
                                                 label_smoothing=flags.label_smoothing,
                                                 num_threads=1,
                                                 buffer=30,
                                                 return_filename=False)

    train_iterator = train_real_data_generator.get_iterator()

    valid_real_data_generator = DatasetGenerator(valid_tfrecord_name,
                                                 num_valid_images,
                                                 batch_size=flags.valid_batch_size,
                                                 num_threads=1,
                                                 buffer=30,
                                                 return_filename=False)

    valid_iterator = valid_real_data_generator.get_iterator()

    test_real_data_generator = DatasetGenerator(test_tfrecord_name,
                                                 num_test_images,
                                                 batch_size=flags.test_batch_size,
                                                 num_threads=1,
                                                 buffer=30,
                                                 return_filename=False)

    test_iterator = test_real_data_generator.get_iterator()

    # grab the resolutions of the images we would like to generate
    image_width = flags.generated_image_resolution[0]
    image_height = flags.generated_image_resolution[1]
    image_channels = flags.generated_image_resolution[2]

    # Initialize noise generator
    noise_generator = NoiseGenerator(image_height,
                                     image_width,
                                     image_channels,
                                     flags.num_tags,
                                     flags.latent_space_vector_dim,
                                     flags.train_batch_size,
                                     num_threads=1,
                                     buffer=30)

    noise_iterator = noise_generator.get_iterator()

    # Initialize Discriminator and Generator models

    # variable to store how many often we would like to see tensorboard
    # summaries generated for the weights of the respective models
    parameter_summary_update_freq = flags.parameter_summary_update_freq


    # If a latent space vector dim for input noise for Generator was provided, use
    # that. Otherwise, just use the image width from the provided image resolution
    if flags.latent_space_vector_dim:
        latent_space_vector_dim = flags.latent_space_vector_dim

    else:
        latent_space_vector_dim = image_width

    generator = SRResNet(image_width,
                         image_height,
                         image_channels,
                         latent_space_vector_dim,
                         flags.num_tags,
                         variable_summary_update_freq=parameter_summary_update_freq,
                         )

    discriminator = Discriminator(image_width,
                                  image_height,
                                  image_channels,
                                  flags.num_tags,
                                  variable_summary_update_freq=parameter_summary_update_freq,
                                  )

    # Initialize Loss function
    dragan_loss = DRAGANLoss(adv_balance_factor=flags.adversarial_balance_factor,
                             gp_balance_factor=flags.gradient_penalty_balance_factor)

    # Implement expontential learning rate decay after specified number of  iterations training
    gan_learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(flags.gan_learning_rate,
                                                                          decay_steps=flags.decay_steps,
                                                                          decay_rate=flags.decay_rate,
                                                                          )

    discriminator_learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(flags.discriminator_learning_rate,
                                                                                    decay_steps=flags.decay_steps,
                                                                                    decay_rate=flags.decay_rate,
                                                                                    )

    # Initialize optimization function for Generator
    generator_optimizer = tf.optimizers.Adam(learning_rate=gan_learning_rate_schedule,
                                             beta_1=flags.gan_beta1,
                                             beta_2=flags.gan_beta2)

    # Initialize optimization function for Discriminator
    discriminator_optimizer = tf.optimizers.Adam(learning_rate=discriminator_learning_rate_schedule,
                                                 beta_1=flags.discriminator_beta1,
                                                 beta_2=flags.discriminator_beta2)



    # Initialize summary writer
    logdir = flags.logdir
    writer = tf.summary.create_file_writer(flags.logdir)

    with writer.as_default():

        # Initiate training loop
        running_discriminator_loss = 0.0
        running_generator_loss = 0.0
        for epoch in range(flags.num_epochs):

            for step in range(len(train_real_data_generator)):

                # Retrieve next batch of real imagery
                train_batch = train_iterator.get_next()
                real_images, real_tags = train_batch[0], train_batch[1]

                # Retrieve seed for Generator
                latent_space_noise, gen_tags = noise_iterator.get_next()

                # concatenate latent space noise and tag vector to feed into generator
                gen_input = tf.concat([latent_space_noise, gen_tags], axis=1)

                # Now generate fake images
                gen_images = generator(latent_space_noise, step=step)

                # Feed real data through discriminator and retrieve output
                # forgery scores as well as label confidences
                y_real, real_tags_confidences = discriminator(real_images, step=step)

                # Do the same with generated images
                y_gen, gen_tags_confidences = discriminator(gen_images, step=step)

                # Calculate the losses for the generator and the discriminator
                discriminator_loss, generator_loss = dragan_loss(real_images,
                                                                 gen_images,
                                                                 y_real,
                                                                 y_gen,
                                                                 real_tags_confidences,
                                                                 gen_tags_confidences,
                                                                 real_tags,
                                                                 gen_tags)

                # Perform optimization of both model's parameters
                discriminator_optimizer.minimize(discriminator_loss)

                generator_optimizer.minimize(generator_loss)

                # Add to running losses for both models
                running_discriminator_loss += discriminator_loss
                running_generator_loss += generator_loss

                # Every 1000th step, display statistics
                if step % 1000 == 0:

                    # log running loss for discriminator
                    writer.add_scalar('training_discriminator_loss',
                                      running_discriminator_loss / 1000,
                                      epoch * len(train_real_data_generator) + step)

                    writer.add_scalar('training_generator_loss',
                                      running_generator_loss / 1000,
                                      epoch * len(train_real_data_generator) + step)

                    # log figure displaying discriminator's predictions on a real image
                    # as well as predictions on a generated image
                    writer.add_figure('real image vs. generated image',
                                      plot_images(















