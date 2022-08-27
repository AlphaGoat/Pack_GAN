"""
Test script to run full DRAGAN architecture

31 Dec 2019

Peter J. Thomas
"""
import tensorflow as tf

import argparse
import os
import sqlite3

from PACK_GAN.loss.loss import DRAGANLoss
from PACK_GAN.models.dragan.generator import SRResNet
from PACK_GAN.models.dragan.discriminator import Discriminator
from PACK_GAN.dataset.real_dataset_generator import DatasetGenerator
from PACK_GAN.dataset.noise_generator import NoiseGenerator
from PACK_GAN.utils.tensorboard_plotting import plot_images


def main(flags):

    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Shows what operations are assigned to what device
    tf.debugging.set_log_device_placement(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    # I've been having a bug where TF fails to fetch the convolution algorithm
    # (a pretty big deal with any CNN architecture). I'm not sure if it has to do
    # with the CUDNN_STATUS_INTERNAL_ERROR msg my log files have also been displaying.
    # This line should help with that
#    tf.config.gpu_options.allow_growth = True

    with tf.device('/cpu:0'):
        train_tfrecord_name = os.path.abspath(flags.train_tfrecord)
        valid_tfrecord_name = os.path.abspath(flags.valid_tfrecord)

        if flags.test_tfrecord:
            test_tfrecord_name = os.path.abspath(flags.test_tfrecord)

    # Initialize data generators
    # data generators for real imagery:
    num_train_images = flags.num_train_images
    num_valid_images = flags.num_valid_images
    num_test_images = flags.num_test_images

    # grab the resolutions of the images we would like to generate
    image_width = flags.image_width
    image_height = flags.image_height
    num_channels = flags.image_channels

    train_data_generator = DatasetGenerator(train_tfrecord_name,
                                            num_train_images,
                                            num_channels,
                                            flags.num_tags,
                                            batch_size=flags.train_batch_size,
                                            label_smoothing=flags.label_smoothing,
                                            num_threads=1,
                                            buffer_size=30,
                                            return_filename=False)

#    train_iterator = train_data_generator.get_iterator()

    valid_data_generator = DatasetGenerator(valid_tfrecord_name,
                                            num_valid_images,
                                            num_channels,
                                            flags.num_tags,
                                            batch_size=flags.valid_batch_size,
                                            num_threads=1,
                                            buffer_size=30,
                                            return_filename=False)

#    valid_iterator = valid_data_generator.get_iterator()

    # Initialize testing data generator only if
    if flags.test_tfrecord:
        test_data_generator = DatasetGenerator(test_tfrecord_name,
                                               num_test_images,
                                               num_channels,
                                               flags.num_tags,
                                               batch_size=flags.test_batch_size,
                                               num_threads=1,
                                               buffer_size=30,
                                               return_filename=False)

#        test_iterator = test_data_generator.get_iterator()


    # Initialize noise generator
    noise_generator = NoiseGenerator(image_height,
                                     image_width,
                                     num_channels,
                                     flags.num_tags,
                                     flags.latent_space_vector_dim,
                                     flags.train_batch_size,
                                     num_threads=1,
                                     buffer=30)

#    noise_iterator = noise_generator.get_iterator()

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
                         num_channels,
                         latent_space_vector_dim,
                         flags.num_tags,
                         variable_summary_update_freq=parameter_summary_update_freq,
                         )

    discriminator = Discriminator(image_width,
                                  image_height,
                                  num_channels,
                                  flags.num_tags,
                                  variable_summary_update_freq=parameter_summary_update_freq,
                                  )

    # Initialize Loss function
    dragan_loss = DRAGANLoss(adv_balance_factor=flags.adversarial_balance_factor,
                             gp_balance_factor=flags.gradient_penalty_balance_factor)

    # Implement expontential learning rate decay after specified number of  iterations training
    learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(flags.learning_rate,
                                                                          decay_steps=flags.decay_steps,
                                                                          decay_rate=flags.decay_rate,
                                                                          )

#    gan_learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(flags.learning_rate,
#                                                                          decay_steps=flags.decay_steps,
#                                                                          decay_rate=flags.decay_rate,
#                                                                          )
#
#    discriminator_learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(flags.learning_rate,
#                                                                                    decay_steps=flags.decay_steps,
#                                                                                    decay_rate=flags.decay_rate,
#                                                                                    )

    # Initialize optimization function for Generator
    generator_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule,
                                             beta_1=flags.beta1,
                                             beta_2=flags.beta2)

    # Initialize optimization function for Discriminator
    discriminator_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule,
                                                 beta_1=flags.beta1,
                                                 beta_2=flags.beta2)


    # Initialize summary writer
    logdir = flags.logdir
    writer = tf.summary.create_file_writer(flags.logdir)

    # retrieve list of tags from sqlite database
    if flags.sqlite_database:
        sqlite_connection = sqlite3.connect(flags.sqlite_database)
        cursor = sqlite_connection.execute("select * from image_tags")
        row = cursor.fetchone()
        tag_list = row.keys()

    # If not sqlite database is provided, make a list of integers iterating
    # up to the number of tags
    else:
        tag_list = [*range(flags.num_tags)]

    with writer.as_default():

        # Initiate training loop
        running_discriminator_loss = 0.0
        running_generator_loss = 0.0
        for epoch in range(flags.num_epochs):

            ###############################
            ######### TRAINING LOOP #######
            ###############################

            for step in range(len(train_data_generator)):

                # Retrieve next batch of real imagery
#                train_batch = train_iterator.get_next()
                train_batch = next(iter(train_data_generator.dataset))
                real_images, real_tags = train_batch[0], train_batch[1]

                # Retrieve seed for Generator
                latent_space_noise, gen_tags = next(iter(noise_generator.dataset))

                # concatenate latent space noise and tag vector to feed into generator
                gen_input = tf.concat([latent_space_noise, gen_tags], axis=1)

                # Now generate fake images
                gen_images = generator(gen_input, step=step)

                # Feed real data through discriminator and retrieve output
                # forgery scores as well as label confidences
                y_real, tag_scores_real = discriminator(real_images, step=step)

                # Do the same with generated images
                y_gen, tag_scores_gen = discriminator(gen_images, step=step)

                # Implement chance of switching labels for real and generated
                # images (noisy labels)
                chance = tf.random.uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32)

                # Calculate the losses for the generator and the discriminator
                discriminator_loss, generator_loss = dragan_loss(real_images,
                                                                 gen_images,
                                                                 y_real,
                                                                 y_gen,
                                                                 tag_scores_real,
                                                                 tag_scores_gen,
                                                                 real_tags,
                                                                 gen_tags)

                # Perform optimization of both model's parameters
                discriminator_optimizer.minimize(discriminator_loss)

                generator_optimizer.minimize(generator_loss)

                # Add to running losses for both models
                running_discriminator_loss += discriminator_loss
                running_generator_loss += generator_loss

                # Every 100th step, display statistics
                if step % 100 == 0:

                    # log running loss for discriminator
                    writer.add_scalar('training_discriminator_loss',
                                      running_discriminator_loss / 100,
                                      epoch * len(train_data_generator) + step)

                    writer.add_scalar('training_generator_loss',
                                      running_generator_loss / 100,
                                      epoch * len(train_data_generator) + step)

                    # Select a real image and a generated image to plot, as well
                    # as associated forgery scores and tag confidences
                    real_image_plt = real_images[0,:,:,:]
                    gen_image_plt = gen_images[0,:,:,:]
                    y_real_plt = y_real[0]
                    y_gen_plt = y_gen[0]
                    tag_scores_real_plt = tag_scores_real[0,:]
                    tag_scores_gen_plt = tag_scores_gen[0,:]
                    real_tags_plt = real_tags[0,:]
                    gen_tags_plt = gen_tags[0,:]

                    # log figure displaying discriminator's predictions on a real image
                    # as well as predictions on a generated image
                    writer.add_figure('real image vs. generated image',
                                      plot_images(real_image_plt,
                                                  gen_image_plt,
                                                  y_real_plt,
                                                  y_gen_plt,
                                                  tag_scores_real_plt,
                                                  tag_scores_gen_plt,
                                                  real_tags_plt,
                                                  gen_tags_plt,
                                                  ))


            #################################
            ######### VALIDATION LOOP #######
            #################################
            for step in range(len(valid_data_generator)):

                # Retrieve next batch of real imagery
                #valid_batch = valid_iterator.get_next()
                valid_batch = valid_data_generator.get_batch()
                real_images, real_tags = valid_batch[0], valid_batch[1]

                # Retrieve seed for Generator
                latent_space_noise, gen_tags = noise_generator.get_batch()

                # concatenate latent space noise and tag vector to feed into generator
                gen_input = tf.concat([latent_space_noise, gen_tags], axis=1)

                # Now generate fake images
                gen_images = generator(latent_space_noise, step=step)

                # Feed real data through discriminator and retrieve output
                # forgery scores as well as label confidences
                y_real, tag_scores_real = discriminator(real_images, step=step)

                # Do the same with generated images
                y_gen, tag_scores_gen = discriminator(gen_images, step=step)

                # Calculate the losses for the generator and the discriminator
                discriminator_loss, generator_loss = dragan_loss(real_images,
                                                                 gen_images,
                                                                 y_real,
                                                                 y_gen,
                                                                 tag_scores_real,
                                                                 tag_scores_gen,
                                                                 real_tags,
                                                                 gen_tags)

                # Add to running losses for both models
                running_discriminator_loss += discriminator_loss
                running_generator_loss += generator_loss

                # Every 100th step, display statistics
                if step % 100 == 0:

                    # log running loss for discriminator
                    writer.add_scalar('validation_discriminator_loss',
                                      running_discriminator_loss / 100,
                                      epoch * len(valid_data_generator) + step)

                    writer.add_scalar('validation_generator_loss',
                                      running_generator_loss / 100,
                                      epoch * len(valid_data_generator) + step)

                    # Select a real image and a generated image to plot, as well
                    # as associated forgery scores and tag confidences
                    real_image_plt = real_images[0,:,:,:]
                    gen_image_plt = gen_images[0,:,:,:]
                    y_real_plt = y_real[0]
                    y_gen_plt = y_gen[0]
                    tag_scores_real_plt = tag_scores_real[0,:]
                    tag_scores_gen_plt = tag_scores_gen[0,:]
                    real_tags_plt = real_tags_plt[0,:]
                    gen_tags_plt = gen_tags_plt[0,:]

                    # log figure displaying discriminator's predictions on a real image
                    # as well as predictions on a generated image
                    writer.add_figure('real image vs. generated image',
                                      plot_images(real_image_plt,
                                                  gen_image_plt,
                                                  y_real_plt,
                                                  y_gen_plt,
                                                  tag_scores_real_plt,
                                                  tag_scores_gen_plt,
                                                  real_tags_plt,
                                                  gen_tags_plt,
                                                  ))

        #################################
        ######### TESTING LOOP #########
        #################################

        # Only run testing loop if testing data is provided
        if flags.test_tfrecord:
            for step in range(len(test_data_generator)):

                # Retrieve next batch of real imagery
                #test_batch = test_iterator.get_next()
                test_batch = test_data_generator.get_batch()
                real_images, real_tags = test_batch[0], test_batch[1]

                # Retrieve seed for Generator
                latent_space_noise, gen_tags = noise_generator.get_batch()

                # concatenate latent space noise and tag vector to feed into generator
                gen_input = tf.concat([latent_space_noise, gen_tags], axis=1)

                # Now generate fake images
                gen_images = generator(latent_space_noise, step=step)

                # Feed real data through discriminator and retrieve output
                # forgery scores as well as label confidences
                y_real, tag_scores_real = discriminator(real_images, step=step)

                # Do the same with generated images
                y_gen, tag_scores_gen = discriminator(gen_images, step=step)

                # Calculate the losses for the generator and the discriminator
                discriminator_loss, generator_loss = dragan_loss(real_images,
                                                                 gen_images,
                                                                 y_real,
                                                                 y_gen,
                                                                 tag_scores_real,
                                                                 tag_scores_gen,
                                                                 real_tags,
                                                                 gen_tags)

                # Add to running losses for both models
                running_discriminator_loss += discriminator_loss
                running_generator_loss += generator_loss

                # Every 100th step, display statistics
                if step % 100 == 0:

                    # log running loss for discriminator
                    writer.add_scalar('testing_discriminator_loss',
                                      running_discriminator_loss / 100,
                                      epoch * len(test_data_generator) + step)

                    writer.add_scalar('testing_generator_loss',
                                      running_generator_loss / 100,
                                      epoch * len(test_data_generator) + step)

                    # Select a real image and a generated image to plot, as well
                    # as associated forgery scores and tag confidences
                    real_image_plt = real_images[0,:,:,:]
                    gen_image_plt = gen_images[0,:,:,:]
                    y_real_plt = y_real[0]
                    y_gen_plt = y_gen[0]
                    tag_scores_real_plt = tag_scores_real[0,:]
                    tag_scores_gen_plt = tag_scores_gen[0,:]
                    real_tags_plt = real_tags[0,:]
                    gen_tags_plt = gen_tags[0,:]

                    # log figure displaying discriminator's predictions on a real image
                    # as well as predictions on a generated image
                    writer.add_figure('real image vs. generated image',
                                      plot_images(real_image_plt,
                                                  gen_image_plt,
                                                  y_real_plt,
                                                  y_gen_plt,
                                                  tag_scores_real_plt,
                                                  tag_scores_gen_plt,
                                                  real_tags_plt,
                                                  gen_tags_plt,
                                                  ))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int,
                        default=100 ,
                        help="Number of training epochs")

    parser.add_argument('--num_tags', type=int,
                        default=34,
                        help="Number of tags to assign to generated imagery")

    parser.add_argument('--learning_rate', type=float,
                        default=1e-3,
                        help="Learning rate for models"
                        )

    parser.add_argument('--decay_steps', type=int,
                        default=50000,
                        help="""Number of training iterations to complete before
                                exponential decay of learning rate for models
                             """
                        )

    parser.add_argument('--decay_rate', type=float,
                        default=0.96,
                        help="Exponential decay rate for the learning rate for models")

    parser.add_argument('--beta1', type=float,
                        default=0.5,
                        help="beta_1 variable value for adam optimizer")

    parser.add_argument('--beta2', type=float,
                        default=0.999,
                        help="beta_2 variable value for adam optimizer")

    parser.add_argument('--num_test_images', type=int,
                        default=10e3,
                        help="Number of images to be used in final testing loop")

    parser.add_argument('--num_valid_images', type=int,
                        default=10000,
                        help="Number of images to be used in full validation loop")

    parser.add_argument('--num_train_images', type=int,
                        default=50000,
                        help="Number of images to be used in full training loop")

    parser.add_argument('--logdir', type=str,
                        default='../logs',
                        help="Path to store tensorboard log files")

    parser.add_argument('--parameter_summary_update_freq', type=int,
                        default=1000,
                        help="Frequency with which to update parameters statistics in tensorboard")

    parser.add_argument('--latent_space_vector_dim', type=int,
                        default=128,
                        help="Dimension of latent space vector input to generator")

    parser.add_argument('--sqlite_database', type=str,
                        help="Path to sqlite database for dataset")

    parser.add_argument('--adversarial_balance_factor', type=float,
                        default=34,
                        help="Balance factor for adversarial loss")

    parser.add_argument('--gradient_penalty_balance_factor', type=float,
                        default=0.5,
                        help="Balance factor for gradient penalty component of loss")

    parser.add_argument('--generated_image_resolution', nargs=3,
                        type=tuple,
                        default=(128, 128, 3),
                        help="Dimensions of image we wish to generate")

    parser.add_argument('--image_width', type=int,
                        default=128,
                        help="Width of imagery to be generated")

    parser.add_argument('--image_height', type=int,
                        default=128,
                        help="Height of imagery to be generated")

    parser.add_argument('--image_channels', type=int,
                        default=3,
                        help="Num of channels in generated imagery")

    parser.add_argument('--train_batch_size', type=int,
                        default=4,
                        help="Size of batches in training loop")

    parser.add_argument('--valid_batch_size', type=int,
                        default=4,
                        help="Size of batches in validation loop")

    parser.add_argument('--test_batch_size', type=int,
                        default=4,
                        help="Size of batches in testing loop")

    parser.add_argument('--label_smoothing', action='store_true',
                        help="Whether or not to implement label smoothing in training data")

    parser.add_argument('--train_tfrecord', type=str,
                        required=True,
                        help="Path to tfrecord serializing training data")

    parser.add_argument('--valid_tfrecord', type=str,
                        required=True,
                        help="Path to tfrecord serializing validation data")

    parser.add_argument('--test_tfrecord', type=str,
                        help="Path to tfrecord serializing testing data")

    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help="Which gpus on machine to use for training")

    parser.add_argument('--noisy_labels', type=float,
                        default=0.0,
                        help="Chance of flipping 'Real' labeled imagery to 'Fake'")

    flags, _ = parser.parse_known_args()

    main(flags)


