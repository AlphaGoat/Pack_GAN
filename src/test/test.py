"""
Test script to run full DRAGAN architecture

31 Dec 2019

Peter J. Thomas
"""
import tensorflow as tf
from tensorflow.keras import Model

import argparse
import os
#import sqlite3
#import tqdm

#from loss.discriminator_loss import DiscriminatorLoss
#from loss.generator_loss import GeneratorLoss
from loss.loss import DRAGANLoss
#from models.dragan.generator import build_generator
#from models.dragan.discriminator import build_discriminator
from models.dragan.discriminator_keras import initialize_discriminator
from models.dragan.generator_keras import initialize_generator
from dataset.real_dataset_generator import DatasetGenerator
from dataset.noise_generator import NoiseGenerator
from utils.tensorboard_plotting import plot_images

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
    image_width = flags.image_pixel_width
    image_height = flags.image_pixel_height
    num_channels = flags.image_channels

    train_data_generator = DatasetGenerator(train_tfrecord_name,
                                            num_train_images,
                                            num_channels,
                                            flags.num_tags,
                                            batch_size=flags.train_batch_size,
                                            label_smoothing=flags.label_smoothing,
                                            num_threads=1,
                                            buffer_size=30,
                                            return_filename=False,
                                            image_reshape=True)

#    train_iterator = train_data_generator.get_iterator()

    valid_data_generator = DatasetGenerator(valid_tfrecord_name,
                                            num_valid_images,
                                            num_channels,
                                            flags.num_tags,
                                            batch_size=flags.valid_batch_size,
                                            num_threads=1,
                                            buffer_size=30,
                                            return_filename=False,
                                            image_reshape=True)

#    valid_iterator = valid_data_generator.get_iterator()

    # Initialize testing data generator only if testing flag has been set
    if flags.test_tfrecord:
        test_data_generator = DatasetGenerator(test_tfrecord_name,
                                               num_test_images,
                                               num_channels,
                                               flags.num_tags,
                                               batch_size=flags.test_batch_size,
                                               num_threads=1,
                                               buffer_size=30,
                                               return_filename=False,
                                               image_reshape=True)

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

    # Instantiate Loss function
#    discriminatorLoss = DiscriminatorLoss()
#    generatorLoss = GeneratorLoss()
    loss = DRAGANLoss()

    # Instantiate generator and discriminator models
#    optimizer = tf.keras.optimizers.Adan(learning_rate=flags.learning_rate,
#                                         beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=flags.learning_rate,
                                         beta_1=0.5)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=flags.learning_rate,
                                             beta_1=0.5)

    gen_img_input = tf.keras.Input(shape=(image_height,))# image_width, num_channels))
    gen_tag_input = tf.keras.Input(shape=(flags.num_tags,))
    gen_output = initialize_generator(gen_img_input, gen_tag_input)
    Generator = Model(inputs=[gen_img_input, gen_tag_input], outputs=gen_output, name="Generator")
    Generator.summary()

#    Generator = initialize_generator(flags.input_shape, flags.num_tags)
#    Generator.compile(optimizer=optimizer)
#    Generator.summary()

    disc_input = tf.keras.Input(shape=(flags.input_shape))
    forgery_score, tag_scores = initialize_discriminator(disc_input, flags.num_tags)
    Discriminator = Model(disc_input, [forgery_score, tag_scores],
                                   name="Discriminator")

#    Discriminator = initialize_discriminator(flags.input_shape)
#    Discriminator.compile(optimizer=optimizer)

    # Construct GAN
#    gan_input = Input(shape=(flags.input_shape,))
#    x = Generator(x)
#    gan_output = Discriminator(x)
#    GAN = Model(inputs=gan_input, outputs=gan_output)
#    GAN.compile(loss=loss, optimizer=optimizer)
#    GAN.summary()

    for e in range(flags.epochs):

        for step in range(flags.num_train_images / flags.batch_size):
            # Get latent space vector noise
            noise = noise_generator.get_batch()

            # Generate fake imagery from noised input
            generated_images = Generator.predict(noise)

            #Get a random set of real imagery
            image_batch = train_data_generator.get_batch()

            with tf.GradientTape() as tape:

                # Feed real and generated data through discriminator
                # and retrieve the discriminator's prediction on
                # whether the data is a forgery and tags
                y_real, tag_scores_real = Discriminator.train_on_batch(image_batch)

                y_gen, tag_scores_gen = Discriminator.train_on_batch(generated_images)

                # Calculate loss for generator and discriminator
                discriminator_loss, generator_loss = loss(image_batch,
                                                          generated_images,
                                                          y_real,
                                                          y_gen,
                                                          tag_scores_real,
                                                          tag_scores_gen,
                                                          )

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss
            disc_grads = tape.gradient(discriminator_loss, Discriminator.trainable_weights)
            gen_grads = tape.gradient(generator_loss, Generator.trainable_weights)

            # Run one step of gradient descent by updating the value of the bariables to
            # minimize loss
            disc_optimizer.apply_gradients(zip(disc_grads, Discriminator.trainable_weights))
            gen_optimizer.apply_gradients(zip(gen_grads, Generator.trainable_weights))

            # Log every 200 batches
            if step % 200 == 0:
                print(
                    "Training Discriminator loss (for one batch) at step %d: %.4f"
                    % (step, float(discriminator_loss)))
                print(
                    "Training Generator loss (for one batch) at step %d: %.4f"
                    % (step, float(generator_loss)))
                print("Seen so far: %s samples" % ((step + 1) * 64))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_list", type=str,
                        default="0",
                        help="which gpus on machine to use for training"
                        )

    parser.add_argument('--num_epochs', type=int,
                        default=100,
                        help="Number of training epochs"
                        )

    parser.add_argument('--num_tags', type=int,
                        default=34,
                        help="Number of tags to assign to generated imagery")

    parser.add_argument('--learning_rate', type=float,
                        default=2e-4,
                        help="Learning rate for models"
                        )

    parser.add_argument('--num_train_images', type=int,
                        default=50000,
                        help="Number of images to be used in full training loop (one epoch)"
                        )

    parser.add_argument('--num_valid_images', type=int,
                        default=20000,
                        help="Number of images to be used in full validation loop"
                        )

    parser.add_argument('--num_test_images', type=int,
                        default=10000,
                        help="Number of images to be used in full testing loop"
                        )

    parser.add_argument('--train_batch_size', type=int,
                        default=16,
                        help="Size of training batch"
                        )

    parser.add_argument('--valid_batch_size', type=int,
                        default=16,
                        help="Size of validation batch"
                        )

    parser.add_argument('--test_batch_size', type=int,
                        default=16,
                        help="size of test batch"
                        )

    parser.add_argument('--train_tfrecord', type=str,
                        default="/home/alphagoat/Projects/PACK_GAN/data/THE_PACK/single_example.tfrecords",
                        help="path for tfrecord containing serialized training data"
                        )

    parser.add_argument('--valid_tfrecord', type=str,
                        default="/home/alphagoat/Projects/PACK_GAN/data/THE_PACK/single_example.tfrecords",
                        help="path for tfrecord containing serialized validation data"
                        )

    parser.add_argument('--test_tfrecord', type=str,
                        default="/home/alphagoat/Projects/PACK_GAN/data/THE_PACK/single_example.tfrecords",
                        help="path for tfrecord containing serialized testing data"
                        )

    parser.add_argument('--image_pixel_width', type=int,
                        default=128,
                        help="pixel width of input imagery"
                        )

    parser.add_argument('--image_pixel_height', type=int,
                        default=128,
                        help="pixel height of input imagery"
                        )

    parser.add_argument('--image_channels', type=int,
                        default=3,
                        help="number of image channels (3 for color imagery, 4 with alpha"
                        )

    parser.add_argument('--label_smoothing', action='store_true',
                        help="""Whether or not to apply label smoothing to training data
                        i.e., set "true" to 0.9 and "false" to 0.1)"""
                        )

    parser.add_argument('--latent_space_vector_dim', type=int,
                        default=128,
                        help="Dimensions of latent space vector for noise input to GAN"
                        )

    flags = parser.parse_args()
    main(flags)
