"""
Test script for debugging Generator architecture.
NOTE: this only provides utility to test the generator's
forward pass. There is no functionality to test updating
parameters during training. In effect, we are only able to
evaluate the generator's ability to produce an output from
random noise

Date of debugging initiation: 22 Dec 2019
Date of first succesful forward pass: TBD

Peter J. Thomas
"""
import tensorflow as tf

import os
import argparse

from models.generator import Generator

def main(flags):

    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Initialize generator model to use
    # fetch initialize variables
    batch_size = flags.batch_size
    latent_space_vector_dim = flags.latent_space_vector_dim
    num_tags = flags.num_tags
    output_width = flags.image_width
    output_height = flags.image_height
    output_channels = flags.image_channels

    generator = Generator(output_width,
                          output_height,
                          output_channels,
                          latent_space_vector_dim,
                          num_tags
                          )


    # Initialize summary_writer
    writer = tf.summary.create_file_writer(flags.logdir)

    # initialize variables needed for test loop
    num_test_loops = flags.num_test_loops

    with writer.as_default():

        # Initiate testing loop
        for step in range(num_test_loops):

            noise = tf.random.uniform((batch_size, latent_space_vector_dim + num_tags))
            generated_tags = noise[:,:-num_tags]

            print("in the main test loop, we are now on step: ", step)
            generated_tensor = generator.forward_pass(noise, step)

            # Display every 10th noise input and generated tensor on tensorboard
            if step % 10 == 0:
                #tf.summary.image("noise_input", tf.cast(noise, dtype=tf.uint8), step=step)

                # Scale output tensor to be an 8-bit image
                scaled_gen_tensor = 255 * generated_tensor
                tf.summary.image("generated_tensor", tf.cast(generated_tensor, dtype=tf.uint8), step=step)

        writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int,
                        default=4,
                        help="Number of examples per batch"
                        )

    parser.add_argument('--latent_space_vector_dim', type=int,
                        default=128,
                        help="Dimension of latent space noise vector to use feed as input to generator"
                        )

    parser.add_argument('--num_tags', type=int,
                        default=34,
                        help="""
                             Dimension of tensor to generate randomly assigning tags to noise vector we
                             are using to define features of Generator output
                             """
                        )

    parser.add_argument('--image_width', type=int,
                        default=128,
                        help="Width of image tensor that we would like to create with Generator"
                        )

    parser.add_argument('--image_height', type=int,
                        default=128,
                        help="Height of image tensor that we would like to create with Generator"
                        )

    parser.add_argument('--image_channels', type=int,
                        default=3,
                        help="""
                        Number of channels of image tensor that we would like to create with Generator.
                        (Default is 3 for RGB)
                        """
                        )

    parser.add_argument('--num_test_loops', type=int,
                        default=5,
                        help="""
                             Number of testing loops to run (NOTE: does not need to be a large number.
                             We are not actually updating Generator parameters during these loops, so we
                             are only really testing the generator's ability to generate noise
                             """
                        )

    parser.add_argument('--logdir', type=str,
                        default="../logs/",
                        help="Directory to generate TensorBoard logs while model is running"
                        )

    parser.add_argument('--gpu_list', type=str,
                        default='0'
                        )

    flags, _ = parser.parse_known_args()

    main(flags)





