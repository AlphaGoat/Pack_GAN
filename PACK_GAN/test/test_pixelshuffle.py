"""
Script to specifically test custom pixel shuffle function
Works by generating 4 x tensors colored red, blue, green,
and cyan, then passes through pixel shuffle function,
displaying results on TensorBoard

Succesful test: 25 Dec 2019 (Merry Christmas!)
Peter J. Thomas
"""
import tensorflow as tf

import argparse
import sys

from models.generator import Generator

def main(flags):

    batch_size = flags.batch_size

    # Generate four tensors of different pixel shading and test
    # the function's ability to shuffle them
    t_ones = tf.ones((batch_size, 16, 16), dtype=tf.float32)

    t_red = tf.stack([255 * t_ones, t_ones, t_ones], axis=3)
    t_green = tf.stack([t_ones, 255 * t_ones, t_ones], axis=3)
    t_blue = tf.stack([t_ones, t_ones, 255 * t_ones], axis=3)
    t_cyan = tf.stack([t_ones, 255 * t_ones, 255 * t_ones], axis=3)

    # Initialize generic Generator to test pixel shuffle method
    test_generator = Generator(128, 128, 3, 128, 1)

    # Stack all input tensors together
    t_colormap = tf.stack([t_red, t_green, t_blue, t_cyan], axis=3)

    # Tile to increase the number of channel (4th dim). We won't touch the 5th dim
    # to preserve color scheme
    if flags.num_tile:
        num_tile = flags.num_tile
        t_colormap = tf.tile(t_colormap, [1, 1, 1, num_tile, 1])

    tf.print("(pjt) t_colormap: ", tf.shape(t_colormap))

    # Pass tensors through pixel shuffle function
    shuffled_pixel_map = test_generator.pixel_shuffle_x2_layer_no_while_loop(t_colormap)
    tf.print("(pjt) shuffled_pixel_map: ", tf.shape(shuffled_pixel_map))

    # Initialize tf writer for tensorboard
    writer = tf.summary.create_file_writer(flags.logdir)

    # Add image summaries for each of these tensors
    with writer.as_default():

        tf.summary.image("red_tensor", tf.cast(t_red, dtype=tf.uint8), step=0)
        tf.summary.image("green_tensor", tf.cast(t_green, dtype=tf.uint8), step=0)
        tf.summary.image("blue_tensor", tf.cast(t_blue, dtype=tf.uint8), step=0)
        tf.summary.image("cyan_tensor", tf.cast(t_cyan, dtype=tf.uint8), step=0)
        tf.summary.image("shuffled_pixel_map", tf.cast(shuffled_pixel_map, dtype=tf.uint8), step=0)

        #writer.flush()
        writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int,
                        default=1,
                        help="batch_size that we would like to test this function with"
                        )

    parser.add_argument('--logdir', type=str,
                        default='../logs',
                        help="Directory where we keep tensorboard logs for this project"
                        )

    parser.add_argument('--num_tile', type=int,
                        default=None,
                        help="""
                             Number of times to tile the input to the
                             pixel shuffle function (test the function's
                             ability to handle multi-channel inputs
                             """
                        )

    flags, _ = parser.parse_known_args()

    main(flags)

