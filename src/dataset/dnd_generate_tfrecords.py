import tensorflow as tf
import numpy as np
#import pandas as pd
import cv2

import argparse
import json
import os
# import datetime

fields = [
    "filename",
    "pix_height",
    "pix_width",
    "num_channels",
    "file_format",
    # Tags for each image, starting with
    # race of character present
    "human_flag",
    "dwarf_flag",
    "elf_flag",
    "halfling_flag",
    "dragonborn_flag",
    "gnome_flag",
    "halforc_flag",
    "tiefling_flag",
    "aasimar_flag",
    # Tags for classes
    "barbarian_flag",
    "bard_flag",
    "cleric_flag",
    "druid_flag",
    "fighter_flag",
    "monk_flag",
    "paladin_flag",
    "ranger_flag",
    "rogue_flag",
    "sorcerer_flag",
    "warlock_flag",
    "wizard_flag",
    # Tags for gender
    "male_flag",
    "female_flag"
]

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    # BytesList won't unpack values from an eager tensor by default,
    # so grab numpy values
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(filepath):
    """
    Load image in dataset into numpy array
    """
    image = cv2.imread(filepath)

    # As cv2 loads images as BGR, convert to RGB
    image = cv2.cv2Color(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return image


def convert_to_tfrecords(filename, image_data, flags):
    """
    Convert single image's data in python dict format to tf.Example proto

    :param image_data: dictionary with following entries:
                        file_format (jpg/png)
                        pix_height
                        pix_width
                        num_channels,
                        tags (list of flags)

    :param flags:     command line arguments
    """

    # unpack fields
    tags = []
    for _, tag_flag in image_data['tags'].items():
        tags.append(tag_flag)


    filepath = os.path.join(flags.datafolder, filename)
    with open(filepath, 'rb') as f:
        image_bytes = f.read()

    return tf.train.Example(features=tf.train.Features(feature={
        'image/raw': _bytes_feature(image_bytes),
        'image/format': _bytes_feature(image_data['file_format'].encode('utf-8')),
        'image/filename': _bytes_feature(filename.encode('utf-8')),
        'image/height': _int64_feature(image_data['img_height']),
        'image/width': _int64_feature(image_data['img_width']),
        'image/channels': _int64_feature(image_data['num_channels']),
        'image/tags': _int64_list_feature(tags),
    }))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--datafolder', type=str,
                        default="/media/alphagoat/Backup Plus/MachineLearningData/BooruCharacterPortraits",
                        help="Path to folder containing dataset"
                        )

    parser.add_argument('--generate_single_example', action='store_true',
                        help="Generate a tfrecord of a single image for testing"
                        )

    flags = parser.parse_args()

    if flags.generate_single_example:

        filename = "thumbnail_2fdcc240a6056f29e70cb1f52c4c0c116d42c286.jpg"
        filepath = os.path.join(flags.datafolder,
                                filename)

#        df = pd.read_csv(os.path.join(flags.datafolder, "metadata.csv"),
#                         index_col='filename',
#                         header=0,
#                         names=fields
#                         )
#
#        # Retrieve fields of relevance to file
#        example_fields = df[filename]

        # Read metadata and labels for image from json
        with open(os.path.join(flags.datafolder, "metadata.json")) as json_file:

            data = json.load(json_file)
            image_data = data[filename]

        savepath = os.path.join(flags.datafolder, "dnd_single_example.tfrecords")
        with tf.io.TFRecordWriter(savepath) as tfrecord_writer:
            example = convert_to_tfrecords(filename, image_data, flags)
            tfrecord_writer.write(example.SerializeToString())
