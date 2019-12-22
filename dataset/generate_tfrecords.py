"""
Script to automatically divy dataset into train, valid, and
test sets and turn each set into a respective tfrecord
"""
import tensorflow as tf
import numpy as np
#from PIL import Image
#from opencv import cv2
import cv2
import sqlite3
import argparse
import random

import os

# So we can keep track of when we created this set
# of tfrecords
import datetime


def pull_data_from_sqlite_database(flags):
    """
    Function that connects to the sqlite database for
    collected imagery and pulls all data we would like to codify
    as tfrecords
    """
    # connect to subreddit database
    db_path = os.path.join(flags.datapath, flags.subreddit)
    metadata_db = sqlite3.connect(os.path.join(db_path, "{}.sqlite3".format(flags.subreddit)))
    cursor = metadata_db.cursor()

    # Get the number of images to encode
    cursor.execute('''SELECT COUNT(*) FROM {}_metadata'''.format(flags.subreddit))
    num_images = cursor.fetchone()[0]
    print("There are {0} images in the {1} dataset".format(num_images, flags.subreddit))

    # Get image ids
    cursor.execute('''
                   SELECT image_id FROM {}_metadata
                   '''.format(flags.subreddit))

    image_ids = cursor.fetch_all()

    # Determine data partition
    train_ids, valid_ids, test_ids = partition_data(image_ids)

    # pull train image data
    cursor.execute('''
            SELECT image_id, file_name, image_height, image_width FROM {0}_metadata
            WHERE ID IN {1}'''.format(flags.subreddit, train_ids))

    train_rows = cursor.fetch_all()

    # repeat with validation data
    cursor.execute('''
            SELECT image_id, file_name, image_height, image_width FROM {0}_metadata
            WHERE ID IN {1}'''.format(flags.subreddit, valid_ids))

    valid_rows = cursor.fetch_all()

    # and test data
    cursor.execute('''
            SELECT image_id, file_name, image_height, image_width FROM {0}_metadata
            WHERE ID IN {1}'''.format(flags.subreddit, test_ids))

    test_rows = cursor.fetch_all()

    # compile data into dictionaries so that it is easier to extract info
    # we need to go into tfrecords later on

    train_data = []
    valid_data = []
    test_data = []
    for i in range(3):

        if i == 0: all_rows = train_rows
        elif i == 1: all_rows == valid_rows
        else: all_rows == test_rows

        for index, row in enumerate(all_rows):
            image_id = row[0]
            file_name = row[1]

            # determine what type of image format the image is encoded as
            if file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                file_format = 'jpeg'
            else:
                file_format = 'png'

            #file_path = os.path.join(db_path, file_name)
            image_height = row[2]
            image_width = row[3]

            # Retrieve tags for image
            cursor.execute('''SELECT * FROM image_tags WHERE image_id=?''', (image_id))
            tags = cursor.fetchone()

            # Initialize data dictionary
            image_data = {
                'file_name': file_name,
                'format': file_format,
                'image_id': image_id,
                'height': image_height,
                'width': image_width,
                'tags': tags,
                        }

            if i == 0:
                train_data.append(image_data)
            elif i == 1:
                valid_data.append(image_data)
            else:
                test_data.append(image_data)

            if index % 100 == 0:
                print("Reading images: %d/%d" % (index, num_images))

        # Close the sqlite database
        metadata_db.close()

        return image_data

def partition_data(image_indices, flags):
    """
    Partitions data into training, validation, and testing sets
    randomly or according to some preselected parameters that
    we would like to use
    """
    # Perform random shuffle on data
    if flags.random_shuffle:
        image_indices = random.shuffle(image_indices)

    # Divy into train, valid, and test sets
    train_indices = tuple(image_indices[:flags.num_train])

    # If there are not a sufficient number of examples left to
    # fully compose valid and test sets as specified by user
    # in flag args, just divide remaining number of examples
    # in two
    num_images_remaining = len(image_indices) - flags.num_train
    if (flags.num_valid + flags.num_test) > num_images_remaining:
        num_valid = num_images_remaining // 2
        valid_indices = tuple(image_indices[flags.num_train:(flags.num_train + num_valid)])
        test_indices = tuple(image_indices[(flags.num_train + num_valid):])

    # If there are enough indices examples left to fully compose valid and
    # test sets...well...compose them
    else:
        valid_indices = tuple(image_indices[flags.num_train:(flags.num_train + flags.num_valid)])
        test_indices = tuple(image_indices[(flags.num_train + flags.num_valid):
                                     (flags.num_train + flags.num_valid + flags.num_test)])

    return train_indices, valid_indices, test_indices

def load_image(file_path):
    image = cv2.imread(file_path)
    # As cv2 loads images as BGR, convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    print("image shape: ", image.shape)
    print("image dtype: ", image.dtype)
    return image

def convert_to_tfrecords(data_dict, flags):
    """
    Conversion of single image's data in python dict format to tf.Example proto

    :param data_dict: info for one image (filename, image format, image id,
                      height, width, and tags)

    :return example: converted tf.Example
    """
    file_path = os.path.join(flags.datapath, flags.subreddit, data_dict['file_name'])
    with open(file_path, 'rb') as f:
        image_bytes = f.read()
        print('type image_bytes: ', type(image_bytes))

    return tf.train.Example(features=tf.train.Features(feature={
    'image/raw': _bytes_feature(image_bytes),
    'image/format': _bytes_feature(data_dict['format'].encode('utf-8')),
    'image/filename': _bytes_feature(data_dict['file_name'].encode('utf-8')),
    'image/id': _int64_feature(data_dict['image_id']),
    'image/height': _int64_feature(data_dict['height']),
    'image/width': _int64_feature(data_dict['width']),
    'image/tags': _int64_list_feature(data_dict['tags']),
    }))

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--subreddit', type=str,
                        default='THE_PACK',
                        help="Dataset we would like to convert into tfrecords"
                        )

    parser.add_argument('--datapath', type=str,
                        default='/mnt/Data/MachineLearningData/',
                        help="Root directory for datasets we've compiled"
                        )

    parser.add_argument('--num_train', type=int,
                        default=40000,
                        help="Number of data examples to encode in train tfrecord"
                        )


    parser.add_argument('--name_train_tfrecord', type=str,
                        default=None,
                        help="Name to use when generating training tfrecord"
                        )

    parser.add_argument('--num_valid', type=int,
                        default=10000,
                        help="Number of data examples to encode in valid tfrecord"
                        )

    parser.add_argument('--name_valid_tfrecord', type=str,
                        default=None,
                        help="Name to use when generating validation tfrecord"
                        )

    parser.add_argument('--num_test', type=int,
                        default=10000,
                        help="Number of data examples to encode in test tfrecord"
                        )

    parser.add_argument('--name_test_tfrecord', type=str,
                        default=None,
                        help="Name to use when generating test tfrecord"
                        )

    parser.add_argument('--generate_single_example', action='store_true',
                        help="Option to generate a tfrecord of a single image for testing purposes"
                        )

    flags = parser.parse_args()

    if flags.generate_single_example:

        # Use test image in THE_PACK data folder
        data_dict = {
            'format': 'jpeg',
            'file_name': 'n22b75kalq241.jpg',
            'image_id': 0,
            'height': 720,
            'width': 801,
            'tags': [0, 37],
        }

        flags.datapath = "/home/alphagoat/Projects/PACK_GAN/data/"
        flags.subreddit = "THE_PACK"

        save_path = os.path.join(flags.datapath, flags.subreddit, "single_example.tfrecords")
        with tf.io.TFRecordWriter(save_path) as tfrecord_writer:
            example = convert_to_tfrecords(data_dict, flags)
            tfrecord_writer.write(example.SerializeToString())

    else:

        # Gather data from sqlite server
        image_data = pull_data_from_sqlite_database(flags)

        # Divy data into train, valid, and test sets
        train_data, valid_data, test_data = partition_data(image_data, flags)

        # Get the path to save tfrecords (default is just to place it in the directory
        # where we saved images:
        tfrecord_save_path = os.path.join(flags.datapath, flags.subreddit)
        current_date = datetime.date.today().strftime("%d%m%y")

        # Determine the naming convention of the tfrecords
        # Training tfrecord
        if flags.name_train_tfrecord:
            train_tfrecord_path = os.path.join(tfrecord_save_path, flags.name_train_tfrecord)

        else:
            train_tfrecord_path = os.path.join(tfrecord_save_path,
                             '{0}_train{1}.tfrecords'.format(flags.subreddit, current_date))

        # Validation tfrecord
        if flags.name_valid_tfrecord:
            valid_tfrecord_path = os.path.join(tfrecord_save_path, flags.name_valid_tfrecord)

        else:
            valid_tfrecord_path = os.path.join(tfrecord_save_path,
                             '{0}_valid{1}.tfrecords'.format(flags.subreddit, current_date))

        # Testing tfrecord
        if flags.name_test_tfrecord:
            test_tfrecord_path = os.path.join(tfrecord_save_path, flags.name_test_tfrecord)

        else:
            test_tfrecord_path = os.path.join(tfrecord_save_path,
                             '{0}_test{1}.tfrecords'.format(flags.subreddit, current_date))

        # Convert data into tfrecords
        # first: training set
        with tf.python_io.TFRecordWriter(train_tfrecord_path) as tfrecord_writer:
            for index, image_data in enumerate(train_data):
                if index % 100 == 0:
                    print("Converting images: %d/%d" % (index, flags.num_train))

                example = convert_to_tfrecords(image_data, flags)
                tfrecord_writer.write(example.SerializeToString())

        # Now the validation set
        with tf.python_io.TFRecordWriter(valid_tfrecord_path) as tfrecord_writer:
            num_valid = len(valid_data)
            for index, image_data in enumerate(valid_data):
                if index % 100 == 0:
                    print("Converting images: %d/%d" % (index, num_valid))

                example = convert_to_tfrecords(image_data, flags)
                tfrecord_writer.write(example.SerializeToString())

        # Finally, the test set
        with tf.python_io.TFRecordWriter(test_tfrecord_path) as tfrecord_writer:
            num_test = len(test_data)
            for index, image_data in enumerate(test_data):
                if index % 100 == 0:
                    print("Converting images: %d/%d" % (index, num_test))

                example = convert_to_tfrecords(image_data, flags)
                tfrecord_writer.write(example.SerializeToString())
