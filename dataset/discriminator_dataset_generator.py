"""
Dataset generator to build pipeline to feed data
into deep learning models

Peter Thomas
15 December 2019

Based on(and basically outright from) Mcquaid's generator
github handle: ianwmcquaid
"""

import tensorflow as tf
import numpy as np

class DatasetGenerator(object):

    def __init__(self,
                 tfrecord_name,
                 num_images,
                 num_channels,
                 batch_size=4,
                 num_threads=1,
                 buffer=30,
                 label_smoothing=False,
                 encoding_function=None,
                 return_filename=False):
        """
        :param tfrecord_name: file path of tfrecord we will be reading
                              from
        :param num_images: number of images to compose dataset generator
                           pipeline out of
        :param num_channels: number of image channels (typically 3 for 'rgb')
        :param batch_size: size of image batches that will be composed in
                           generator
        :param num_threads: number of threads to spool generator out of
        :param buffer:  Prefetch buffer size to use in processing
        :param encoding_function: Custom encoding function to map to image pixel data
                                  (dependent on the recipient network, e.g. may need
                                  to resize images to a consistent height x width)
        """
        self.tfrecord_name = tfrecord_name
        self.num_images = num_images
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.buffer = buffer
        self.label_smoothing = label_smoothing
        self.encoding_function = encoding_function
        self.return_filename = return_filename
        self.dataset = self.build_pipeline(tfrecord_name,
                                           batch_size,
                                           self.num_threads,
                                           self.buffer)

    def build_pipeline(self,
                       tfrecord_path,
                       batch_size,
                       num_threads,
                       buffer,
                       augment=False,
                       cache_dataset_memory=False,
                       cache_dataset_file=False,
                       cache_name=None
                       ):
        """
        Reads in data from a TFRecord file, applies augmentation chain (if
        desired), shuffles and batches data.
        Supports prefetching and multithreading, the intent being to pipeline
        the training process to lower latency
        """
        # Create TFRecord dataset
        data = tf.data.TFRecordDataset(tfrecord_path)

        # parse record into tensors
        data = data.map(self._parse_data)

        # TODO: define augmentations that we would like
        #       to apply to input imagery
        # if augment: write some augmentation code here

        # label smoothing: if selected, convert '1.0' labels to '0.9'
        if self.label_smoothing:
            data = data.map(lambda image, tag: (image, 0.9 * tag))

        # If the destination network requires a special encoding (or
        # we would like to apply our own to try to lower the complexity
        # of the generation problem, do that here
        if self.encoding_function is not None:
            data = data.map(self.encoding_function, num_parallel_calls=num_threads)

        if cache_dataset_memory:
            data = data.cache()
        elif cache_dataset_file:
            data = data.cache("./" + cache_name + "CACHE")

        data = data.repeat()

        # batch the data
        data = data.batch(batch_size)

        # prefetch with multiple threads
        data.prefetch(buffer_size=buffer)

        # TODO: finish building pipeline

        # return a reference to this data pipeline
        return data

    def _parse_data(self, example_proto):

        # Define how to parse the example
        features = {
            'image/encoded': tf.VarLenFeature(dtype=tf.string),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64),
            'image/tags': tf.FixedLenFeature([], dtype=tf.int64),
            'image/filename': tf.VarLenFeature([], dtype=tf.string),
            'image/format': tf.VarLenFeature([], dtype=tf.string)
        }

        # Parse the example
        features_parsed = tf.parse_single_example(serialized=example_proto,
                                                  feature=features)
        width = tf.cast(features_parsed['image/width'], tf.float32)
        height = tf.cast(features_parsed['image/height'], tf.float32)

        filename = tf.cast(tf.sparse_tensor_to_dense(features_parsed['image/filename'], default_value=""),
                           tf.string)
        image_format = tf.cast(tf.sparse_tensor_to_dense(features_parsed['image/format'], default_value=""),
                               tf.string)
        tags = tf.cast(tf.sparse_tensor_to_dense(features_parsed['image/tags']), tf.float32)

        # Decode imagery from raw bytes
        images = tf.sparse_tensor_to_dense(features_parsed['image/raw'], default_value="")
        images = tf.decode_raw(images, tf.uint16)
        images = tf.reshape(images, [height, width, self.num_channels])

        # Normalize the images pixels to zero mean and unit variance
        images = tf.image.per_image_standardization(images)

        if self.return_filename:
            return images, tags, filename
        else:
            return images, tags

    def get_iterator(self):
        # Create and return iterator
        print("Got here")
        return self.dataset.make_one_shot_iterator()









