#!/usr/bin/env bash
_now=$(date +"%d%b%Y_%H%M")
_file="/home/alphagoat/Projects/PACK_GAN/logs/run_$_now.txt"
nohup python test_gan.py \
    --num_train_images=1000 \
    --train_batch_size=1 \
    --train_tfrecord="../data/THE_PACK/single_example.tfrecords" \
    --valid_tfrecord="../data/THE_PACK/single_example.tfrecords" \
    --generated_image_width=128 \
    --generated_image_height=128 \
    --generated_image_channels=3 \
    --label_smoothing \
    --noisy_labels=0.05 \
    --num_tags=4 \
    &>  $_file
