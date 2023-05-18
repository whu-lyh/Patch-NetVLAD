#!/usr/bin/env bash

python train.py \
--config_path patchnetvlad/configs/train.ini \
--cache_path=/data-lyh2/KITTI360/patch-netvlad_tmp/cache \
--save_path=/data-lyh2/KITTI360/patch-netvlad_tmp/checkpoint \
--dataset_root_dir=/data-lyh \
--cluster_path=/data-lyh2/KITTI360/patch-netvlad_tmp/cache/centroids/vgg16_mapillary_16_desc_cen.hdf5