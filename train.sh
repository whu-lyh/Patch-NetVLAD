#!/usr/bin/env bash
# local machine
python train.py \
--config_path patchnetvlad/configs/train.ini \
--cache_path=/data-lyh2/KITTI360/patch-netvlad_tmp/cache \
--save_path=/data-lyh2/KITTI360/patch-netvlad_tmp/checkpoint \
--dataset_root_dir=/data-lyh \
--cluster_path=/data-lyh2/KITTI360/patch-netvlad_tmp/cache/centroids/vgg16_mapillary_16_desc_cen.hdf5

# remote server kitti360panorama
# python train.py \
# --config_path patchnetvlad/configs/train.ini \
# --cache_path=/root/lyh/data_lyh/kitti360/cache \
# --save_path=/root/lyh/data_lyh/kitti360/checkpoint \
# --dataset_root_dir=/root/public/data/Kitti/kitti360 \
# --cluster_path=/root/lyh/data_lyh/kitti360/cache/centroids/vgg16_mapillary_16_desc_cen.hdf5

# # remote server msls
# python train.py \
# --config_path patchnetvlad/configs/train.ini \
# --cache_path=/root/lyh/data_lyh/kitti360/cache \
# --save_path=/root/lyh/data_lyh/kitti360/checkpoint \
# --dataset_root_dir=/root/public/data/MSLS_Data/ \
# --cluster_path=/root/lyh/data_lyh/kitti360/cache/centroids/vgg16_mapillary_16_desc_cen.hdf5