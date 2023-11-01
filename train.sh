#!/usr/bin/env bash

# local machine kitti360panorama
# python train.py \
#     --config_path=/workspace/Patch-NetVLAD/patchnetvlad/configs/train.ini \
#     --cache_path=/workspace/Patch-NetVLAD/log/cache \
#     --save_path=/workspace/Patch-NetVLAD/log/checkpoint \
#     --dataset_root_dir=/lyh/KITTI360 \
#     --identifier="KITTI360_IMAGE" \
#     --threads=1 \
#     --cluster_path=/workspace/Patch-NetVLAD/log/cache/centroids/vgg16_KITTI360_16_desc_cen.hdf5

# local machine msls
python train.py \
    --config_path=/workspace/Patch-NetVLAD/patchnetvlad/configs/train.ini \
    --cache_path=/workspace/Patch-NetVLAD/log/cache \
    --save_path=/workspace/Patch-NetVLAD/log/checkpoint \
    --dataset_root_dir=/lyh/MSLS \
    --dataset='msls' \
    --identifier="mapillary_nopanos_smd" \
    --threads=8
    # --cluster_path=/workspace/Patch-NetVLAD/log/cache/centroids/vgg16_KITTI360_16_desc_cen.hdf5

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