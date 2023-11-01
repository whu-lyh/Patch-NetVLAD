
import json
import os
import re
from os.path import join

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image


default_cities = {
    # 'train': ["0", "2", "4", "5", "6", "7", "9", "10"],
    'train': ["0"],
    'val': ["3"],
    'test': ["3"]
}


def arange_as_seq(data, path_img, seq_length):
    seq_keys, seq_idxs = [], []
    for idx in data.index:
        # find surrounding frames in sequence
        seq_idx = idx
        seq = data.iloc[seq_idx]
        img_num = int(re.sub('[a-z]', '', seq['key']))
        seq_key = join(path_img, 'pano', 'data_rgb', '%010d' % img_num + '.png')           
        seq_keys.append(seq_key)
        seq_idxs.append([seq_idx])         
    return seq_keys, np.asarray(seq_idxs)


if __name__ == "__main__":
    
    root_dir = "/lyh/KITTI360"
    subdir_img = 'data_2d_pano'
    cities = default_cities["train"]
    
    for city in cities:
        city='2013_05_28_drive_%04d_sync' % int(city)
        print(join(root_dir, subdir_img, city, 'query.csv'))
        print(join(root_dir, subdir_img, city, 'database.csv'))
        # load query data
        qData = pd.read_csv(join(root_dir, subdir_img, city, 'query.csv'), index_col=0)
        # load database data
        dbData = pd.read_csv(join(root_dir, subdir_img, city, 'database.csv'), index_col=0)
        # arange based on task
        qSeqKeys, qSeqIdxs = arange_as_seq(qData, join(root_dir, subdir_img, city), 1)
        print(len(qSeqKeys))
        # print(qSeqKeys)
        print(len(qSeqIdxs))
        # print(qSeqIdxs)
        for image in qSeqKeys:
            Image.open(image).convert('RGB')
        dbSeqKeys, dbSeqIdxs = arange_as_seq(dbData, join(root_dir, subdir_img, city), 1)
        print(len(dbSeqKeys))
        # print(dbSeqKeys)
        print(len(dbSeqIdxs))
        # print(dbSeqIdxs)
        for image in qSeqKeys:
            Image.open(image).convert('RGB')
        