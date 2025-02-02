'''
Copyright (c) Facebook, Inc. and its affiliates.

MIT License

Copyright (c) 2020 mapillary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified by Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

'''


import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.utils.data as data
import pandas as pd
from os.path import join
from sklearn.neighbors import NearestNeighbors
import cv2
import math
import torch
import random
import sys
import itertools
from tqdm import tqdm


default_cities = {
    'train': ["0", "2", "4", "5", "6", "7", "9", "10"],
    'val': ["3"],
    'test': ["3"]
}

def downsample_gaussian_blur(img,ratio):
    sigma=(1/ratio)/3
    # ksize=np.ceil(2*sigma)
    ksize=int(np.ceil(((sigma-0.8)/0.3+1)*2+1))
    ksize=ksize+1 if ksize%2==0 else ksize
    img=cv2.GaussianBlur(img,(ksize,ksize),sigma,borderType=cv2.BORDER_REFLECT101)
    return img

def resize_img(img_in, ratio):
    # if ratio>=1.0: return img
    img = cv2.cvtColor(np.array(img_in),cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape
    hn, wn = int(np.round(h * ratio)), int(np.round(w * ratio))
    img_out = cv2.resize(downsample_gaussian_blur(img, ratio), (wn, hn), cv2.INTER_LINEAR)
    img_out = Image.fromarray(cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB))
    return img_out


class ImagesFromList(Dataset):
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = resize_img(img, 0.25)
        img = self.transform(img)

        return img, idx


class KITTI360PANORAMA(Dataset):
    def __init__(self, root_dir, save=False, cities='', nNeg=5, transform=None, mode='train', task='im2im', 
                 seq_length=1, posDistThr=10, negDistThr=25, cached_queries=1000, cached_negatives=2000, bs=24, threads=8, margin=0.1):

        # initializing
        assert mode in ('train', 'val', 'test')
        assert task in ('im2im', 'im2seq', 'seq2im', 'seq2seq')
        assert seq_length % 2 == 1
        assert (task == 'im2im' and seq_length == 1) or (task != 'im2im' and seq_length > 1)

        if cities in default_cities:
            self.cities = default_cities[cities]
        elif cities == '':
            self.cities = default_cities[mode]
        else:
            self.cities = cities.split(',')

        self.qIdx = []
        self.qImages = []
        self.pIdx = []
        self.nonNegIdx = []
        self.dbImages = []
        self.qEndPosList = []
        self.dbEndPosList = []

        self.all_pos_indices = []

        # hyper-parameters
        self.nNeg = nNeg
        self.margin = margin
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.cached_queries = cached_queries
        self.cached_negatives = cached_negatives

        # flags
        self.cache = None
        self.mode = mode

        # other
        self.transform = transform

        # define sequence length based on task
        if task == 'im2im':
            seq_length_q, seq_length_db = 1, 1

        # load data
        for city in self.cities:
            city='2013_05_28_drive_%04d_sync' % int(city)
            print("=====> {}".format(city))
            subdir_img = 'data_2d_pano'
            # get len of images from cities so far for indexing
            _lenQ = len(self.qImages)
            _lenDb = len(self.dbImages)

            # when GPS / UTM is available
            if self.mode in ['train', 'val']:
                # load query data
                qData = pd.read_csv(join(root_dir, subdir_img, city, 'query.csv'), index_col=0)

                # load database data
                dbData = pd.read_csv(join(root_dir, subdir_img, city, 'database.csv'), index_col=0)

                # arange based on task
                qSeqKeys, qSeqIdxs = self.arange_as_seq(qData, join(root_dir, subdir_img, city), seq_length_q)
                dbSeqKeys, dbSeqIdxs = self.arange_as_seq(dbData, join(root_dir, subdir_img, city), seq_length_db)

                unique_qSeqIdx = np.unique(qSeqIdxs)
                unique_dbSeqIdx = np.unique(dbSeqIdxs)

                # if a combination of city, task and subtask is chosen, where there are no query/dabase images,
                # then continue to next city
                if len(unique_qSeqIdx) == 0 or len(unique_dbSeqIdx) == 0:
                    continue

                self.qImages.extend(qSeqKeys)
                self.dbImages.extend(dbSeqKeys)

                self.qEndPosList.append(len(qSeqKeys))
                self.dbEndPosList.append(len(dbSeqKeys))

                qData = qData.loc[unique_qSeqIdx]
                dbData = dbData.loc[unique_dbSeqIdx]

                # useful indexing functions
                seqIdx2frameIdx = lambda seqIdx, seqIdxs: seqIdxs[seqIdx]
                # frameIdx2seqIdx = lambda frameIdx, seqIdxs: np.where(seqIdxs == frameIdx)[0][1]
                frameIdx2uniqFrameIdx = lambda frameIdx, uniqFrameIdx: np.where(np.in1d(uniqFrameIdx, frameIdx))[0]
                uniqFrameIdx2seqIdx = lambda frameIdxs, seqIdxs: \
                    np.where(np.in1d(seqIdxs, frameIdxs).reshape(seqIdxs.shape))[0]

                # utm coordinates
                utmQ = qData[['east', 'north']].values.reshape(-1, 2)
                utmDb = dbData[['east', 'north']].values.reshape(-1, 2)

                # find positive images for training
                neigh = NearestNeighbors(algorithm='brute')
                neigh.fit(utmDb)
                pos_distances, pos_indices = neigh.radius_neighbors(utmQ, self.posDistThr)
                # print("len(pos_indices): ", len(pos_indices))
                self.all_pos_indices.extend(pos_indices)

                if self.mode == 'train':
                    nD, nI = neigh.radius_neighbors(utmQ, self.negDistThr)

                for q_seq_idx in range(len(qSeqKeys)):

                    q_frame_idxs = seqIdx2frameIdx(q_seq_idx, qSeqIdxs)
                    q_uniq_frame_idx = frameIdx2uniqFrameIdx(q_frame_idxs, unique_qSeqIdx)

                    p_uniq_frame_idxs = np.unique([p for pos in pos_indices[q_uniq_frame_idx] for p in pos])

                    # the query image has at least one positive
                    if len(p_uniq_frame_idxs) > 0:
                        p_seq_idx = np.unique(uniqFrameIdx2seqIdx(unique_dbSeqIdx[p_uniq_frame_idxs], dbSeqIdxs))

                        self.pIdx.append(p_seq_idx + _lenDb)
                        self.qIdx.append(q_seq_idx + _lenQ)

                        # in training we have two thresholds, one for finding positives and one for finding images
                        # that we are certain are negatives.
                        if self.mode == 'train':

                            n_uniq_frame_idxs = np.unique([n for nonNeg in nI[q_uniq_frame_idx] for n in nonNeg])
                            n_seq_idx = np.unique(uniqFrameIdx2seqIdx(unique_dbSeqIdx[n_uniq_frame_idxs], dbSeqIdxs))

                            self.nonNegIdx.append(n_seq_idx + _lenDb)

            # when GPS / UTM / pano info is not available
            elif self.mode in ['test']:
                # load images
                qIdx = pd.read_csv(join(root_dir, subdir_img, city, 'query.csv'), index_col=0)
                dbIdx = pd.read_csv(join(root_dir, subdir_img, city, 'database.csv'), index_col=0)

                # arange in sequences
                qSeqKeys, qSeqIdxs = self.arange_as_seq(qIdx, join(root_dir, subdir_img, city), seq_length_q)
                dbSeqKeys, dbSeqIdxs = self.arange_as_seq(dbIdx, join(root_dir, subdir_img, city),seq_length_db)

                self.qImages.extend(qSeqKeys)
                self.dbImages.extend(dbSeqKeys)

                # add query index
                self.qIdx.extend(list(range(_lenQ, len(qSeqKeys) + _lenQ)))

                # if a combination of cities, task and subtask is chosen, where there are no query/database images,
                # then exit
        if len(self.qImages) == 0 or len(self.dbImages) == 0:
            print("Exiting...")
            print(
                "A combination of cities, task and subtask have been chosen, where there are no query/database images.")
            print("Try choosing a different subtask or more cities")
            sys.exit()

        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx)
        #self.qIdx = np.asarray(self.qIdx,dtype=object) # wired bugs but works for the warnings
        self.qImages = np.asarray(self.qImages)
        # self.pIdx = np.asarray(self.pIdx)
        self.pIdx = np.asarray(self.pIdx,dtype=object)
        # self.nonNegIdx = np.asarray(self.nonNegIdx)
        self.nonNegIdx = np.asarray(self.nonNegIdx,dtype=object)
        self.dbImages = np.asarray(self.dbImages)

        if save:
            np.save(join(root_dir, 'npys_patch_netvlad', 'msls_' + self.mode + '_qIdx.npy'), self.qIdx)
            np.save(join(root_dir, 'npys_patch_netvlad', 'msls_' + self.mode + '_qImages.npy'), self.qImages)
            np.save(join(root_dir, 'npys_patch_netvlad', 'msls_' + self.mode + '_dbImages.npy'), self.dbImages)
            np.save(join(root_dir, 'npys_patch_netvlad', 'msls_' + self.mode + '_pIdx.npy'), self.pIdx)
            np.save(join(root_dir, 'npys_patch_netvlad', 'msls_' + self.mode + '_nonNegIdx.npy'), self.nonNegIdx)
            if mode == 'val':
                np.save(join(root_dir, 'npys_patch_netvlad', 'msls_' + self.mode + '_all_pos_indices.npy'), np.array(self.all_pos_indices))
                np.save(join(root_dir, 'npys_patch_netvlad', 'msls_' + self.mode + '_qEndPosList.npy'), np.array(self.qEndPosList))
                np.save(join(root_dir, 'npys_patch_netvlad', 'msls_' + self.mode + '_dbEndPosList.npy'), np.array(self.dbEndPosList))

        # decide device type ( important for triplet mining )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.threads = threads
        self.bs = bs

        if mode == 'train':
            # for now always 1-1 lookup.
            self.negCache = np.asarray([np.empty((0,), dtype=int)] * len(self.qIdx))
            self.weights = np.ones(len(self.qIdx)) / float(len(self.qIdx))


    @staticmethod
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

    @staticmethod
    def filter(seqKeys, seqIdxs, center_frame_condition):
        keys, idxs = [], []
        for key, idx in zip(seqKeys, seqIdxs):
            if idx[len(idx) // 2] in center_frame_condition:
                keys.append(key)
                idxs.append(idx)
        return keys, np.asarray(idxs)

    @staticmethod
    def collate_fn(batch):
        """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

        Args:
            batch: list of tuple (query, positive, negatives).
                - query: torch tensor of shape (3, h, w).
                - positive: torch tensor of shape (3, h, w).
                - negative: torch tensor of shape (n, 3, h, w).
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """

        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None

        query, positive, negatives, indices = zip(*batch)

        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
        negatives = torch.cat(negatives, 0)
        indices = list(itertools.chain(*indices))

        return query, positive, negatives, negCounts, indices

    def __len__(self):
        return len(self.triplets)

    def new_epoch(self):

        # find how many subset we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)

        # get all indices
        arr = np.arange(len(self.qIdx))

        # apply positive sampling of indices
        arr = random.choices(arr, self.weights, k=len(arr))

        # calculate the subcache indices
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)

        # reset subset counter
        self.current_subset = 0

    def update_subcache(self, net=None, outputdim=None, epoch_num=0):
        # reset triplets, here only data idx is stored
        self.triplets = []

        # if there is no network associate to the cache, then we don't do any hard negative mining.
        # Instead we just create some naive triplets based on distance.
        if net is None:# or epoch_num <= 2:
            qidxs = np.random.choice(len(self.qIdx), self.cached_queries, replace=False)

            for q in qidxs:

                # get query idx
                qidx = self.qIdx[q]

                # get positives
                pidxs = self.pIdx[q]

                # choose a random positive (within positive range (default 10 m))
                pidx = np.random.choice(pidxs, size=1)[0]

                # get negatives
                while True:
                    nidxs = np.random.choice(len(self.dbImages), size=self.nNeg)

                    # ensure that non of the choice negative images are within the negative range (default 25 m)
                    if sum(np.in1d(nidxs, self.nonNegIdx[q])) == 0:
                        break

                # package the triplet and target
                triplet = [qidx, pidx, *nidxs]
                target = [-1, 1] + [0] * len(nidxs)

                self.triplets.append((triplet, target))

            # increment subset counter
            self.current_subset += 1

            return

        # take n query images
        if self.current_subset >= len(self.subcache_indices):
            tqdm.write('Reset epoch - FIX THIS LATER!')
            self.current_subset = 0
        qidxs = np.asarray(self.subcache_indices[self.current_subset])

        # take their positive in the database
        pidxs = np.unique([i for idx in self.pIdx[qidxs] for i in idx])

        # take m = 5*cached_queries is number of negative images
        nidxs = np.random.choice(len(self.dbImages), self.cached_negatives, replace=False)

        # and make sure that there is no positives among them
        nidxs = nidxs[np.in1d(nidxs, np.unique([i for idx in self.nonNegIdx[qidxs] for i in idx]), invert=True)]

        # make dataloaders for query, positive and negative images
        opt = {'batch_size': self.bs, 'shuffle': False, 'num_workers': self.threads, 'pin_memory': True}
        qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages[qidxs], transform=self.transform), **opt)
        ploader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[pidxs], transform=self.transform), **opt)
        nloader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[nidxs], transform=self.transform), **opt)

        # calculate their descriptors
        net.eval()
        with torch.no_grad():

            # initialize descriptors
            qvecs = torch.zeros(len(qidxs), outputdim).to(self.device)
            pvecs = torch.zeros(len(pidxs), outputdim).to(self.device)
            nvecs = torch.zeros(len(nidxs), outputdim).to(self.device)

            bs = opt['batch_size']

            # compute descriptors
            for i, batch in tqdm(enumerate(qloader), desc='compute query descriptors', total=len(qidxs) // bs,
                                 position=2, leave=False):
                X, y = batch
                image_encoding = net.encoder(X.to(self.device))
                vlad_encoding = net.pool(image_encoding)
                qvecs[i * bs:(i + 1) * bs, :] = vlad_encoding
            for i, batch in tqdm(enumerate(ploader), desc='compute positive descriptors', total=len(pidxs) // bs,
                                 position=2, leave=False):
                X, y = batch
                image_encoding = net.encoder(X.to(self.device))
                vlad_encoding = net.pool(image_encoding)
                pvecs[i * bs:(i + 1) * bs, :] = vlad_encoding
            for i, batch in tqdm(enumerate(nloader), desc='compute negative descriptors', total=len(nidxs) // bs,
                                 position=2, leave=False):
                X, y = batch
                image_encoding = net.encoder(X.to(self.device))
                vlad_encoding = net.pool(image_encoding)
                nvecs[i * bs:(i + 1) * bs, :] = vlad_encoding

        tqdm.write('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        pScores = torch.mm(qvecs, pvecs.t())
        pScores, pRanks = torch.sort(pScores, dim=1, descending=True)

        # calculate distance between query and negatives
        nScores = torch.mm(qvecs, nvecs.t())
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)

        # convert to cpu and numpy
        pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()

        # selection of hard triplets
        for q in range(len(qidxs)):

            qidx = qidxs[q]

            # find positive idx for this query (cache idx domain)
            cached_pidx = np.where(np.in1d(pidxs, self.pIdx[qidx]))

            # find idx of positive idx in rank matrix (descending cache idx domain)
            pidx = np.where(np.in1d(pRanks[q, :], cached_pidx))

            # take the closest positve
            dPos = pScores[q, pidx][0][0]

            # get distances to all negatives
            dNeg = nScores[q, :]

            # how much are they violating
            loss = dPos - dNeg + self.margin ** 0.5
            violatingNeg = 0 < loss

            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg:
                continue

            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]

            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]

            # select the closest positive (back to cache idx domain)
            cached_pidx = pRanks[q, pidx][0][0]

            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            pidx = pidxs[cached_pidx]
            hardestNeg = nidxs[cached_hardestNeg]

            # package the triplet and target
            triplet = [qidx, pidx, *hardestNeg]
            target = [-1, 1] + [0] * len(hardestNeg)

            self.triplets.append((triplet, target))

        #print('triplet number:\t',len(self.triplets))
        # increment subset counter
        self.current_subset += 1

    def __getitem__(self, idx):
        # get triplet
        triplet, target = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]
        # print("self.qImages[qidx]: ", self.qImages[qidx])
        # print("self.dbImages[pidx]: ", self.dbImages[pidx])
        # print("nidx: ", nidx)
        # load images into triplet list
        query = self.transform(Image.open(self.qImages[qidx]).convert('RGB'))
        positive = self.transform(Image.open(self.dbImages[pidx]).convert('RGB'))
        negatives = [self.transform(Image.open(self.dbImages[idx]).convert('RGB')) for idx in nidx]
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [qidx, pidx] + nidx
    
        # uncomment this for MSLS style visualization
        # # load images into triplet list
        # output = [torch.stack([self.transform(Image.open(im)) for im in self.qImages[qidx].split(',')])]
        # output.append(torch.stack([self.transform(Image.open(im)) for im in self.dbImages[pidx].split(',')]))
        # output.extend([torch.stack([self.transform(Image.open(im)) for im in self.dbImages[idx].split(',')]) for idx in nidx])
        # # the size of output and target are identical, and the order matters
        # return torch.cat(output), torch.tensor(target)
