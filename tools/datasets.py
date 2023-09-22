"""
This file is derived from the following sources:
- https://github.com/shoopshoop/OMC/blob/main/models/deeplabcut/OpenMonkey.py
- https://github.com/yaoxx340/MonkeyDataset/blob/main/OpenMonkey.py
Modifications were required to accomodate the annotation format used in
the OpenMonkeyCompetetion dataset
"""

import json
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys
from itertools import compress


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


# REye-LEye-Nose-Head-Neck-RShoulder-RElbow-RHand-LShoulder-LElbow-LHand-Hip-RKnee-RFoot-LKnee-LFoot-Tail
#  0   1    2    3    4       5       6      7      8        9      10   11   12    13   14     15   16
colors = [
    (255, 153, 204),  # REye-Nose
    (255, 153, 204),  # LEye-Nose
    (153, 51, 255),  # nose-head
    (51, 51, 255),  # head-neck
    (204, 102, 0),  # neck-RShoulder
    (230, 140, 61),  # RShoulder-RElbow
    (255, 178, 102),  # RElbow-RHand
    (255, 102, 102),  # neck-LShoulder
    (255, 179, 102),  # LShoulder-LElbow
    (255, 255, 102),  # LElbow-LHand
    (51, 153, 255),  # neck-hip
    (102, 204, 0),  # hip-RKnee
    (204, 255, 153),  # RKnee-RFoot
    (0, 204, 102),  # hip-LKnee
    (102, 255, 178),  # LKnee-LFoot
    (102, 255, 255),  # hip-tail
]
I = np.array([1, 2, 3, 4, 5, 6, 4, 8, 9, 4, 11, 12, 11, 14, 11])
J = np.array([2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])


class Dataset:
    # def __init__(self, annotation_file=None, root=None):
    def __init__(self, project):
        self.project = project
        self.dataset, self.split, self.landmarks, self.specs, self.imgs = dict(), dict(), dict(), dict(), dict()
        self.bbox, self.visibility = None, None
        if self.project.db is not None:
            self.dataset, self.n_train, self.n_val = self.create_dict_from_json()
            assert type(self.dataset) == dict, 'annotation file format {} not supported'.format(type(self.dataset))
            self.createIndex()
            print(f'Annotations loaded with {self.n_train} training examples and {self.n_val} validation examples')

    def createIndex(self):
        # create index
        i = 0
        split, landmarks, specs, imgs, bbox, visibility = {}, {}, {}, {}, {}, {}
        if 'data' in self.dataset:
            doubled_boolean_keypoints = [x for x in self.project.boolean_keypoints for _ in range(2)]
            for idx, sample in enumerate(self.dataset['data']):
                if sample['species'] in self.project.species:
                    if idx < self.n_train:
                        split[i] = 'train'
                    else:
                        split[i] = 'val'
                    landmarks[i] = list(compress(sample['landmarks'], doubled_boolean_keypoints))
                    imgs[i] = sample['file']
                    specs[i] = sample['species']
                    bbox[i] = sample['bbox']
                    if self.project.only_visible_kpt:
                        visibility[i] = list(compress(sample['visibility'], self.project.boolean_keypoints))
                    else:
                        visibility[i] = [1 for _ in range(len(self.project.keypoints))]
                    i += 1
        # create class members
        self.split = split
        self.landmarks = landmarks
        self.imgs = imgs
        self.specs = specs
        self.bbox = bbox
        self.visibility = visibility

    def create_dict_from_json(self):
        dataset = {'data': []}
        train_annotation = f'./data/{self.project.db}/train_annotation.json'
        train = json.load(open(train_annotation, 'r'))
        n_train = len(train['data'])
        val_annotation = f'./data/{self.project.db}/val_annotation.json'
        val = json.load(open(val_annotation, 'r'))
        n_val = len(val['data'])
        dataset['data'].extend(train['data'])
        dataset['data'].extend(val['data'])
        return dataset, n_train, n_val

