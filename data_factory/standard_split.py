import os
from os.path import join 
import numpy as np
import torch
from torch.utils.data import Dataset
from random import shuffle
from copy import deepcopy
from util import *
from .data_transform import data_transform

class SSFactory(Dataset):
    def __init__(self, 
                data_path, 
                attr_file, 
                cls_file, 
                cls_indice_file, 
                transform,
                batch_size,
                im_size,
                triplet_selections=4):
        self.triplet_k = triplet_selections
        self.triplet_p = batch_size // self.triplet_k
        self.transform = data_transform(transform, im_size[0])
        self.dataset_path = data_path

        if not os.path.exists(self.dataset_path):
            raise RuntimeError('[!] dataset not found: {}'.format(self.dataset_path))
        
        all_attr = prepare_attribute_matrix(attr_file)
        all_cls_names = prepare_cls_names(cls_file)

        self.factory_cls_names = loadtxt(cls_indice_file)
        cls_indice = []
        for cls in self.factory_cls_names:
            if cls in all_cls_names:
                cls_indice.append(all_cls_names.index(cls))

        assert(len(cls_indice) == len(self.factory_cls_names))
        self.attr_selected = all_attr[np.asarray(cls_indice), :]

        # Build File Dictionary
        self.file_dict = {}
        self.batch_count = {}
        self.batch_sentry = {}
        self._length = 0
        self._size   = 0
        for cls in self.factory_cls_names:
            im_path = os.path.join(self.dataset_path, cls)
            im_names = [os.path.join(im_path, f) for f in os.listdir(im_path) if is_image(f)]
            shuffle(im_names)
            self.file_dict[cls] = im_names
            self.batch_count[cls] = len(im_names) // self.triplet_k
            self.batch_sentry[cls] = 0
            self._length += self.batch_count[cls]
            self._size   += len(im_names)

        self.factory_cls_names_cp = deepcopy(self.factory_cls_names)

    def __len__(self):
        return self._length // self.triplet_p

    def __getitem__(self, index):
        batch_cls = []
        im, attr_indice = [], []
        for i in range(self.triplet_p):
            selected_cls = randpick(self.factory_cls_names_cp, exception=batch_cls)
            batch_cls.append(selected_cls)
            sentry = self.batch_sentry[selected_cls]
            for j in range(self.triplet_k):
                im.append(self.transform(load_pilimage(
                    self.file_dict[selected_cls][sentry * self.triplet_k + j])))
                attr_indice.append(self.factory_cls_names.index(selected_cls))
            self.batch_sentry[selected_cls] += 1
            if self.batch_sentry[selected_cls] == self.batch_count[selected_cls]:
                self.factory_cls_names_cp.remove(selected_cls)

        if len(self.factory_cls_names_cp) < self.triplet_p:
            self.factory_cls_names_cp = deepcopy(self.factory_cls_names)
            for cls in self.factory_cls_names:
                shuffle(self.file_dict[cls])
                self.batch_sentry[cls] = 0

        im = torch.cat(im, dim=0)
        attr_mask = np.zeros((len(attr_indice), len(self.factory_cls_names)), dtype=np.float32)
        for i, j in enumerate(attr_indice):
            attr_mask[i, j] = 1
        
        return im, attr_mask

    def selected_attr(self):
        return self.attr_selected

    def size(self):
        return self._size
