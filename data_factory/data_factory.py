import os
from os.path import join 
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from util import *
from .data_transform import data_transform
from .balanced_batch_sampler import BalancedBatchSampler

class ZSLDataFactory(Dataset):
    def __init__(
        self, data_path, attr_file, cls_file, train_file, transform, im_size
    ):
        self.transform = data_transform(transform, im_size[0])
        all_attr = prepare_attribute_matrix(attr_file)
        all_cls_names = prepare_cls_names(cls_file)

        self.all_train_files = loadtxt(train_file)

        all_train_cls = [f[:f.find('/')] for f in self.all_train_files]
        self.factory_cls_names = sorted(list(set(all_train_cls)))

        cls_indice = []
        for cls in self.factory_cls_names:
            if cls in all_cls_names:
                cls_indice.append(all_cls_names.index(cls))

        assert(len(cls_indice) == len(self.factory_cls_names))
        self.attr_selected = all_attr[np.asarray(cls_indice), :]

        self.all_im_names = [
            os.path.join(data_path, im_path) for im_path in self.all_train_files
        ]
        self.labels = [
            self.factory_cls_names.index(cls) for cls in all_train_cls
        ]

    def selected_attr(self):
        return self.attr_selected

    def size(self):
        return len(self)

    def __getitem__(self, index):
        im = self.transform(load_pilimage(self.all_im_names[index]))
        attr_mask = np.zeros((1, len(self.factory_cls_names)), dtype=np.float32)
        attr_mask[0, self.labels[index]] = 1

        return im, attr_mask

    def __len__(self):
        return len(self.all_im_names)

def build_data_loader(
    data_path, attr_file, cls_file, train_file, transform, im_size, n_classes, n_samples
):
    dataset = ZSLDataFactory(
        data_path, attr_file, cls_file, train_file, transform, im_size
    )
    batch_sampler = BalancedBatchSampler(
        dataset.labels, n_classes, n_samples
    )
    return dataset.selected_attr(), dataset.size(), DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=4
    )
