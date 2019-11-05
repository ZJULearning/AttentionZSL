import numpy as np
from PIL import Image
import torch
import sys
import os

def th_cos_similarity(x, y, dim=2):
    x_norm = x.norm(p=2, dim=dim);
    y_norm = y.norm(p=2, dim=dim);
    sim = torch.sum(x * y, 2) / (x_norm * y_norm)
    return sim 

def th_normalize(x, dim=1, p=2, keepdim=True):
    norm = x.norm(p=p, dim=dim, keepdim=keepdim)
    return x.div(norm.expand_as(x))

def prepare_attribute_matrix(attr_file):
    res = np.loadtxt(attr_file).astype(np.float32)
    res = np.clip(res, a_min=0.0, a_max=None)
    return res

def prepare_cls_names(cls_file):
    """
    cls_file (#cls * 2):
        1 antelope
        2 grizzly+bear
            ...
    """
    res = np.loadtxt(cls_file, dtype=str)
    res = res[:, 1].tolist()
    return res

def loadtxt(path):
    with open(path, 'r') as fp:
        txt = fp.read()
        data = txt.strip().split('\n')
        return data

def randpick(lst, exception=None):
    if not exception:
        return lst[np.random.randint(len(lst))]
    else:
        lst_cp = list(lst)
        if isinstance(exception, list):
            for exc in exception:
                if exc in lst_cp:
                    lst_cp.remove(exc)
        else:
            if exception in lst_cp:
                lst_cp.remove(exception)
        return lst_cp[np.random.randint(len(lst_cp))]

def load_pilimage(path, size=None, options='RGB', crop_im=False):
    if crop_im:
        return crop_image(path, shape=size)
    else:
        im = Image.open(path).convert(options)
        if size is not None:
            im = im.resize(size)
        return im 

def pardir_basename(path):
    return os.path.basename(os.path.abspath(os.path.join(path, os.pardir)))

def ridge_regression(attr_data_file,
                     seen_cls_indice,
                     unseen_cls_indice):
    from sklearn import datasets, linear_model

    all_attributes = prepare_attribute_matrix(attr_data_file)
    all_attributes = all_attributes / np.linalg.norm(all_attributes, axis=1, keepdims=True)

    # 85 * 40 for AwA2
    s_attributes = all_attributes[np.asarray(seen_cls_indice), :].transpose()
    u_attributes = all_attributes[np.asarray(unseen_cls_indice), :]

    beta = []
    regr = linear_model.Ridge(alpha=1.0)
    for i in range(len(u_attributes)):
        regr.fit(s_attributes, u_attributes[i, :])
        beta.append(np.asarray(regr.coef_)[np.newaxis, :])
    beta = np.concatenate(beta, axis=0)

    # 10 * 40, 85 for AwA2
    return beta

def crop_image(path, shape=None, crop='center', option='RGB'):
    img = Image.open(path).convert('RGB')

    if isinstance(shape, (list, tuple)):
        # crop to obtain identical aspect ratio to shape
        width, height = img.size
        target_width, target_height = shape[0], shape[1]

        aspect_ratio = width / float(height)
        target_aspect = target_width / float(target_height)

        if aspect_ratio > target_aspect: # if wider than wanted, crop the width
            new_width = int(height * target_aspect)
            if crop == 'right':
                img = img.crop((width - new_width, 0, width, height))
            elif crop == 'left':
                img = img.crop((0, 0, new_width, height))
            else:
                img = img.crop(((width - new_width) / 2, 0, (width + new_width) / 2, height))
        else: # else crop the height
            new_height = int(width / target_aspect)
            if crop == 'top':
                img = img.crop((0, 0, width, new_height))
            elif crop == 'bottom':
                img = img.crop((0, height - new_height, width, height))
            else:
                img = img.crop((0, (height - new_height) / 2, width, (height + new_height) / 2))

        # resize to target now that we have the correct aspect ratio
        img = img.resize((target_width, target_height))
    elif isinstance(shape, (int, float)):
        width, height = img.size
        large = max(width, height)
        ratio = shape / float(large)
        width_n, height_n = ratio * width, ratio * height
        img = img.resize((int(width_n), int(height_n)))
    return img

def is_image(f):
    valid_extension = ['.bmp', '.jpg', '.jpeg', '.png']
    if [e for e in valid_extension if e in f.lower()]:
        return True
    else:
        return False
