import os
import glob
import torch
import numpy as np
from tqdm import tqdm

import models

from torch.utils.data import DataLoader
from data_factory.evaluator_factory import EvaluationFactory

croptimes = {"FiveCrop": 5, "TenCrop": 10, "resize_crop": 1, "TenCrop+Resize": 11}

def load_model(checkpoint_path, cfg, device):
    model, image_size = models.load_model(cfg.model, k=cfg.attr_dims)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    return model, image_size

def parse_checkpoint_paths(epoch_str, base_path):
    if epoch_str == "all":
        pth = glob.glob(os.path.join(base_path, "ckpt_epoch_*.pth"))
        pth = sorted(pth, key=lambda p: int(os.path.basename(p).replace("ckpt_epoch_", "").replace(".pth", "")), reverse=True)
    else:
        epochs = epoch_str.strip().split(',')
        epochs = [int(e.strip()) for e in epochs]
        epochs = sorted(epochs)[::-1]
        pth = [os.path.join(base_path, "ckpt_epoch_{}.pth".format(e)) for e in epochs]
    return pth

def inference(cfg, paths, eval_cls, batch_size, model, device, image_size):
    pred_attr = []
    pred_latent = []
    
    pred_all_paths = []
    for cls_indice, cls in enumerate(eval_cls):
        pred_all_paths.extend(paths[cls_indice])
    evaluate_dataset = EvaluationFactory(pred_all_paths, cfg.test.imload_mode, image_size)
    evaluate_dataloader = DataLoader(
        dataset=evaluate_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    with torch.no_grad():
        for ims in tqdm(evaluate_dataloader):
            ims = ims.to(device)
            ims = ims.view(-1, 3, ims.size()[-2], ims.size()[-1])
            pred_attr_batch, pred_latent_batch = model(ims)
            pred_attr.append(pred_attr_batch.detach().cpu().numpy())
            pred_latent.append(pred_latent_batch.detach().cpu().numpy())
    
    pred_attr = np.concatenate(pred_attr, axis=0)
    pred_latent = np.concatenate(pred_latent, axis=0)
    
    return pred_attr, pred_latent

def crop_voting(cfg, labels):
    if croptimes[cfg.test.imload_mode] == 1:
        return labels

    labels_ret = [] 
    for label in labels:
        label = label.reshape(-1, croptimes[cfg.test.imload_mode])
        voted_label = np.zeros(len(label))
        for i in range(len(label)):
            bincount = np.bincount(label[i, :])
            bincount_label = np.argmax(bincount)
            bincount_voting = np.max(bincount)
            if bincount_voting == 1:
                voted_label[i] = label[i, -1]
            else:
                voted_label[i] = bincount_label
        labels_ret.append(voted_label)
    return labels_ret


def report_evaluation(path, rst_dict):
    with open(path, 'w') as fp:
        for pth, rst in rst_dict.items():
            fp.write(pth + '\n')
            best_rst = 0.0
            best_sa_steps = -1
            for i, r in enumerate(rst):
                if isinstance(r, list):
                    if r[2] > best_rst:
                        best_rst = r[2]
                        best_sa_steps = i
                else:
                    if r > best_rst:
                        best_rst = r
                        best_sa_steps = i
            r = rst[best_sa_steps]
            if isinstance(r, list):
                fp.write('\t tr: {:.5f}, ts: {:.5f}, H: {:.5f}\n'.format(r[0], r[1], r[2]))
            else:
                fp.write('\t acc: {:.5f}\n'.format(r))
