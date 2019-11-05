import os
import os.path as osp
import sys
import argparse

import torch

import numpy as np 
from easydict import EasyDict as edict

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))
from util import *
from configs.default import cfg, update_datasets
from evaluation.utils import load_model, parse_checkpoint_paths, inference, crop_voting, report_evaluation
from evaluation.metric import mca, harmonic_mean
from evaluation.self_adaptation import self_adaptation

def preprocess_ss_c(cfg, all_cls_names):
    unseen_cls = loadtxt(cfg.ss_test)
    unseen_cls_indice = []
    for cls in unseen_cls:
        unseen_cls_indice.append(all_cls_names.index(cls))

    ground_truth_indice = []
    eval_paths = []
    for cls_indice, cls in enumerate(unseen_cls):
        im_names = [os.path.join(cfg.image, cls, f) for 
                f in os.listdir(os.path.join(cfg.image, cls)) if is_image(f)]
        eval_paths.append(im_names)
        ground_truth_indice.extend([cls_indice] * len(im_names))
    return unseen_cls, unseen_cls_indice, eval_paths, ground_truth_indice

def preprocess_ps_c(cfg, all_cls_names):
    unseen_cls = loadtxt(cfg.ps_unseen_cls)
    unseen_files = loadtxt(cfg.ps_test_unseen)
    unseen_files_cls = [f[:f.find('/')] for f in unseen_files]

    unseen_cls_indice = []
    for cls in unseen_cls:
        unseen_cls_indice.append(all_cls_names.index(cls))

    ground_truth_indice = []
    eval_paths = []
    for cls_indice, cls in enumerate(unseen_cls):
        im_names = [os.path.join(cfg.image, f) 
                for f, fcls in zip(unseen_files, unseen_files_cls) if cls == fcls]
        eval_paths.append(im_names)
        ground_truth_indice.extend([cls_indice] * len(im_names))
    return unseen_cls, unseen_cls_indice, eval_paths, ground_truth_indice

def preprocess_ps_g(cfg, all_cls_names):
    seen_cls = loadtxt(cfg.ps_seen_cls)
    unseen_cls = loadtxt(cfg.ps_unseen_cls)
    eval_cls = seen_cls + unseen_cls
    
    seen_files = loadtxt(cfg.ps_test_seen)
    unseen_files = loadtxt(cfg.ps_test_unseen)
    all_files = seen_files + unseen_files
    all_files_cls = [f[:f.find('/')] for f in all_files]

    cls_indice = []
    for cls in eval_cls:
        cls_indice.append(all_cls_names.index(cls))

    ground_truth_indice = []
    eval_paths = []
    for i, cls in enumerate(eval_cls):
        im_names = [os.path.join(cfg.image, f) 
                for f, fcls in zip(all_files, all_files_cls) if cls == fcls]
        eval_paths.append(im_names)
        ground_truth_indice.extend([i] * len(im_names))
    return eval_cls, cls_indice, eval_paths, ground_truth_indice

def parse_all(cfg, checkpoint_full_path):
    all_attr = prepare_attribute_matrix(cfg.attribute)
    all_cls_names = prepare_cls_names(cfg.class_name)
    if cfg.test.setting == 'c' and cfg.split == 'SS':
        selected_cls, selected_cls_indice, eval_paths, gt_indice = \
                preprocess_ss_c(cfg, all_cls_names)
    elif cfg.test.setting == 'c' and cfg.split == 'PS':
        selected_cls, selected_cls_indice, eval_paths, gt_indice = \
                preprocess_ps_c(cfg, all_cls_names)
    elif cfg.test.setting == 'g' and cfg.split == 'PS':
        selected_cls, selected_cls_indice, eval_paths, gt_indice = \
                preprocess_ps_g(cfg, all_cls_names)
    else:
        raise NotImplementedError
    ckpt_pths = parse_checkpoint_paths(cfg.test.epoch, checkpoint_full_path)
    selected_attr = all_attr[np.asarray(selected_cls_indice), :]
    return edict({
            "selected_cls": selected_cls, "selected_attr": selected_attr, "eval_paths": eval_paths,
            "gt_indice": gt_indice, "ckpt_pths": ckpt_pths
        })

def predict_all(pth, cfg, parsing, device):
    saving_pred_path = pth.replace(
        'ckpt', f"{cfg.db_name}_pred_{cfg.test.setting}_{cfg.test.imload_mode}"
    ).replace(".pth", ".npy")
    if not os.path.exists(saving_pred_path):
        model, image_size = load_model(pth, cfg, device)
        pred_attr, pred_latent = inference(
            cfg, parsing.eval_paths, parsing.selected_cls, cfg.test.batch_size, model, device, image_size
        )
        if cfg.test.save_predictions:
            np.save(saving_pred_path, {'attr': pred_attr, 'latent': pred_latent})
    else:
        data = np.load(saving_pred_path)[()]
        pred_attr = data['attr']
        pred_latent = data['latent']
    return pred_attr, pred_latent

def record_evaluation(cfg, pred_labels, parsing):
    eval_rst = []
    if cfg.test.setting == "c":
        for i, labels in enumerate(pred_labels):
            avg_cls_acc = mca(parsing.gt_indice, labels, len(parsing.selected_attr))
            eval_rst.append(avg_cls_acc)
            print("\t {:d}\t{:.5f}".format(i, avg_cls_acc))
    elif cfg.test.setting == "g":
        saving_results_str = ""
        for i, labels in enumerate(pred_labels):
            tr, ts, H = harmonic_mean(parsing.gt_indice, labels, len(parsing.selected_attr), cfg.nseen)
            print("\t {:d}\t tr:{:.5f}\t ts:{:.5f}\t H:{:.5f}".format(i, tr, ts, H))
            eval_rst.append([tr, ts, H])
    return eval_rst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument('--device', type=str, dest='device', default='')
    parser.add_argument('--imload_mode', '-i', type=str, dest='imload_mode', default='')
    parser.add_argument('--checkpoint_base', '-c', default="./checkpoint_best_sa")
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    update_datasets()
    
    device = args.device if args.device else cfg.gpu
    cfg.test.imload_mode = args.load if args.imload_mode else cfg.test.imload_mode
    cfg.test.batch_size  = 256

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=device

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    parsing = parse_all(cfg, os.path.join(args.checkpoint_base, cfg.ckpt_name))

    report_rst  = {}
    for pth in parsing.ckpt_pths:
        print(os.path.basename(pth))
        pred_attr, pred_latent = predict_all(pth, cfg, parsing, device)
        labels = self_adaptation(
            pred_attr, pred_latent, parsing.selected_attr, step=cfg.test.self_adaptions, 
            ensemble=cfg.self_adaptation.ensemble, metric=cfg.self_adaptation.similarity_metric
        )
        labels = crop_voting(cfg, labels)
        report_rst[os.path.basename(pth)] = record_evaluation(cfg, labels, parsing)

    report_evaluation(cfg.test.report_path, report_rst)

if __name__ == "__main__":
    main()
