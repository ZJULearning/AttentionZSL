import os
import os.path as osp
import sys
import argparse

import numpy as np 
import torch
from easydict import EasyDict as edict

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))
from util import *
from configs.default import cfg, update_datasets
from evaluation.utils import load_model, parse_checkpoint_paths, inference, crop_voting, report_evaluation, croptimes
from evaluation.metric import mca, harmonic_mean
from evaluation.hybrid import hybrid_labeling

def preprocess_ss(cfg, all_cls_names):
    seen_cls = loadtxt(cfg.ss_train)
    seen_cls_indice = []

    unseen_cls = loadtxt(cfg.ss_test)
    unseen_cls_indice = []
    for cls in unseen_cls:
        unseen_cls_indice.append(all_cls_names.index(cls))
    for cls in seen_cls:
        seen_cls_indice.append(all_cls_names.index(cls))

    train_paths = []
    test_paths  = []

    for cls_indice, cls in enumerate(seen_cls):
        im_names = [os.path.join(cfg.image, cls, f) for
                f in os.listdir(os.path.join(cfg.image, cls)) if is_image(f)]
        train_paths.append(im_names)
        
    for cls_indice, cls in enumerate(unseen_cls):
        im_names = [os.path.join(cfg.image, cls, f) for
                f in os.listdir(os.path.join(cfg.image, cls)) if is_image(f)]
        test_paths.append(im_names)
    return seen_cls_indice, unseen_cls_indice, train_paths, test_paths

def preprocess_ps(cfg, all_cls_names):
    seen_cls = loadtxt(cfg.ps_seen_cls)
    seen_cls_indice = []

    unseen_cls = loadtxt(cfg.ps_unseen_cls)
    unseen_cls_indice = []

    for cls in unseen_cls:
        unseen_cls_indice.append(all_cls_names.index(cls))
    for cls in seen_cls:
        seen_cls_indice.append(all_cls_names.index(cls))

    train_seen_files = loadtxt(cfg.ps_train)
    train_seen_files_cls = [f[:f.find('/')] for f in train_seen_files]
    unseen_files = loadtxt(cfg.ps_test_unseen)
    unseen_files_cls = [f[:f.find('/')] for f in unseen_files]
    test_seen_files = loadtxt(cfg.ps_test_seen)
    test_seen_files_cls = [f[:f.find('/')] for f in test_seen_files]
        
    train_seen_paths = []
    test_unseen_paths = []
    test_seen_paths = []
    for cls_indice, cls in enumerate(unseen_cls):
        im_names = [os.path.join(cfg.image, f) 
                for f, fcls in zip(unseen_files, unseen_files_cls) if cls == fcls]
        test_unseen_paths.append(im_names)
    for cls_indice, cls in enumerate(seen_cls):
        im_names = [os.path.join(cfg.image, f) 
                for f, fcls in zip(train_seen_files, train_seen_files_cls) if cls == fcls]
        train_seen_paths.append(im_names)
    for cls_indice, cls in enumerate(seen_cls):
        im_names = [os.path.join(cfg.image, f) 
                for f, fcls in zip(test_seen_files, test_seen_files_cls) if cls == fcls]
        test_seen_paths.append(im_names)
    return seen_cls_indice, unseen_cls_indice, \
            train_seen_paths, test_seen_paths, test_unseen_paths

def parse_all(cfg):
    attr = prepare_attribute_matrix(cfg.attribute)
    attr = attr / np.linalg.norm(attr, axis=1, keepdims=True)
    class_name = prepare_cls_names(cfg.class_name)

    if cfg.split == 'SS':
        seen_cls_indice, unseen_cls_indice, train_paths, test_unseen_paths = \
            preprocess_ss(cfg, class_name)
        test_seen_paths = None # placeholder
    elif cfg.split == 'PS':
        seen_cls_indice, unseen_cls_indice, train_paths, test_seen_paths, test_unseen_paths = \
            preprocess_ps(cfg, class_name)
        ground_truth_indice = []
    else:
        raise NotImplementedError
    
    if cfg.test.setting == 'c':
        ground_truth_indice = []
        for i, _ in enumerate(unseen_cls_indice):
            ground_truth_indice.extend([i] * len(test_unseen_paths[i]))
    elif cfg.test.setting == 'g':
        assert cfg.split == 'PS'

        for i, _ in enumerate(seen_cls_indice):
            ground_truth_indice.extend([i] * len(test_seen_paths[i]))
        for i, _ in enumerate(unseen_cls_indice):
            ground_truth_indice.extend([i + len(seen_cls_indice)] * len(test_unseen_paths[i]))

    beta = ridge_regression(cfg.attribute, seen_cls_indice, unseen_cls_indice)
    return edict({
        "attr": attr, "seen_cls_indice": seen_cls_indice, "unseen_cls_indice": unseen_cls_indice,
        "train_paths": train_paths, "test_seen_paths": test_seen_paths, "test_unseen_paths": test_unseen_paths,
        "beta": beta, "ground_truth_indice": ground_truth_indice
    })

def predict_all(pth, cfg, parsing, device):
    saving_prototype_path = pth.replace(
        'ckpt', f"{cfg.db_name}_prototype_{cfg.test.setting}_{cfg.test.imload_mode}"
    ).replace(".pth", ".npy")
    saving_pred_path = pth.replace(
        'ckpt', f"{cfg.db_name}_pred_{cfg.test.setting}_{cfg.test.imload_mode}"
    ).replace(".pth", ".npy")

    if not osp.exists(saving_pred_path) or not osp.exists(saving_prototype_path):
        model, image_size = load_model(pth, cfg, device)

    if not os.path.exists(saving_prototype_path):
        pred_attr, pred_latent = inference(
            cfg, parsing.train_paths, parsing.seen_cls_indice, cfg.test.batch_size,
            model=model, device=device, image_size=image_size
        )
        seen_latent_prototype = np.zeros((len(parsing.seen_cls_indice), cfg.attr_dims), dtype=np.float32)
        cnt = 0
        for i, cls_indice in enumerate(parsing.seen_cls_indice):
            end = cnt + len(parsing.train_paths[i]) * croptimes[cfg.test.imload_mode]
            seen_latent_prototype[i, :] = np.mean(pred_latent[cnt: end, :], 0)
            cnt += len(parsing.train_paths[i]) * croptimes[cfg.test.imload_mode]
        unseen_latent_prototype = np.matmul(parsing.beta, seen_latent_prototype)
        latent_prototype = np.concatenate([
            seen_latent_prototype, unseen_latent_prototype], 0)
        if cfg.test.save_predictions:
            np.save(saving_prototype_path, {'prototype': latent_prototype})
    else:
        data = np.load(saving_prototype_path)[()]
        latent_prototype = data['prototype']

    if not os.path.exists(saving_pred_path):
        pred_attr, pred_latent = inference(
            cfg, parsing.test_unseen_paths, parsing.unseen_cls_indice, cfg.test.batch_size,
            model=model, device=device, image_size=image_size
        )
        if parsing.test_seen_paths is not None and cfg.test.setting == 'g':
            pred_attr_seen, pred_latent_seen = inference(
                cfg, parsing.test_seen_paths, parsing.seen_cls_indice, cfg.test.batch_size,
                model=model, device=device, image_size=image_size
            )
            pred_attr = np.concatenate([pred_attr_seen, pred_attr], 0)
            pred_latent = np.concatenate([pred_latent_seen, pred_latent], 0)
        np.save(saving_pred_path, {'attr': pred_attr, 'latent': pred_latent})
    else:
        data = np.load(saving_pred_path)[()]
        pred_attr = data['attr']
        pred_latent = data['latent']
    return latent_prototype, pred_attr, pred_latent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument('--device', type=str, dest='device', default='')
    parser.add_argument('--imload_mode', '-i', type=str, dest='imload_mode', default='')
    parser.add_argument('--checkpoint_base', '-c', default="./checkpoint_best_hybrid")
    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg)
    update_datasets()

    cfg.test.imload_mode = args.imload_mode if args.imload_mode else cfg.test.imload_mode

    checkpoint_paths = parse_checkpoint_paths(cfg.test.epoch, osp.join(args.checkpoint_base, cfg.ckpt_name))
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device if args.device else cfg.gpu
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    parsing = parse_all(cfg)

    report_rst  = {}
    for pth in checkpoint_paths:
        eval_rst = []
        latent_prototype, pred_attr, pred_latent = predict_all(pth, cfg, parsing, device)
        print(os.path.basename(pth))
        if cfg.test.setting == 'c':
            pred_labels = hybrid_labeling(
                pred_attr, pred_latent, parsing.attr[np.asarray(parsing.unseen_cls_indice), :],
                latent_prototype[len(parsing.seen_cls_indice):, :], ensemble=cfg.hybrid.ensemble, metric=cfg.hybrid.similarity_metric
            )
            pred_labels = crop_voting(cfg, [pred_labels])[0]
            avg_cls_acc = mca(parsing.ground_truth_indice, pred_labels, len(parsing.unseen_cls_indice))
            eval_rst.append(avg_cls_acc)
            print("\tMCA: {:.4f}".format(mca(parsing.ground_truth_indice, pred_labels, len(parsing.unseen_cls_indice))))
        elif cfg.test.setting == 'g':
            cls_indice = np.concatenate([parsing.seen_cls_indice, parsing.unseen_cls_indice], 0)
            pred_labels = hybrid_labeling(
                pred_attr, pred_latent, 
                parsing.attr[cls_indice, :],
                latent_prototype, ensemble=cfg.hybrid.ensemble, metric=cfg.hybrid.similarity_metric
            )
            pred_labels = crop_voting(cfg, [pred_labels])[0]
            tr, ts, H = harmonic_mean(parsing.ground_truth_indice, pred_labels, len(cls_indice), len(parsing.seen_cls_indice))
            print("\t tr:{:.5f}\t ts:{:.5f}\t H:{:.5f}".format(tr, ts, H))
            eval_rst.append([tr, ts, H])
        report_rst[os.path.basename(pth)] = eval_rst
        report_evaluation(cfg.test.report_path, report_rst)
            
if __name__ == "__main__":
    with torch.no_grad():
        main()

