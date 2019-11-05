import os.path as osp
from .collections import AttrDict

cfg = AttrDict()

cfg.ckpt_name   = "default"
cfg.db_name     = "AWA2"
cfg.split       = "PS"
cfg.gpu         = "3"
cfg.model       = "GoogleNet"

# For evaluation settings
cfg.test                  = AttrDict()
cfg.test.batch_size       = 256
cfg.test.setting          = "c" # c => conventional, g => generalized
cfg.test.epoch            = "all"
cfg.test.imload_mode      = "resize_crop"
cfg.test.self_adaptions   = 10
cfg.test.save_predictions = True

cfg.test.report_base_path = osp.abspath("./predictions")

# For training settings
cfg.train              = AttrDict()
cfg.train.batch_size   = 32
cfg.train.epochs       = 15
cfg.train.lr_decay     = 20
cfg.train.lr           = 1e-5
cfg.train.beta         = 1.0
cfg.train.log_inter    = 32
cfg.train.ckpt_inter   = 1
cfg.train.data_aug     = "resize_random_crop" # auto, patch_crop
cfg.train.triplet_mode = "batch_hard" # batch_hard, batch_all

cfg.self_adaptation = AttrDict()
cfg.self_adaptation.similarity_metric = "cosine"
cfg.self_adaptation.ensemble = True

cfg.hybrid = AttrDict()
cfg.hybrid.similarity_metric = "cosine"
cfg.hybrid.ensemble = True

def update_datasets(self_adaptation=False):
    if cfg.db_name == "AwA2":
        cfg.data_root = "./data/AwA2/"
        cfg.attr_dims = 85
        cfg.nseen = 40
    elif cfg.db_name == "CUB":
        cfg.data_root = "./data/CUB/"
        cfg.attr_dims = 312
        cfg.nseen = 150
    elif cfg.db_name == "SUN":
        cfg.data_root = "./data/SUN/"
        cfg.attr_dims = 102
        cfg.nseen = 645
    else:
        raise NotImplementedError

    cfg.attribute       = osp.join(cfg.data_root, "predicate-matrix-continuous.txt")
    cfg.class_name      = osp.join(cfg.data_root, "classes.txt")
    cfg.image           = osp.join(cfg.data_root, "JPEGImages")
    cfg.ss_train        = osp.join(cfg.data_root, "trainclasses.txt")    
    cfg.ss_test         = osp.join(cfg.data_root, "testclasses.txt")
    cfg.ps_train        = osp.join(cfg.data_root, "proposed_split/trainval_ps.txt")
    cfg.ps_test_seen    = osp.join(cfg.data_root, "proposed_split/test_seen_ps.txt")
    cfg.ps_test_unseen  = osp.join(cfg.data_root, "proposed_split/test_unseen_ps.txt")
    cfg.ps_seen_cls     = osp.join(cfg.data_root, "proposed_split/seen_cls.txt")
    cfg.ps_unseen_cls   = osp.join(cfg.data_root, "proposed_split/unseen_cls.txt")

    postfix = "sa" if self_adaptation else "hybrid"
    cfg.test.report_path = osp.join(
        cfg.test.report_base_path, f"{cfg.ckpt_name}_{cfg.test.setting}_{cfg.test.imload_mode}_{postfix}.txt"
    )
