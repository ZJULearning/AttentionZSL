import os
import sys
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim import Adagrad
from torch.utils.data import DataLoader

import models
import data_factory
import losses
from glob import glob

class trainer(object):
    def __init__(self, args, cfg, checkpoint_dir):
        self.batch_size = cfg.train.batch_size
        self.learning_rate = cfg.train.lr
        self.epochs = cfg.train.epochs
        self.start_epoch = 1
        self.lr_decay_epochs = cfg.train.lr_decay
        self.log_interval = cfg.train.log_inter
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = cfg.train.ckpt_inter
        self.lambda_ = cfg.train.beta
        self.attr_dims = cfg.attr_dims
        self.device = torch.device(
            'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        self.triplet_batch = 4

        self.fnet, self.optimizer, self.im_size = self.build_model(cfg)
        if os.path.exists(cfg.ckpt_name) and args.fine_tuning:
            pth = glob(os.path.join(cfg.ckpt_name, "ckpt_epoch_*.pth"))
            pth = sorted(pth, 
                         key=lambda p: int(os.path.basename(p).replace("ckpt_epoch_", "").replace(".pth", "")), 
                         reverse=True)
            if pth:
                self.load(pth[0])
                self.start_epoch = int(
                        ''.join([c for c in os.path.basename(pth[0]) if c.isdigit()])
                        ) + 1
            
        self.attr_data, self.dataset_size, self.data_loader = self.prepare_dataloader(cfg)
        #self.attr_data = torch.from_numpy(self.attr_data).to(self.device)
        self.online_zsl_loss = losses.ZeroShotLearningLoss(self.attr_data)
        
        if cfg.train.triplet_mode == "batch_all":
            self.online_triplet_loss = \
                        losses.BatchAllTripletLoss(self.device, 
                                                   self.batch_size // self.triplet_batch, 
                                                   self.triplet_batch)
        else:
            self.online_triplet_loss = \
                        losses.BatchHardTripletLoss(self.device,
                                                    self.batch_size // self.triplet_batch,
                                                    self.triplet_batch)

    def build_model(self, cfg):
        fnet, im_size = models.load_model(cfg.model, k=self.attr_dims)
        optimizer = Adam(fnet.parameters(), self.learning_rate)
        return fnet.to(self.device), optimizer, im_size 

    def prepare_dataloader(self, cfg):
        if cfg.split == "SS":
            dataset = data_factory.SSFactory(
                cfg.image, cfg.attribute, cfg.class_name, cfg.ss_train, 
                transform=cfg.train.data_aug, batch_size=self.batch_size, im_size=self.im_size
            )
        elif cfg.split == "PS":
            dataset = data_factory.PSFactory(
                cfg.image, cfg.attribute, cfg.class_name, cfg.ps_train, 
                transform=cfg.train.data_aug, batch_size=self.batch_size, im_size=self.im_size
            )
        else:
            raise NotImplementedError
        attr_data = dataset.selected_attr()
        attr_data = torch.from_numpy(attr_data).to(self.device)
        dataset_size = dataset.size()

        dataset.im_size = self.im_size
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        return attr_data, dataset_size, data_loader

    def exp_lr_scheduler(self, epoch, lr_decay_epoch, lr_decay=0.1):
        if epoch % lr_decay_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay

    def run(self):
        for e in range(self.start_epoch, self.epochs + self.start_epoch):
            self.exp_lr_scheduler(e, self.lr_decay_epochs)
            self.fnet.train()

            agg_loss = {"loss": 0., "attr_loss": 0., "latent_loss": 0}
            current_size = 0
            for batch_id, (x, attr_mask) in enumerate(self.data_loader):
                current_size += self.batch_size

                x = x.to(self.device)   # 1 x 3#batch_size x 299 x 299
                attr_mask = attr_mask.to(self.device).squeeze() # 1 x #batch_size x k

                attr_embed, latent_embed = \
                        self.fnet(x.view(-1, 3, x.size(2), x.size(3)))
                latent_embed = latent_embed.view(
                        self.batch_size // self.triplet_batch, self.triplet_batch, latent_embed.size(1))

                latent_loss = self.online_triplet_loss(latent_embed)
                attr_loss = self.online_zsl_loss(attr_embed,
                                                 attr_mask)
                attr_loss = attr_loss * self.lambda_
                loss = latent_loss + attr_loss 
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fnet.parameters(), 5)
                self.optimizer.step()

                agg_loss["loss"] += loss.item()
                agg_loss["attr_loss"] += attr_loss.item()
                agg_loss["latent_loss"] += latent_loss.item()
                if current_size % self.log_interval == 0:
                    mesg = "[E{} {}/{} Cur/Agg]\t tl:{:.3f}/{:.3f}\t al:{:.3f}/{:.3f}\t total:{:.3f}/{:.3f}".format(
                        e, current_size, self.dataset_size,
                        latent_loss.item(),
                        agg_loss["latent_loss"] / (batch_id + 1), 
                        attr_loss.item(),
                        agg_loss["attr_loss"] / (batch_id + 1),
                        loss.item(),
                        agg_loss["loss"] / (batch_id + 1)
                    )
                    print(mesg)
            
            if self.checkpoint_dir is not None and e % self.checkpoint_interval == 0:
                self.save(self.checkpoint_dir, e)

    def save(self, checkpoint_dir, e):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.fnet.eval()
        ckpt_model_filename = os.path.join(checkpoint_dir, "ckpt_epoch_" + str(e) + ".pth")
        state_dict = self.fnet.state_dict()
        torch.save(state_dict, ckpt_model_filename)
        self.fnet.train()

    def load(self, checkpoint_dir):
        state_dict = torch.load(checkpoint_dir)
        
        self.fnet.load_state_dict(state_dict)
        self.fnet.to(self.device)
