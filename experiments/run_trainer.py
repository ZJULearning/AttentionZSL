import os
import os.path as osp
import sys
import argparse

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from trainer import trainer
from configs.default import cfg, update_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument('--device', type=str, dest='device', default='')
    parser.add_argument('--fine_tuning', type=int, dest='fine_tuning', default=1)
    parser.add_argument('--checkpoint_base', type=str, dest='checkpoint_base', default='./checkpoint')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    update_datasets()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device if args.device else cfg.gpu

    checkpoint_dir = osp.join(args.checkpoint_base, cfg.ckpt_name)
    print("[*] Target Checkpoint Path: {}".format(checkpoint_dir))
    model = trainer(args, cfg, checkpoint_dir)
    model.run()

if __name__ == "__main__":
    main()
