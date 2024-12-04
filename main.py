from easydict import EasyDict as edict
import yaml
import argparse
import numpy as np
import random
import torch
from optimize_final import FinalTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_file', type=str, default='config/synthetic.yaml', help='path to config file')

    args = parser.parse_args()

    config_file = args.config_file
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    ### optimize
    trainer = FinalTrainer(config)
    trainer.optimize()

if __name__ =='__main__':
    main()
