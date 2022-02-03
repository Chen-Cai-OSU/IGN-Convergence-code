from signor.utils.random_ import fix_seed
fix_seed()
import os
import sys
from pprint import pprint

from signor.format.format import banner

sys.path.append('../data_loader/')
from data_generator import DataGenerator

sys.path.append('../models/')
from invariant_basic import invariant_basic

sys.path.append('../trainers/')
from trainer import Trainer

sys.path.append('../utils/')
from data_generator import DataGenerator
from config import process_config
from dirs import create_dirs
import doc_utils
from utils import get_args, process_config2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import numpy as np
np.random.seed(100)

def process(config, exp, fold, dev='cpu', skip_train = False):
    print("Experiment num = {0}\nFold num = {1}".format(exp, fold))
    # create your data generator
    config.num_fold = fold
    data = DataGenerator(config)
    # create an instance of the model you want
    model = invariant_basic(config, data, device=dev)
    # create trainer and pass all the previous components to it
    trainer = Trainer(model, data, config, dev=dev)
    # here you train your model
    acc, loss = trainer.train(skip_train = skip_train)
    acc = []
    loss = []
    doc_utils.doc_results(acc, loss, exp, fold, config.summary_dir)

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        config = process_config2(config, args)

    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    print(f'CUDA: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    print("lr = {0}".format(config.learning_rate))
    print("decay = {0}".format(config.decay_rate))
    print(config.architecture)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    pprint(config)
    banner()
    for exp in range(1, config.num_exp + 1):
        for fold in range(1):
            process(config, exp, fold, dev=args.dev, skip_train = True) # todo: somehow this is not working

    print(config.summary_dir)

    # commented out
    # doc_utils.summary_10fold_results(config.summary_dir)


if __name__ == '__main__':
    # fix_seed()
    main()
