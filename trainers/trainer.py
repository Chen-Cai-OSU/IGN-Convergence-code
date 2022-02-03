import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from signor.format.format import red
from signor.utils.np import tonp

from tqdm import tqdm
import numpy as np

import sys
sys.path.append('../utils/')
import doc_utils

from signor.monitor.time import timefunc
from signor.ml.pytorch.model import num_trainable_params
from signor.monitor.probe import summary
from signor.configs.util import subset_dict, dict2name
from collections import defaultdict


class Trainer():
    @timefunc
    def __init__(self, model, data_loader, config, dev='cpu'):
        super(Trainer, self).__init__()
        # Assign all class attributes
        print(model)
        print(red('before model placement'))
        self.model = model.to(dev)
        print(red('after model placement'))
        self.dev = dev
        if data_loader is not None:
            self.data_loader = data_loader
        self.config = config

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr = self.config.learning_rate)

    def train(self, skip_train = False):
        if skip_train:
            return self.test(1)

        for cur_epoch in range(self.config.num_epochs):
            # train epoch
            train_acc, train_loss = self.train_epoch(cur_epoch)

            # validation step
            if self.config.val_exist:
                test_acc, test_loss = self.test(cur_epoch)
                # document results
                doc_utils.write_to_file_doc(train_acc, train_loss, test_acc, test_loss, cur_epoch, self.config)
        if self.config.val_exist:
            # creates plots for accuracy and loss during training
            doc_utils.create_experiment_results_plot(self.config.exp_name, "accuracy", self.config.summary_dir)
            doc_utils.create_experiment_results_plot(self.config.exp_name, "loss", self.config.summary_dir, log=True)
        return self.test(cur_epoch)

    def train_epoch(self, num_epoch = None):
        """
        implement the logic of epoch:
        -loop on the number of iterations in the config and call the train step

        Train one epoch
        :param epoch: cur epoch number
        :return accuracy and loss on train set
        """
        # initialize dataset
        self.data_loader.initialize(is_train = True)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="epoch-{}-".format(num_epoch))

        total_loss = 0.
        total_correct = 0.

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, correct = self.train_step()
            # update results from train_step func
            total_loss += loss
            total_correct += correct

        # save model
        """
        if num_epoch % self.config.save_rate == 0:
            print("Save model")
            #self.model.save(self.sess)
        """

        loss_per_epoch = total_loss/self.data_loader.train_size
        acc_per_epoch = total_correct/self.data_loader.train_size
        print("""
        Epoch-{}  loss:{:.4f} -- train-acc:{:.4f}
                """.format(num_epoch, loss_per_epoch, acc_per_epoch))

        tt.close()
        return acc_per_epoch, loss_per_epoch

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - :return any accuracy and loss on current batch
       """

        graphs, labels = self.data_loader.next_batch()
        # graphs, labels = graphs.to(self.dev), labels.to(self.dev)
        t0 = time.time()
        # print(f'before put to {self.dev}')
        output = self.model(torch.tensor(graphs, dtype = torch.float64).to(self.dev))
        # print(f'after put to {self.dev}. Took {int(time.time()-t0)}s')
        predict = output.argmax(dim = 1, keepdim = True)
        target = torch.from_numpy(np.array(labels)).to(torch.long).to(self.dev)
        loss = F.nll_loss(output, target)
        correct = predict.eq(target.view_as(predict)).sum().item()
        loss.backward()
        self.optimizer.step()     

        return loss, correct

    @timefunc
    def test(self, epoch):
        # initialize dataset
        self.data_loader.new_initialize()
        num_trainable_params(self.model, verbose=True)
        # self.data_loader.initialize(is_train=False)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.val_size), total=self.data_loader.val_size,
                  desc="Val-{}-".format(epoch))


        total_loss = 0.
        total_correct = 0.
        ret = defaultdict()

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            graph, label = self.data_loader.next_batch()
            label = np.expand_dims(label, 0)

            with torch.no_grad():
                assert graph.shape[3] == graph.shape[2]
                output = self.model(torch.tensor(graph, dtype = torch.float64, device=self.dev))
                k, v = graph.shape[3], tonp(output).flatten().tolist()
                ret[k] = v
                # print(f'Graph of size {graph.shape[3]}. Output: {output}')
                loss, correct = -1, -1
                continue

                # summary(output, 'output')
                predict = output.argmax(dim = 1, keepdim = True)
                target = torch.from_numpy(np.array(label)).to(torch.long).to(self.dev)
                loss = F.nll_loss(output, target)
                correct = predict.eq(target.view_as(predict)).sum().item()

            # update metrics returned from train_step func
            total_loss += loss
            total_correct += correct

        _d = subset_dict(dict(self.config)).include(['n_graph', 'name', 'sample_method', 'estimate', 'seed'])
        json_file = dict2name(_d, remove_underscore=True) + '.json'
        json_file = f'./result/{json_file}'
        print(f'save at {red(json_file)}')
        with open(json_file, 'w') as fp:
            json.dump(ret, fp)
        test_loss = total_loss/self.data_loader.val_size
        test_acc = total_correct/self.data_loader.val_size

        print("""
        Val-{}  loss:{:.4f} -- acc:{:.4f}
        """.format(epoch, test_loss, test_acc))

        tt.close()
        return test_acc, test_loss
