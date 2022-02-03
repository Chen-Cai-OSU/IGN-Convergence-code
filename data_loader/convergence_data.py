# generate dataset for convergence IGN paper
from bisect import bisect

from signor.utils.random_ import fix_seed

fix_seed()

import json
import os
from matplotlib import pyplot as plt

from signor.ioio.dir import cur_dir

from functools import partial

import networkx as nx
import numpy as np
from signor.format.format import red
from signor.ml.Esme import graphonlib
from signor.monitor.probe import summary
from signor.monitor.time import timefunc
from signor.viz.matrix import matshow
from tqdm import tqdm
import os.path as osp
import argparse
import sys
sys.path.append('../main_scripts')
from global_var import graph_dict

DIR = os.path.join(eval(cur_dir()), '..', 'main_scripts','result')
tf = partial(timefunc, threshold=-1)


class RandomSampleGenerator:
    def __init__(self, n_seeds=10):
        self.n_seeds = n_seeds
        self.f = osp.join(eval(cur_dir()), '..', 'main_scripts', 'random_sample.json')
        if not os.path.isfile(self.f):
            self.generate()
        self.d = None

    def generate(self):
        ret = dict()
        for n in tqdm(range(50, 6000, 75), desc='Generating random sample'):
            for seed in range(10):
                k = (n, seed)
                v = np.random.random_sample(n)
                v.sort()
                ret[k] = v.tolist()
        with open(self.f, 'w') as fp:
            ret = {str(k): v for k, v in ret.items()}
            json.dump(ret, fp)

    def load(self, n, seed):
        if self.d is None:
            with open(self.f, 'r') as fp:
                self.d = json.load(fp)
        k = (n, seed)
        return np.array(self.d[str(k)])


class DataSet:
    def __init__(self):
        self.w = None
        self.x = None
        self.set_x()
        self.isconstant = None
        self.rsg = RandomSampleGenerator(n_seeds=10)

    @timefunc
    def getW(self, n=10):
        assert self.w is not None
        w = np.zeros((n, n))
        if self.isconstant:
            w += self.w(0, 0)
        else:
            for i in range(n):
                for j in range(n):
                    w[i][j] = self.w(i / n, j / n)
        np.fill_diagonal(w, 0)
        return w

    def getPWW(self, n_piece):
        # get piece wise induced graphon
        # deterministic sampling; used for slides preparation
        n = 1000
        step = n // n_piece
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                tmpi, tmpj = (i // step) * (1/n_piece), (j // step) * (1/n_piece)
                if tmpi != tmpj:
                    w[i][j] = self.w(tmpi, tmpj)
                else:
                    w[i][j] = 0
        return w

    def getPWW_random(self, n_piece):
        # get piece wise induced graphon
        # random sampling; used for slides preparation
        def grade(score, breakpoints=[100, 300, 400, 600, 900, 1000], grades=[0, 100, 300, 400, 600, 900, 1000]):
            i = bisect(breakpoints, score)
            return grades[i]

        n = 1000
        step = n // n_piece
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                tmpi, tmpj = grade(i)/n, grade(j)/n # (i // step) * (1/n_piece), (j // step) * (1/n_piece)
                if tmpi != tmpj:
                    w[i][j] = self.w(tmpi, tmpj)
                else:
                    w[i][j] = 0
        return w


    ### set X ###
    def set_x(self):
        self.x = lambda u: 1 #  u ** 2 TODO: change later

    ### different W ###
    def set_er(self, p=0.1):
        self.w = lambda u, v: p + (u + v) * 0  # make it esay to vectorize
        self.isconstant = True

    def set_sbm(self, n_blocks=2, probability=[[0.1, 0.25], [0.25, 0.4]]):
        # todo: more generic implementation
        self.w = lambda u, v: (u > 0.5) * 0.15 + (v > 0.5) * 0.15 + 0.1
        # def func(u, v):
        #     if u <= 0.5 and v <= 0.5:
        #         return probability[0][0]
        #     elif (u <= 0.5 and v >= 0.5) or (u >= 0.5 and v <= 0.5):
        #         assert probability[0][1] == probability[1][0]
        #         return probability[0][1]
        #     elif u > 0.5 and v > 0.5:
        #         return probability[1][1]
        #     else:
        #         print(red('Unexpected'))
        #         return 0.1
        # self.w = func

    def set_graphon(self):
        # general graphon
        self.w = lambda u, v: (u + v) / 4 + 1 / 4

    def set_PW_graphon(self, n_piece=2):
        l = 1.0/n_piece
        self.w = lambda u, v: (u%l + v%l) / 4 + 1 / 4

    ### sample and estimations ###
    @timefunc
    def estimate_probability(self, adj, h=0.3):
        # given a 0-1 matrix adj, estimate edge probablity of W
        assert set(np.unique(adj)) == set([0, 1]), f'{len(set(np.unique(adj)))} different values'
        p_zhang = graphonlib.smoothing.zhang.smoother(adj, h=h)
        return p_zhang

    @timefunc
    def _fillw(self, it, n):
        try:
            return self._fillw2(it, n)
        except:
            print(red('revert back to old implementation'))
            ret = np.zeros((n, n))
            for i, u in enumerate(it):
                for j, v in enumerate(it):
                    ret[i][j] = self.w(u, v)
            np.fill_diagonal(ret, 0)
        return ret

    def _fillw2(self, it, n):
        assert len(it) == n
        xv, yv = np.meshgrid(it, it)
        ret = self.w(xv, yv)
        np.fill_diagonal(ret, 0)
        return ret

    @timefunc
    def _fillw_probability(self, it, n):
        ret = self._fillw(it, n)
        # summary(ret, 'prob matrix')
        ret = np.random.binomial(1, ret, size=(n, n))
        # summary(ret, 'sample 0-1 matrix')
        # matshow(ret)
        # exit()
        return ret

    def _fillx(self, it, n):
        ret = np.zeros(n)
        for i, u in enumerate(it):
            ret[i] = self.x(u)
        return np.diag(ret)

    @timefunc
    def grid_sample(self, n):
        # deterministic sample using a nxn grid
        it = np.linspace(0, 1, n)
        return self._fillw(it, n), self._fillx(it, n)

    # @jit(nopython=True)
    @timefunc
    def random_sample(self, n, seed=1):
        # similar to grid_sample but just randomly sample over [0,1]
        # it = np.random.random_sample(n)
        # it.sort()
        it = self.rsg.load(n, seed)
        return self._fillw(it, n), self._fillx(it, n)

    def _assert_adj(self, adj):
        unique = np.unique(adj)
        assert set(unique) == {0, 1}, f'Expect 0-1 matrix but got matrix of {len(unique)} unique values'

    # @jit(nopython=True)
    @timefunc
    def grid_Gsample(self, n, estimate=False):
        # sample a 0-1 matrix according to edge probablity (weights)
        it = np.linspace(0, 1, n)
        ret = self._fillw_probability(it, n)
        self._assert_adj(ret)
        ret = self.estimate_probability(ret) if estimate else ret
        x = self._fillx(it, n)
        return ret, x

    @timefunc
    def random_Gsample(self, n, estimate=False, seed=1):
        # sample a 0-1 matrix according to edge probablity (weights)
        # it = np.random.random_sample(n)
        # it.sort()
        it = self.rsg.load(n, seed)
        ret = self._fillw_probability(it, n)
        self._assert_adj(ret)
        ret = self.estimate_probability(ret) if estimate else ret
        x = self._fillx(it, n)
        return ret, x

        ### Generate graphs ###

    def _set_generative_model(self, name='er0.05'):
        if name == 'er0.05':
            self.set_er(0.05)
        elif name == 'er0.01':
            self.set_er(0.01)
        elif name == 'er0.1':
            self.set_er(0.1)
        elif name == 'sbm':
            self.set_sbm()
        elif name == 'graphon':
            self.set_graphon()
        elif name == 'graphonPW2':
            self.set_PW_graphon(n_piece=2)
        elif name == 'graphonPW3':
            self.set_PW_graphon(n_piece=3)
        else:
            raise NotImplementedError
        assert self.w is not None

    def _sample(self, n, method='random_sample', estimate=False, seed=1):
        if method == 'grid_sample':
            assert estimate == False, 'estimate has to be False'
            g, x = self.grid_sample(n)
        elif method == 'random_sample':
            assert estimate == False, 'estimate has to be False'
            g, x = self.random_sample(n, seed=seed)
        elif method == 'grid_Gsample':
            g, x = self.grid_Gsample(n, estimate=estimate)
        elif method == 'random_Gsample':
            g, x = self.random_Gsample(n, estimate=estimate, seed=seed)
        else:
            raise NotImplementedError
        g = np.stack([g, x], axis=2)
        return g

    def generate_graphs(self, n_graph=10, name='er0.05',
                        sample_method='grid_Gsample',
                        estimate=False,
                        seed=-1):
        print(red(n_graph, name, sample_method, estimate))
        # exit()
        assert n_graph % 2 == 0, 'Use even number n_graph'
        graphs = []
        self._set_generative_model(name=name)

        for i in tqdm(range(n_graph // 2), desc=f'Generate {name} graphs'):  # tqdm(range(5, (n_graph+1)*5, 5)):
            n = 50 + i * 75  # 100 + i * 200
            g = self._sample(n, method=sample_method, estimate=estimate, seed=seed)
            for _ in range(2):  # somehow has to add this; otherwise always encouter some weired np.array reshape error.
                graphs.append(g)

        print(red(f'max node number is {n}'))
        g0 = graphs[0]
        if n_graph < 188:
            graphs += [g0] * (188 - n_graph)
        labels = np.random.randint(2, size=188).tolist()

        # followed from data_helper.
        graphs = np.array(graphs)
        for i in range(graphs.shape[0]):
            graphs[i] = np.transpose(graphs[i], [2, 0, 1])
        return np.array(graphs), np.array(labels)


def ER(n_graph=188):
    graphs, labels = [], []
    p = 0.05
    for i in tqdm(range(n_graph // 2), desc='Generate ER graphs'):  # tqdm(range(5, (n_graph+1)*5, 5)):
        for _ in range(2):  # somehow has to add this; otherwise always encouter some weired np.array reshape error.
            # n = np.random.randint(20, 300)  #
            n = 100 + i * 10
            g = nx.erdos_renyi_graph(n, p)
            g = nx.convert_matrix.to_numpy_matrix(g)
            g = np.array(g) * p
            g = np.stack([g] * 1, axis=2)
            graphs.append(g)
            label = np.random.randint(2, size=1)[0]
            labels.append(label)

    # followed from data_helper.
    graphs = np.array(graphs)
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1])
    return np.array(graphs), np.array(labels)


parser = argparse.ArgumentParser(description='Data Generation')
parser.add_argument('--n', type=int, default=50, help='number of nodes')
parser.add_argument('--h', type=float, default=0.3, help='number of nodes')
parser.add_argument('--name', type=str, default='er0.1', help='graph names')
parser.add_argument('--save', action='store_true', help='Save plot')

if __name__ == '__main__':
    args = parser.parse_args()
    n = args.n
    name = args.name
    ds = DataSet()
    # ds.generate_graphs(2)
    # exit()
    p = 0.1

    if name == 'er0.1':
        ds.set_er(p)
    elif name == 'sbm':
        ds.set_sbm()
    elif name == 'graphon':
        ds.set_graphon()
    elif name == 'graphonPW2':
        ds.set_PW_graphon(n_piece=2)
    elif name == 'graphonPW3':
        ds.set_PW_graphon(n_piece=3)
    else:
        raise NotImplementedError
    # adj, _ = ds.random_sample(n, seed=2)
    adj, _ = ds.grid_Gsample(n)
    print(adj)
    w = ds.getW(n=1000)
    matshow(w, var_name='w')
    pww_random = ds.getPWW_random(n_piece=10)
    matshow(pww_random, var_name='pww_random')
    exit()
    pww = ds.getPWW(n_piece=10)
    matshow(pww, var_name='pww')
    # summary(adj, 'adj')

    edge_prob = ds.estimate_probability(adj, h=args.h)
    abs_diff = abs(edge_prob - ds.getW(n=n))
    print(f'norm diff (normalized) is {np.linalg.norm(abs_diff) / n}')
    diff = abs_diff / ds.getW(n=n)
    np.fill_diagonal(diff, 0)
    # matshow(100 * diff, 'diff-percent')
    summary(100 * diff, 'diff-percent')
    # matshow(adj)
    # matshow( ds.getW(n=n), var_name='P')
    # exit()
    if args.save:
        plt.imshow(ds.getW(n=n))
        plt.colorbar()
        plt.title(f'{graph_dict[args.name]}')

        name_ = name if 'er' not in name else 'er'
        fname = os.path.join(DIR, '..', 'fig', f'{name_}.pdf')
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.show()
        print(f'save fig at {fname}')

    rsg = RandomSampleGenerator()
    print(rsg.f)
    rsg.generate()
    print(rsg.load(50, 2))
    exit()
