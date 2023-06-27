import numpy as np
import time

try:
    from utils import RESULT
except:
    from hebb.utils import RESULT
import torch
import os
import os.path as op


class LogSupBatch:
    def __init__(self):
        self.data = []
        self.metric_dict = {0: 'n', 1: 'train_loss', 2: 'train_acc', 3: 'convergence', 4: 'R1', 5: 'lr'}

    def step(self, n, train_loss, train_acc, convergence=0, R1=0, lr=1):
        self.data.append([n, train_loss, train_acc, convergence, R1, lr])

    def get_summary(self):
        if self.data:
            data_np = self.get_numpy()
            nb_samle = data_np[:, 0].sum()
            mean_loss = data_np[:, 1].sum() / nb_samle
            mean_acc = data_np[:, 2].sum() / nb_samle
            return mean_loss, 100 * mean_acc
        else:
            return 0, 0

    def get_numpy(self):
        return np.array(self.data, dtype=np.float32)

    def reset(self):
        self.__init__()

    def to_dict(self):
        return {'data': self.data}

    def from_dict(self, dict_):
        self.data = dict_['data']
        return self


class LogSup:
    def __init__(self, config):
        self.metric_dict = {0: 'train_loss', 1: 'train_acc', 2: 'test_loss', 3: 'test_acc', 4: 'convergence', 5: 'R1'}
        self.metric = ''
        self.mode = ''
        self.batch = []
        self.initial_start = time.time()
        self.start = self.initial_start
        self.convergence = None
        self.epoch_time = 0
        if config is not None:
            self.epcoh = 0
            self.lr = config['lr']
            self.nb_epoch = config['nb_epoch']
            self.print_freq = config['print_freq']
            self.data = []
            self.metric = 'test_acc'
            self.metric_id = {value: key for key, value in self.metric_dict.items()}[self.metric]
            self.mode = 'min' if self.metric.endswith('loss') else 'max'
            self.best_perf = 0 if self.mode == 'max' else 100
            self.perf = self.best_perf
            self.is_best = True

    def step(self, epoch, logbatch, test_loss, test_acc, lr, save=False):

        train_loss, train_acc = logbatch.get_summary()
        self.lr = float(lr)
        self.data.append([int(epoch), float(train_loss), float(train_acc), float(test_loss), float(test_acc)])
        self.perf = self.data[-1][self.metric_id]
        self.is_best = self.perf > self.best_perf if self.mode == 'max' else self.perf < self.best_perf
        self.best_perf = max(self.perf, self.best_perf) if self.mode == 'max' else min(self.perf, self.best_perf)
        if save:
            self.batch.append(logbatch)
        self.epoch_time = time.time() - self.start
        self.start = time.time()
        return self.new_log_batch()

    def new_log_batch(self):
        return LogSupBatch()

    def verbose(self):
        epoch, train_loss, train_acc, test_loss, test_acc = self.data[-1]

        print('Epoch: [{0}/{1}]\t'
              'lr: {lr:.2e}\t'
              'time: {total_time}\t'
              'Loss_train {train_loss:.5f}\t'
              'Acc_train {train_acc:.2f}\t/\t'
              'Loss_test {test_loss:.5f}\t'
              'Acc_test {test_acc:.2f}'
              .format(epoch, self.nb_epoch, lr=self.lr, time=self.epoch_time,
                      total_time=time.strftime("%H:%M:%S", time.gmtime(time.time() - self.initial_start)),
                      train_acc=train_acc, train_loss=train_loss,
                      test_loss=test_loss, test_acc=test_acc))

    def get_numpy(self):
        return np.array(self.data, dtype=np.float32)

    def to_dict(self):
        return {'data': self.data,
                'metric': self.metric,
                'best_perf': self.best_perf,
                'mode': self.mode,
                'batch': [b.to_dict() for b in self.batch]
                }

    def from_dict(self, dict_):
        self.data = dict_['data']
        self.batch = [LogSupBatch().from_dict(d) for d in dict_['batch']]
        self.batch.append(LogSupBatch())
        self.best_perf = dict_['best_perf']
        if dict_['metric'] != self.metric:
            if self.mode == 'max':
                self.best_perf = self.get_numpy()[:, self.metric_id].max()
            elif self.mode == 'min':
                self.best_perf = self.get_numpy()[:, self.metric_id].min()
        return self


class LogUnsup(LogSup):
    def __init__(self, config):
        super().__init__(config)
        self.metric_dict = {0: 'train_acc', 1: 'test_acc', 2: 'convergence', 3: 'R1'}
        self.nb_epoch = config['nb_epoch'] if config is not None else 0
        self.metric = 'test_acc'
        self.metric_id = 1
        self.mode = 'max'
        self.best_perf = 100
        self.perf = self.best_perf
        self.is_best = True
        self.info = ''

    def step(self, epoch, train_acc, test_acc, info, convergence, R1, lr):
        self.lr = float(lr)
        self.data.append([int(epoch), float(train_acc), float(test_acc), float(convergence), int(R1)])
        self.info = info
        self.perf = self.data[-1][self.metric_id]
        self.is_best = self.perf > self.best_perf if self.mode == 'max' else self.perf < self.best_perf
        self.best_perf = max(self.perf, self.best_perf) if self.mode == 'max' else min(self.perf, self.best_perf)
        self.epoch_time = time.time() - self.start
        self.start = time.time()

    def verbose(self):
        epoch, train_acc, test_acc, convergence, R1 = self.data[-1]
        print('Epoch: [{0}/{1}]\t'
              'lr: {lr:.2e}\t'
              'time: {total_time}\t'
              'Acc_train {train_acc:.2f}\t'
              'Acc_test {test_acc:.2f}\t'
              'convergence: {convergence:.2e}\t'
              'R1: {R1}\t'
              'Info {info}'
              .format(epoch, self.nb_epoch, lr=self.lr, time=self.epoch_time,
                      total_time=time.strftime("%H:%M:%S", time.gmtime(time.time() - self.initial_start)),
                      train_acc=train_acc, convergence=convergence, R1=R1, info=self.info, test_acc=test_acc))

    def get_numpy(self):
        return np.array(self.data, dtype=np.float32)


class Log:
    def __init__(self, configs={}):
        self.sup = {}
        self.unsup = {}
        for id, config in configs.items():
            if config['mode'] == 'unsupervised':
                self.unsup[id] = LogUnsup(config)
            else:
                self.sup[id] = LogSup(config)

    def to_dict(self):
        return {'sup': {id: sup.to_dict() for id, sup in self.sup.items()},
                'unsup': {id: unsup.to_dict() for id, unsup in self.unsup.items()}}

    def from_dict(self, dict_):
        self.sup = {}
        self.unsup = {}
        for id, config in dict_['sup'].items():
            self.sup[id] = LogSup(None).from_dict(config)
        for id, config in dict_['unsup'].items():
            self.unsup[id] = LogUnsup(None).from_dict(config)
        return self


def save_logs(log, model_name, filename='final.pth.tar'):
    folder_path = op.join(RESULT, 'network', model_name, 'measures')
    if not op.isdir(folder_path):
        os.mkdir(folder_path)

    torch.save({
        'log': log.to_dict()
    }, op.join(folder_path, filename))


def load_logs(model_name, filename='final.pth.tar'):
    folder_path = op.join(RESULT, 'network', model_name, 'measures')
    if not op.isdir(folder_path):
        os.mkdir(folder_path)
    dict = torch.load(op.join(folder_path, filename))['log']
    log = Log().from_dict(dict)
    return dict, log
