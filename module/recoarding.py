
#记录训练过程中的信息
import numpy as np
import time
import os
import logging
from datetime import datetime
from subprocess import call
from types import ModuleType


def getTime():
    return datetime.now().strftime('%m-%d %H:%M:%S')


class Timer(object):
    curr_record = None
    prev_record = None

    @classmethod
    def record(cls):
        cls.prev_record = cls.curr_record
        cls.curr_record = time.time()

    @classmethod
    def interval(cls):
        if cls.prev_record is None:
            return 0
        return cls.curr_record - cls.prev_record


def wrapColor(string, color):
    try:
        header = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'darkcyan': '\033[36m',
            'bold': '\033[1m',
            'underline': '\033[4m'}[color.lower()]
    except KeyError:
        raise ValueError("Unknown color: {}".format(color))
    return header + string + '\033[0m'


def info(logger, msg, color=None):
    msg = '[{}]'.format(getTime()) + msg
    if logger is not None:
        logger.info(msg)

    if color is not None:
        msg = wrapColor(msg, color)
    print(msg)


def summaryArgs(logger, args, color=None):
    if isinstance(args, ModuleType):
        args = vars(args)
    keys = [key for key in args.keys() if key[:2] != '__']
    keys.sort()
    length = max([len(x) for x in keys])
    msg = [('{:<' + str(length) + '}: {}').format(k, args[k]) for k in keys]

    msg = '\n' + '\n'.join(msg)
    info(logger, msg, color)


class SaveParams(object):
    def __init__(self, model, snapshot, model_name, num_save=5):
        self.model = model
        self.snapshot = snapshot
        self.model_name = model_name
        self.num_save = num_save
        self.save_params = []

    def save(self, n_epoch):
        self.save_params += [
            os.path.join(self.snapshot, '{}-{:04d}.params'.format(self.model_name, n_epoch)),
            os.path.join(self.snapshot, '{}-{:04d}.states'.format(self.model_name, n_epoch))]
        self.model.save_params(self.save_params[-2])
        self.model.save_optimizer_states(self.save_params[-1])

        if len(self.save_params) > 2 * self.num_save:
            call(['rm', self.save_params[0], self.save_params[1]])
            self.save_params = self.save_params[2:]
        return self.save_params[-2:]

    def __call__(self, n_epoch):
        return self.save(n_epoch)


def getLogger(snapshot, model_name):
    if not os.path.exists(snapshot):
        os.makedirs(snapshot)
    logging.basicConfig(filename=os.path.join(snapshot, model_name + '.log'), level=logging.INFO)
    logger = logging.getLogger()
    return logger


class LrScheduler(object):
    def __init__(self, method, init_lr, kwargs):
        self.method = method
        self.init_lr = init_lr

        if method == 'step':
            self.step_list = kwargs['step_list']
            self.factor = kwargs['factor']
            self.get = self._step
        elif method == 'poly':
            self.num_epoch = kwargs['num_epoch']
            self.power = kwargs['power']
            self.get = self._poly
        elif method == 'ramp':
            self.ramp_up = kwargs['ramp_up']
            self.ramp_down = kwargs['ramp_down']
            self.num_epoch = kwargs['num_epoch']
            self.scale = kwargs['scale']
            self.get = self._ramp
        else:
            raise ValueError(method)

    def _step(self, current_epoch):
        lr = self.init_lr
        step_list = [x for x in self.step_list]
        while len(step_list) > 0 and current_epoch >= step_list[0]:
            lr *= self.factor
            del step_list[0]
        return lr

    def _poly(self, current_epoch):
        lr = self.init_lr * ((1. - float(current_epoch) / self.num_epoch) ** self.power)
        return lr

    def _ramp(self, current_epoch):
        if current_epoch < self.ramp_up:
            decay = np.exp(-(1 - float(current_epoch) / self.ramp_up) ** 2 * self.scale)
        elif current_epoch > (self.num_epoch - self.ramp_down):
            decay = np.exp(-(float(current_epoch + self.ramp_down - self.num_epoch) / self.ramp_down) ** 2 * self.scale)
        else:
            decay = 1.
        lr = self.init_lr * decay
        return lr


class GradBuffer(object):
    def __init__(self, model):
        self.model = model
        self.cache = None

    def write(self):
        if self.cache is None:
            self.cache = [[None if g is None else g.copyto(g.context) for g in g_list] \
                          for g_list in self.model._exec_group.grad_arrays]
        else:
            for gs_src, gs_dst in zip(self.model._exec_group.grad_arrays, self.cache):
                for g_src, g_dst in zip(gs_src, gs_dst):
                    if g_src is None:
                        continue
                    g_src.copyto(g_dst)

    def read_add(self):
        assert self.cache is not None
        for gs_src, gs_dst in zip(self.model._exec_group.grad_arrays, self.cache):
            for g_src, g_dst in zip(gs_src, gs_dst):
                if g_src is None:
                    continue
                g_src += g_dst


# if __name__ == '__main__':
#     import argparse
#
#     logger = getLogger('E:\\xwf\\CIAN-master\\cccc', 'resnet101_largefov')  # 第一个参数为目录，第二个参数为文件名      创建log
#     summaryArgs(logger, vars('args'), 'green')
#     info(logger, "Learning rate: {}".format(3), 'yellow')
#     Timer.record()
#     print(1)


