import numpy as np

import optim
from coco_utils import sample_coco_minibatch


class CaptioningSolver(object):
 

  def __init__(self, model, data, **kwargs):
 
    self.model = model
    self.data = data

    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.lr_decay = kwargs.pop('lr_decay', 1.0)
    self.batch_size = kwargs.pop('batch_size', 100)
    self.num_epochs = kwargs.pop('num_epochs', 10)

    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())
      raise ValueError('Unrecognized arguments %s' % extra)

    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)

    self._reset()


  def _reset(self):
 
    self.epoch = 0
    self.best_val_acc = 0
    self.best_params = {}
    self.loss_history = []
    self.train_acc_history = []
    self.val_acc_history = []

    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.items()}
      self.optim_configs[p] = d


  def _step(self):
 
    # minibatch = sample_coco_minibatch(self.data,
    #               batch_size=self.batch_size,
    #               split='train')
    # captions, features, urls = minibatch

    mask = np.random.randint(len(self.data)-self.batch_size)
    captions = self.data[mask:mask+self.batch_size]
    captions = np.array(captions).reshape(1, -1)

    # Compute loss and gradient
    # loss, grads = self.model.loss(features, captions)

    loss, grads = self.model.loss(captions)
    self.loss_history.append(loss)

    # Perform a parameter update
    for p, w in self.model.params.items():
      dw = grads[p]
      config = self.optim_configs[p]
      next_w, next_config = self.update_rule(w, dw, config)
      self.model.params[p] = next_w
      self.optim_configs[p] = next_config

  
 
  def check_accuracy(self, X, y, num_samples=None, batch_size=100):
 
    # return 0.0
    
    N = X.shape[0]
    if num_samples is not None and N > num_samples:
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

    num_batches = int(N / batch_size)
    if N % batch_size != 0:
      num_batches += 1
    y_pred = []
    for i in range(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = self.model.loss(X[start:end])
      y_pred.append(np.argmax(scores, axis=1))
    y_pred = np.hstack(y_pred)
    acc = np.mean(y_pred == y)

    return acc


  def train(self):
 
    # num_train = self.data['train_captions].shape[0]

    num_train = len(self.data)
    iterations_per_epoch = max(int(num_train / self.batch_size), 1)
    num_iterations = self.num_epochs * iterations_per_epoch

    for t in range(num_iterations):
      self._step()

      if self.verbose and t % self.print_every == 0:
        print ('(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1]))


      epoch_end = (t + 1) % iterations_per_epoch == 0

      if epoch_end:
        self.epoch += 1
        for k in self.optim_configs:
          self.optim_configs[k]['learning_rate'] *= self.lr_decay

      # first_it = (t == 0)
      # epoch_end = (t + 1) % iterations_per_epoch == 0
      #
      # if first_it or epoch_end or t % self.print_every == 0:
      #   if epoch_end:
      #     self.epoch += 1
      #     for k in self.optim_configs:
      #       self.optim_configs[k]['learning_rate'] *= self.lr_decay
      #
      #   train_acc = self.check_accuracy(self.data['train_features'],
      #                                   self.data['train_captions'], num_samples=self.batch_size)
      #   self.train_acc_history.append(train_acc)
      #
      #   val_acc = self.check_accuracy(self.data['val_features'],
      #                                 self.data['val_captions'], num_samples=self.batch_size)
      #   self.val_acc_history.append(val_acc)
      #
      #   if val_acc > self.best_val_acc:
      #     self.best_val_acc = val_acc
      #     for p in self.model.params:
      #       self.best_params[p] = self.model.params[p].copy()
      #
      #   if self.verbose:
      #     print('Finished epoch %d / %d: cost %f, train_acc: %f, val_acc: %f, lr: %e'
      #           %(self.epoch, self.num_epochs, self.loss_history[-1], train_acc, val_acc,
      #             self.optim_config['learning_rate']))

    # self.model.params = self.best_params

