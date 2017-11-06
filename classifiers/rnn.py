import numpy as np

from layers import *
from rnn_layers import *


class CaptioningRNN(object):

  def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
               hidden_dim=128, cell_type='rnn', dtype=np.float32, optim=False):

    if cell_type not in {'rnn', 'lstm'}:
      raise ValueError('Invalid cell_type "%s"' % cell_type)
    
    self.cell_type = cell_type
    self.dtype = dtype
    self.word_to_idx = word_to_idx
    self.idx_to_word = {i: w for w, i in word_to_idx.items()}
    self.params = {}

    self.optim = optim
    
    vocab_size = len(word_to_idx)   # V

    self._null = word_to_idx['<NULL>']
    self._start = word_to_idx.get('<START>', None)
    self._end = word_to_idx.get('<END>', None)
    
    # Initialize word vectors
    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)    #  V by W
    self.params['W_embed'] /= 100
    
    # Initialize CNN -> hidden state projection parameters
    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)    # D by H
    self.params['W_proj'] /= np.sqrt(input_dim)
    self.params['b_proj'] = np.zeros(hidden_dim)

    # Initialize parameters for the RNN
    dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
    self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)    # W by H(4H)
    self.params['Wx'] /= np.sqrt(wordvec_dim)
    self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)    # W by H(4H)
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(dim_mul * hidden_dim)
    
    # Initialize output to vocab weights
    self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)    # H by V
    self.params['W_vocab'] /= np.sqrt(hidden_dim)
    self.params['b_vocab'] = np.zeros(vocab_size)
      
    # Cast parameters to correct dtype
    for k, v in self.params.items():
      self.params[k] = v.astype(self.dtype)


  def loss(self, features, captions):

    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    
    mask = (captions_out != self._null)

    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

    W_embed = self.params['W_embed']

    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    loss, grads = 0.0, {}

    h0 = np.dot(features, W_proj) + b_proj  # Initial hidden state (N, H)
    x, cache0 = word_embedding_forward(captions_in, W_embed)  # transform the words in captions_in from indices to vectors  (N, T, W)
    
    if self.cell_type == 'rnn':
      if self.optim:
        h, cache1 = rnn_forward_caption(x, h0, Wx, Wh, b)
      else:
        h, cache1 = rnn_forward(x, h0, Wx, Wh, b)  # process the sequence of input word vectors and produce hidden state vectors for all timesteps (N, T, H)
    else:
      if self.optim:
        h, cache1 = lstm_forward_caption(x, h0, Wx, Wh, b)
      else:
        h, cache1 = lstm_forward(x, h0, Wx, Wh, b)
    
    out, cache2 = temporal_affine_forward(h, W_vocab, b_vocab)  # compute scores over the vocabulary at every timestop using the hidden states (N, T, V)
    loss, dout = temporal_softmax_loss(out, captions_out, mask)

    dh, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dout, cache2)
    
    if self.cell_type == 'rnn':
      if self.optim:
        dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward_caption(dh, cache1)
      else:
        dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dh, cache1)
    else:
      if self.optim:
        dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward_caption(dh, cache1)
      else:
        dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dh, cache1)
    
    grads['W_embed'] = word_embedding_backward(dx, cache0)
    grads['b_proj'] = np.sum(dh0, axis=0)
    grads['W_proj'] = np.dot(features.T, dh0)
    
    return loss, grads


  def sample(self, features, max_length=30):

    N = features.shape[0]
    captions = self._null * np.ones((N, max_length), dtype=np.int32)

    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

    h0 = np.dot(features, W_proj) + b_proj

    next_h = h0
    next_c = np.zeros_like(h0)
    x = np.array([W_embed[self._start],] * N)

    for i in range(max_length):
        if self.cell_type == 'rnn':
            next_h, cache = rnn_step_forward(x, next_h, Wx, Wh, b)
        else:
            next_h, next_c, cache = lstm_step_forward(x, next_h, next_c, Wx, Wh, b)
        score = np.dot(next_h, W_vocab) + b_vocab
        captions[:, i] = np.argmax(score, axis=1)
        x = W_embed[captions[:, i]]

    return captions


  def sample_caption(self, features, max_length=30):

    N = features.shape[0]
    captions = self._null * np.ones((N, max_length), dtype=np.int32)

    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

    h0 = np.dot(features, W_proj) + b_proj

    next_h = np.zeros_like(h0)
    next_c = np.zeros_like(h0)
    x = np.array([W_embed[self._start],] * N)

    for i in range(max_length):
      if self.cell_type == 'rnn':
        next_h, cache = rnn_step_forward(x, next_h + h0, Wx, Wh, b)
      else:
        next_h, next_c, cache = lstm_step_forward(x, next_h + h0, next_c, Wx, Wh, b)
      scores = np.dot(next_h, W_vocab) + b_vocab
      captions[:, i] = np.argmax(scores, axis=1)
      x = W_embed[captions[:, i]]

    return captions
