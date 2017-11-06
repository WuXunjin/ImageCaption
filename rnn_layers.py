import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):

  next_h, cache = None, None

  hh = np.dot(prev_h, Wh)
  hx = np.dot(x, Wx)
  next_h = np.tanh(hh + hx + b)
  cache = (x, prev_h, Wx, Wh, next_h)

  return next_h, cache


def rnn_step_backward(dnext_h, cache):

  dx, dprev_h, dWx, dWh, db = None, None, None, None, None

  x, prev_h, Wx, Wh, next_h = cache
  dhraw = (1 - next_h * next_h) * dnext_h
  db = np.sum(dhraw, axis=0)
  dWh = np.dot(prev_h.T, dhraw)
  dprev_h = np.dot(dhraw, Wh.T)
  dWx = np.dot(x.T, dhraw)
  dx = np.dot(dhraw, Wx.T)

  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):

  h, cache = None, None

  N, T, D = x.shape
  H = h0.shape[1]
  
  xs, hs = {}, {}
  hs[-1] = h0
  h = np.zeros((N, T, H))
  for i in range(T):
    xs[i] = x[:, i, :]
    hs[i], _ = rnn_step_forward(xs[i], hs[i - 1], Wx, Wh, b)
    h[:, i, :] = hs[i]

  cache = (x, h0, Wx, Wh, hs)

  return h, cache



def rnn_backward(dh, cache):

  dx, dh0, dWx, dWh, db = None, None, None, None, None

  x, h0, Wx, Wh, hs = cache
  N, T, D = x.shape
  H = h0.shape[1]

  # dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache) # x, prev_h, Wx, Wh, next_h = cache

  dx, dh0, dWx, dWh = np.zeros_like(x), np.zeros_like(h0), np.zeros_like(Wx), np.zeros_like(Wh)
  db = np.zeros((H, ))
  dh0 = np.zeros_like(h0)
  for i in reversed(range(T)):
    t_cache = (x[:, i, :], hs[i - 1], Wx, Wh, hs[i])
    dx[:, i, :], dh0, t_dWx, t_dWh, t_db = rnn_step_backward(dh[:, i, :] + dh0, t_cache)
    db += t_db
    dWh += t_dWh
    dWx += t_dWx

  return dx, dh0, dWx, dWh, db


def rnn_forward_caption(x, h0, Wx, Wh, b):
  h, cache = None, None

  N, T, D = x.shape
  H = h0.shape[1]

  xs, hs = {}, {}
  # hs[-1] = h0
  hs[-1] = np.zeros_like(h0)
  h = np.zeros((N, T, H))
  for i in range(T):
    xs[i] = x[:, i, :]
    hs[i], _ = rnn_step_forward(xs[i], hs[i - 1] + h0, Wx, Wh, b)
    h[:, i, :] = hs[i]

  cache = (x, h0, Wx, Wh, hs)

  return h, cache


def rnn_backward_caption(dh, cache):

  dx, dh0, dh0_, dWx, dWh, db = None, None, None, None, None, None

  x, h0, Wx, Wh, hs = cache
  N, T, D = x.shape
  H = h0.shape[1]

  dx, dh0, dWx, dWh = np.zeros_like(x), np.zeros_like(h0), np.zeros_like(Wx), np.zeros_like(Wh)
  db = np.zeros((H,))
  dh0_ = np.zeros_like(h0)
  dh0 = np.zeros_like(h0)
  for i in reversed(range(T)):
    t_cache = (x[:, i, :], hs[i - 1], Wx, Wh, hs[i])
    dx[:, i, :], dh0_, t_dWx, t_dWh, t_db = rnn_step_backward(dh[:, i, :] + dh0_, t_cache)
    dh0 += dh0_
    db += t_db
    dWh += t_dWh
    dWx += t_dWx

  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):

  out, cache = None, None

  N, T = x.shape
  V, D = W.shape

  out = np.zeros((N, T, D))
  for i in range(N):
    for j in range(T):
      out[i, j] = W[x[i, j]]
  cache = (x, W)

  return out, cache


def word_embedding_backward(dout, cache):

  dW = None

  x, W = cache
  N, T = x.shape
  V, D = W.shape

  dW = np.zeros_like(W)
  for i in range(N):
    for j in range(T):
      dW[x[i, j]] += dout[i, j]

  return dW



def sigmoid(x):

  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):

  next_h, next_c, cache = None, None, None

  H = prev_h.shape[1]
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  i = sigmoid(a[:, :H])
  f = sigmoid(a[:, H:2 * H])
  o = sigmoid(a[:, 2 * H : 3 * H])
  g = np.tanh(a[:, 3 * H :])
 
  next_c = f * prev_c + i * g 
  next_h = o * np.tanh(next_c)
  cache = (x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c, next_h)

  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):

  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None

  x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c, next_h = cache
  do = dnext_h * np.tanh(next_c)
  ddnext_c = dnext_c + (1 - np.tanh(next_c) * np.tanh(next_c)) * dnext_h * o
  df = ddnext_c * prev_c
  dprev_c = ddnext_c * f
  di = ddnext_c * g
  dg = ddnext_c * i

  ddi = (1 - i) * i * di
  ddf = (1 - f) * f * df
  ddo = (1 - o) * o * do
  ddg = (1 - g * g) * dg

  da = np.hstack((ddi, ddf, ddo, ddg))
  db = np.sum(da, axis=0)
  # a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  dx = np.dot(da, Wx.T)
  dWx = np.dot(x.T, da)

  dprev_h = np.dot(da, Wh.T)
  dWh = np.dot(prev_h.T, da)


  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):

    h, cache = None, None
    N, T, D = x.shape
    H = h0.shape[1]

    h = np.zeros((N, T, H))
    cs = {}
    cs[-1] = np.zeros((N, H))
    h_interm = h0
    cache = []

    for i in range(T):
        # hs[i], cs[i], ps[i] = lstm_step_forward(x[:, i, :], hs[i -1], cs[i - 1], Wx, Wh, b)
        h [:, i, :], cs[i], cache_sub = lstm_step_forward(x[:, i, :], h_interm, cs[i - 1], Wx, Wh, b)
        h_interm = h[:, i, :]
        cache.append(cache_sub)

    return h, cache


def lstm_backward(dh, cache):

    dx, dh0, dWx, dWh, db = None, None, None, None, None

    x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c, next_h = cache[-1]
    N, D = x.shape
    T, H = dh.shape[1:]

    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros((4 * H, ))
    dc = np.zeros((N, H))

    for i in reversed(range(T)):
        dx[:, i, :], dh0, dc, t_dWx, t_dWh, t_db = lstm_step_backward(dh[:, i, :] + dh0, dc, cache.pop())
        dWx += t_dWx
        dWh += t_dWh
        db += t_db

    return dx, dh0, dWx, dWh, db


def lstm_forward_caption(x, h0, Wx, Wh, b):

    h, cache = None, None
    N, T, D = x.shape
    H = h0.shape[1]

    h = np.zeros((N, T, H))
    cs = {}
    cs[-1] = np.zeros((N, H))
    # h_interm = h0
    h_interm = np.zeros_like(h0)
    cache = []

    for i in range(T):
        h [:, i, :], cs[i], cache_sub = lstm_step_forward(x[:, i, :], h_interm + h0, cs[i - 1], Wx, Wh, b)
        h_interm = h[:, i, :]
        cache.append(cache_sub)

    return h, cache


def lstm_backward_caption(dh, cache):

    dx, dh0, dh0_, dWx, dWh, db = None, None, None, None, None, None

    x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c, next_h = cache[-1]
    N, D = x.shape
    T, H = dh.shape[1:]

    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dh0_ = np.zeros((N, H))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros((4 * H, ))
    dc = np.zeros((N, H))

    for i in reversed(range(T)):
        dx[:, i, :], dh0_, dc, t_dWx, t_dWh, t_db = lstm_step_backward(dh[:, i, :] + dh0_, dc, cache.pop())
        dh0 += dh0_
        dWx += t_dWx
        dWh += t_dWh
        db += t_db

    return dx, dh0, dWx, dWh, db




def temporal_affine_forward(x, w, b):

  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):

  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print ('dx_flat: ', dx_flat.shape)
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

