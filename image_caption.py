
import time, os, json, pickle
import numpy as np
import matplotlib.pyplot as plt

from rnn_layers import *
from captioning_solver import CaptioningSolver
from classifiers.rnn import CaptioningRNN
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from image_utils import image_from_url



data = load_coco_data(pca_features=True)

small_data = load_coco_data(max_train=40000)

# small_lstm_model = CaptioningRNN(cell_type='lstm',
#           word_to_idx=data['word_to_idx'],
#           input_dim=data['train_features'].shape[1],
#           hidden_dim=512,
#           wordvec_dim=256,
#           dtype=np.float32,
#           optim=True)
#
# small_lstm_solver = CaptioningSolver(small_lstm_model, small_data,
#            update_rule='adam',
#            num_epochs=60,
#            batch_size=50,
#            optim_config={
#              'learning_rate': 1e-3,
#            },
#            lr_decay=0.92,
#            verbose=True, print_every=100,)
#
# small_lstm_solver.train()
#
# # Plot the training losses
# plt.plot(small_lstm_solver.loss_history)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training loss history')
# plt.show()

# with open('caption_model.pkl', 'wb') as f:
#     pickle.dump(small_lstm_model, f)

with open('caption_model1.pkl', 'rb') as f:
  small_lstm_model = pickle.load(f)

split = 'val'
mini_batch = sample_coco_minibatch(small_data, split=split, batch_size=15)
gt_captions, features, urls = mini_batch
gt_captions = decode_captions(gt_captions, data['idx_to_word'])

sample_captions = small_lstm_model.sample_caption(features)
sample_captions = decode_captions(sample_captions, data['idx_to_word'])

for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
  plt.imshow(image_from_url(url))
  plt.title('%s\n%s\nGT: %s' %(split, sample_caption, gt_caption))
  plt.axis('off')
  plt.show()

