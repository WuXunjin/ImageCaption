import urllib.request, tempfile

import numpy as np
from scipy.misc import imread


def image_from_url(url):
  """
  Read an image from a URL. Returns a numpy array with the pixel data.
  """
  try:
    f = urllib.request.urlopen(url)
    _, fname = tempfile.mkstemp()
    with open(fname, 'wb') as ff:
      ff.write(f.read())
    img = imread(fname)

    return img
  except urllib.request.URLError as e:
    print ('URL Error: ', e.reason, url)
  except urllib.request.HTTPError as e:
    print ('HTTP Error: ', e.code, url)
