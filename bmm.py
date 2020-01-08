import pandas as pd
import numpy as np
from PIL import Image

# Loading the data
df = pd.read_csv("dataset/ds81.csv")
N = df.shape[0]


# Creating an "album"
album = np.zeros((N, 28, 28))
for i in range(0, N):
  temp = np.reshape( df[i:i+1].to_numpy(), (-1, 28))
  temp = (temp>127).astype(np.uint8)
  album[i] = temp


def get_random_probs(ds_size, clusters_amount):
    rand_matrix = np.random.randint(
      low=0,
      high=5000,
      size=(ds_size, clusters_amount))
    
    sum_along_row = np.repeat(
      rand_matrix.sum(axis=1).reshape((-1, 1)),
      repeats=2,
      axis=1)
    
    prob_matrix = rand_matrix/sum_along_row
    return prob_matrix

def get_cond_prob(x, p):
  p = np.expand_dims(p, axis=0)
  p = np.repeat(p, repeats=x.shape[0], axis=0)
  p = p.reshape((p.shape[0], p.shape[1], -1))
  
  x = np.expand_dims(x, axis=1)
  x = np.repeat(x, repeats=2, axis=1)
  x = x.reshape((x.shape[0], x.shape[1], -1))
  
  a = np.power(p, x).prod(axis=-1)
  b = np.power(1 - p, (1 - x)).prod(axis=-1)
  return np.multiply(a, b)

def get_apr(histo):
    p = histo.mean(axis=0)
    return p

def get_aps(conds, apr):
  temp = conds[:, 0]*apr[0] + conds[:, 1]*apr[1]
  aps_a = (conds[:, 0]*apr[0])/temp
  aps_b = (conds[:, 1]*apr[1])/temp
  return np.stack((aps_a, aps_b), axis=1)


def get_p(apr_probs):
  p = np.zeros((2, 28, 28))

  for i in range(0, 28):
    for j in range(0, 28):
      a1 = 0
      a2 = 0
      b1 = 0
      b2 = 0
      for z in range(0, N):
        a1 = a1 + album[z, i, j] * apr_probs[z, 0]
        a2 = a2 + apr_probs[z, 0]
        b1 = b1 + album[z, i, j] * apr_probs[z, 1]
        b2 = b2 + apr_probs[z, 1]
      p[0, i, j] = a1 / a2
      p[1, i, j] = b1 / b2

  return p

# Initial
p_kz = get_random_probs(N, 2)

print("cycle:")
for i in range(0, 20):
    print(i)
    # P(K)
    Pk = get_apr(p_kz)

    #p_kij
    p = get_p(p_kz)

    #p(z | k)
    p_zk = get_cond_prob(album, p)

    #p_kz
    p_kz = 1 - get_aps(p_zk, Pk)

im = Image.fromarray(np.uint8(p[1] * 255) , 'L')
#im.show()

im = Image.fromarray(np.uint8(p[0] * 255) , 'L')
#im.show()
