from __future__ import print_function
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, concatenate
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from tensorflow.python.keras.layers.core import Dropout, Reshape, Permute, RepeatVector, Flatten, Lambda
#import tensorflow.keras.layers.BatchNormalization as BN
#from tensorflow.keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import RMSprop, Adam, Adagrad, Nadam, SGD, Adadelta, Adamax
#from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
import numpy as np
import random
import sys
import glob
from time import gmtime, strftime
import copy
import pickle
import json
import tensorflow as tf
import math
import os
import re
import subprocess
def pred(wakatialllists, fnb, fnp, bunrui):
  rng = 5
  writetexts = []
  poslistdic = {}
  print(fnb)
  with open('./' + bunrui + '/' + fnp, "rb") as fr:
    text = [row.decode('utf-8').split(",")[0].replace('\t', ',') for row in fr]
  for textline in sorted(set(text)):
    word, pos = textline.split(",")
    poslistdic.setdefault(word, []).append(pos)


  for wakatialllis in wakatialllists:
    wakatialllist = wakatialllis.strip()
    print("start to loading term_vec and poslist")
    ansindx = []
    term_vec = []
    term_vec_lists = pickle.loads(open('./termvecpkl/' + wakatialllist + '.term_vec.pkl', 'rb').read())
    allposlistdic = pickle.loads(open('./allposlistset/' + wakatialllist + '.allposlist.pkl', 'rb').read())
    text = open('/home/ubuntu/content/' + wakatialllist[0:2] + '/' + fnb, 'r').read().replace('\n', ' ').split()
    if len(text) < 11:
      break
    picking_up = []
    for term_vec_list in term_vec_lists:
      term_vec.append([term_vec_list[0], term_vec_list[2]])
      ansindx.append(term_vec_list[0])
    print(len(term_vec))
    term_vec_lists = []
    for i in range(rng, len(text) - rng, 1):
      try:
        head = list(map(lambda x:term_vec[ansindx.index(x)][1], text[i-rng:i] )) 
        tail = list(map(lambda x:term_vec[ansindx.index(x)][1], text[i+1:i+rng] )) 
      except ValueError as e:
        continue
      head.extend(tail)
      x = np.array(head)
      y = text[i]
      picking_up.append( (x, y, text[i-rng:i+rng]) )
    print(len(picking_up))
    answers   = []
    texts     = []
    writetext = []
    sentences = []
    for dbi, picked in enumerate(picking_up):
      x, y,  pure_text = picked
      sentences.append(x)
      answers.append(y)
      texts.append(pure_text)
    if len(sentences) == 0:
      break
    X = np.zeros((len(sentences), len(sentences[0]), 128), dtype=np.float64)
    for i, sentence in enumerate(sentences):
      if i%10000 == 0:
        print("building training vector... iter %d"%i)
      for t, vec in enumerate(sentence):
        X[i, t, :] = vec
    model_type = './modelzip/models' + wakatialllist + '/snapshot.000000030.model'
    print("model type is %s"%model_type)
    model  = load_model(model_type)
    results = model.predict(X)
    del X
    del model
    itext = 0
    j = 0
    for sent, xtext, result in zip(sentences, texts, results):
      itext = itext + 1
      while not text[itext] == xtext[5]:
        writetext.append(text[itext] + ':,')
        itext = itext + 1
      wtext = xtext[5] + ':'
      for i,f in sorted([(i,f) for i,f in enumerate(result.tolist())], key=lambda x:x[1]*-1):
        if f < 0.99:
          writetext.append(wtext)
          break
        if xtext[5] != ansindx[i] and set(poslistdic[xtext[5]]) & set(allposlistdic[ansindx[i]]):
          wtext = wtext + ansindx[i] + ';' + str(f) + ','
        j += 1
        if j % 5 == 0:
          writetext.append(wtext)
          break
    writetexts.append(writetext)
  return writetexts
dictwakatiall = {}
lswakatiall = " ".join(['ls','./01-50wakatiall/*.wwakatiall'])
output = subprocess.getoutput(lswakatiall)
for fni in output.split('\n'):
  print(fni[-17:-15])
  if fni[-17:-15] in dictwakatiall:
      dictwakatiall[fni[-17:-15]].append(fni.strip()[-17:])
  else:
      dictwakatiall.update({fni[-17:-15] : [fni.strip()[-17:]]})
print(dictwakatiall) 
f=open('./bunruiwall-sort-uniq-11.csv')
lines=f.readlines()
f.close()
writetext = []
writetexts = []
outputs = []
for linei in lines:
  print(linei)
  line = linei
  wakatialllists = dictwakatiall[line[0:2]]
  fnb = line[3:64]
  fnp = line[3:56] + '.poslist'
  bunrui = line[0:2]
  writetexts = pred(wakatialllists, fnb, fnp, bunrui)
  with open('pred' + fnb + '.txt', 'a') as fwt:
    for ls in zip(*writetexts):
      fwt.write('|'.join(ls) + "\n")

