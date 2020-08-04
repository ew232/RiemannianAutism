### check if two groups have different eigenvalues

import os, glob
import numpy as np
import tensorflow as tf
import spd_backprop_functions as spd
import csv
import math
import pandas as pd
import scipy.io as sio
import shutil
import random

import pickle
import gc

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_spd_matrix
from scipy import stats
from scipy.linalg import fractional_matrix_power
from scipy.linalg import expm


sqrtBInv = computeC(Xmu) # Xmu is the Frechet mean of sample covariance matrices

######

tf.reset_default_graph()
minEval = 0.01 # 1e-3
keepProb = tf.placeholder(tf.float32)
trainPhase = tf.placeholder(tf.bool)
lrC = tf.placeholder(tf.float32)
lrW = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None,spdDim,spdDim])
inputLabels = tf.placeholder(tf.float32, [None,nClasses])
activation = x

### biMapAIM
Cidx = 0
C = tf.get_variable('BiMapAIMC', initializer=sqrtBInv)
CXC = tf.map_fn(lambda y: tf.matmul(C, tf.matmul(y, C)), activation)
activation = CXC

### log layer
eigval, eigu = tf.linalg.eigh(activation, name='eigh')
logCXC = tf.matmul(eigu, tf.matmul(tf.linalg.diag(tf.log(eigval)), eigu, adjoint_b=True))

logCXClinear = tf.map_fn(lambda y: tf.reshape(y, [-1]), logCXC)
activation = logCXClinear

### dropout layer
activation = tf.nn.dropout(activation,0.9) # 0.75

### fully connected
WFc = tf.get_variable('WFc', initializer=0*spd.xavier_init([int(activation.shape[1]), nClasses]))
bFc = tf.get_variable('bFc', initializer=0*spd.xavier_init([nClasses]))
logits = tf.matmul(activation, WFc) + bFc
predict = tf.nn.softmax(logits)

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=inputLabels, logits=logits))

trainableVars = tf.trainable_variables()
trainableVarsC = trainableVars[0]
trainableVarsW = trainableVars[1:]
optimizerC = tf.train.AdamOptimizer(learning_rate = lrC)
optimizerW = tf.train.AdamOptimizer(learning_rate = lrW)
gvC = optimizerC.compute_gradients(cost, trainableVarsC)
gvCsym = 0.5*(gvC[0][0]+tf.transpose(gvC[0][0]))
gvW = optimizerW.compute_gradients(cost, trainableVarsW)

CnextPoint = tfExpMap(gvC[0][1], -lrC*gvCsym)
CnextPointSym = 0.5*(CnextPoint+tf.transpose(CnextPoint))
eigvalC, eiguC = tf.linalg.eigh(CnextPointSym)
CnextPointFix = tf.matmul(eiguC, tf.matmul(tf.linalg.diag(tf.abs(eigvalC)), eiguC, adjoint_b=True))

optimizerMinC = C.assign(CnextPointFix)
optimizerMinW = optimizerW.apply_gradients(gvW)


