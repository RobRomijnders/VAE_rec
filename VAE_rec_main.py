# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders

TODO
- Cross validate over different learning-rates
"""
import sys
import socket

if 'rob-laptop' in socket.gethostname():
  sys.path.append('/home/rob/Dropbox/ml_projects/basket_local/')
  sys.path.append('/home/rob/Dropbox/ml_projects/basket_local/SportVU-seq')
  #The folder where your dataset is. Note that is must end with a '/'
  direc = '/home/rob/Dropbox/ml_projects/basket_local/SportVU-seq/'
elif 'rob-com' in socket.gethostname():
  sys.path.append('/home/rob/Documents/nn_sportvu')
#Rajiv: you can add your computer name here


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from basket_util import *
from VAE_util import *
import sklearn as sk

from sklearn.metrics import roc_auc_score,roc_curve

from data_loader_class import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.mlab as mlab
from VAE_rec_model import *



"""Hyperparameters"""
config = {}
config['num_layers'] = 1
config['hidden_size'] = 180  #Inherited from Rajiv's .ipynb, we need to discuss this
config['max_grad_norm'] = 10
config['batch_size'] = batch_size = 64
config['sl'] = sl = 22          #sequence length
config['mixtures'] = 1
config['learning_rate'] = .005
config['num_l'] = 20



ratio = 0.8         #Ratio for train-val split
plot_every = 100    #How often do you want terminal output for the performances
max_iterations = 500
dropout = 0.6       #Dropout rate in the fully connected layer



"""Load the data"""
#The name of the dataset. Note that it must end with '.csv'
csv_file = 'seq_all_9feet.csv'
#Load an instance
center = np.array([5.25, 25.0, 10.0])
dl = DataLoad(direc,csv_file, center)
#Munge the data. Arguments see the class
db = 4   #distance to basket
dl.munge_data(11,sl,db)
#Center the data
dl.center_data(center)
dl.split_train_test(ratio = 0.8)
data_dict = dl.data
dl.plot_traj_2d(20,'at %.0f feet from basket'%db)

X_train = np.transpose(data_dict['X_train'],[0,2,1])
#y_train = data_dict['y_train']
X_val = np.transpose(data_dict['X_val'],[0,2,1])
#y_val = data_dict['y_val']

N,crd,_ = X_train.shape
Nval = X_val.shape[0]

config['crd'] = crd

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

model = Model(config)

# For now, we collect performances in a Numpy array.
# In future releases, I hope TensorBoard allows for more
# flexibility in plotting
perf_collect = np.zeros((7,int(np.floor(max_iterations /plot_every))))

sess = tf.Session()

#with tf.Session() as sess:
if True:
  writer = tf.train.SummaryWriter("/home/rob/Dropbox/ml_projects/basket_local/nn_sportvu/log_tb", sess.graph)

  sess.run(tf.initialize_all_variables())

  step = 0      # Step is a counter for filling the numpy array perf_collect
  for i in range(max_iterations):
    batch_ind = np.random.choice(N,batch_size,replace=False)
#    debug = sess.run(model.sl_t,feed_dict={model.x:X_train[batch_ind], model.y_: y_train[batch_ind], model.keep_prob: dropout})
#    print(np.max(debug[0]))
#    print(np.max(debug[1]))
    if i%plot_every == 0:
      #Check training performance
      fetch = [model.cost_seq, model.cost_kld, model.cost_xstart]

      result = sess.run(fetch,feed_dict = { model.x: X_train[batch_ind]})
      perf_collect[0,step] = cost_train_seq = result[0]
      perf_collect[1,step] = cost_train_kld = result[1]
      perf_collect[4,step] = cost_train_xstart = result[2]

      #Check validation performance
      batch_ind_val = np.random.choice(Nval,batch_size,replace=False)
      fetch = [model.cost_seq, model.cost_kld, model.cost_xstart]  #, model.merged

      result = sess.run(fetch, feed_dict={ model.x: X_val[batch_ind_val]})
      perf_collect[2,step] = cost_val_seq = result[0]
      perf_collect[3,step] = cost_val_kld = result[1]
      perf_collect[5,step] = cost_val_xstart = result[2]

#      #Write information to TensorBoard
#      summary_str = result[3]
#      writer.add_summary(summary_str, i)
#      writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
      print("At %6s / %6s train (%5.3f,%5.3f,%5.3f) val (%5.3f,%5.3f,%5.3f)" % (i,max_iterations,cost_train_seq,cost_train_kld,cost_train_xstart,cost_val_seq,cost_val_kld,cost_val_xstart  ))
      step +=1
    sess.run(model.train_step,feed_dict={model.x:X_train[batch_ind]})
  #In the next line we also fetch the softmax outputs
  batch_ind_val = np.random.choice(Nval,batch_size,replace=False)
  result = sess.run([model.accuracy,model.numel], feed_dict={ model.x: X_val[batch_ind_val]})
  acc_test = result[0]
  print('The network has %s trainable parameters'%(result[1]))





#
#
#plt.figure()
#plt.plot(perf_collect[1],label= 'Train class cost')
#plt.plot(perf_collect[3],label = 'Valid class cost')
#if MDN:
#  plt.plot(perf_collect[4],label= 'Train seq cost')
#  plt.plot(perf_collect[5],label = 'Valid seq cost')
#plt.legend()
#plt.show()
# We can now open TensorBoard. Run the following line from your terminal
# tensorboard --logdir=/home/rob/Dropbox/ml_projects/basket_local/nn_sportvu/log_tb