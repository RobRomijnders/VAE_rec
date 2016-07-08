# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders

TODO
- Cross validate over different learning-rates
"""
import sys
sys.path.append('/home/rob/Dropbox/ml_projects/VAE_rec')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from VAE_util import *



class Model():
  def __init__(self,config):
    """Hyperparameters"""
    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    max_grad_norm = config['max_grad_norm']
    batch_size = config['batch_size']
    sl = config['sl']
    mixtures = config['mixtures']
    crd = config['crd']
    learning_rate = config['learning_rate']
    num_l = config['num_l']
    self.sl = sl
    self.crd = crd
    self.batch_size = batch_size


    #Function for initialization
    def xv_init(arg_in, arg_out,shape=None):
      low = -np.sqrt(6.0/(arg_in + arg_out))
      high = np.sqrt(6.0/(arg_in + arg_out))
      if shape is None:
        tensor_shape = (arg_in, arg_out)
      return tf.random_uniform(tensor_shape, minval=low, maxval=high, dtype=tf.float32)

    # Nodes for the input variables
    self.x = tf.placeholder("float", shape=[batch_size, crd,sl], name = 'Input_data')
    x_next = tf.sub(self.x[:,:3,1:],  self.x[:,:3,:sl-1])
    xn1,xn2,xn3 = tf.split(1,3,x_next)   #Now tensors in [batch_size,1,seq_len-1]
    rev_dims = [False, False, True]
    xn1 = tf.reverse(xn1,rev_dims)
    xn2 = tf.reverse(xn2,rev_dims)
    xn3 = tf.reverse(xn3,rev_dims)


    with tf.variable_scope("Enc") as scope:
      cell_enc = tf.nn.rnn_cell.LSTMCell(hidden_size)
      cell_enc = tf.nn.rnn_cell.MultiRNNCell([cell_enc] * num_layers)

      #Initial state
      initial_state_enc = cell_enc.zero_state(batch_size, tf.float32)


      outputs_enc = []
      self.states_enc = []
      state = initial_state_enc
      for time_step in range(sl):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell_enc(self.x[:, :, time_step], state)
        outputs_enc.append(cell_output)

    with tf.name_scope("Enc_2_lat") as scope:
      #m_enc,h_enc = tf.split(1,2,self.final_state)
      #layer for mean of z
      W_mu = tf.Variable(xv_init(hidden_size,num_l))
      b_mu = tf.Variable(tf.constant(0.1,shape=[num_l],dtype=tf.float32))
      self.z_mu = tf.nn.xw_plus_b(cell_output,W_mu,b_mu)  #mu, mean, of latent space

      #layer for sigma of z
      W_sig = tf.Variable(xv_init(hidden_size,num_l))
      b_sig = tf.Variable(tf.constant(0.1,shape=[num_l],dtype=tf.float32))
      z_sig_log_sq = tf.nn.xw_plus_b(cell_output,W_sig,b_sig)  #sigma of latent space, in log-scale and squared.
      # This log_sq will save computation later on. log(sig^2) is a real number, so no sigmoid is necessary

    with tf.name_scope("Latent_space") as scope:
      self.eps = tf.random_normal(tf.shape(self.z_mu),0,1,dtype=tf.float32)
      self.z = self.z_mu + tf.mul(tf.sqrt(tf.exp(z_sig_log_sq)),self.eps)   #Z is the vector in latent space

    with tf.variable_scope("Lat_2_dec") as scope:
      #Create initial vector
      params_xend = 3 + 4    # 3 (X,Y,Z) plus 4 (sx,sx,sz,rho)
      W_xend = tf.Variable(xv_init(num_l,params_xend))
      self.b_xend = tf.Variable(tf.constant(0.1,shape=[params_xend],dtype=tf.float32))
      self.parameters_xend = tf.nn.xw_plus_b(self.z,W_xend,self.b_xend)

      mu1x,mu2x,mu3x,s1x,s2x,s3x,rhox = tf.split(1,params_xend,self.parameters_xend)  #Individual vectors in [batch_size,1]
      s1x = tf.exp(s1x)
      s2x = tf.exp(s2x)
      s3x = tf.exp(s3x)
      rhox = tf.tanh(rhox)
      x_end = tf.concat(1,[mu1x,mu2x,mu3x])


      #Reconstruction loss for x_end
      xs1,xs2,xs3 = tf.split(1,3,self.x[:,:3,sl-1])
      pxend12 = tf_2d_normal(xs1, xs2, mu1x, mu2x, s1x, s2x, rhox)   #probability in x1x2 plane
      pxend3 = tf_1d_normal(xs3,mu3x,s3x)
      pxend = tf.mul(pxend12,pxend3)
      loss_xend = -tf.log(tf.maximum(pxend, 1e-20)) # at the beginning, some errors are exactly zero.
      self.cost_xstart = tf.reduce_mean(loss_xend)###tf.constant(0.0)#
      #Create initial hidden state and memory state
      W_hstart = tf.Variable(xv_init(num_l,hidden_size))
      b_hstart = tf.Variable(tf.constant(0.01,shape=[hidden_size],dtype=tf.float32))
      h_start = tf.nn.xw_plus_b(self.z,W_hstart,b_hstart)

    with tf.variable_scope("Out_layer") as scope:
      params = 7 # x,y,z,sx,sy,sz,rho
      output_units = mixtures*params  #Two for distribution over hit&miss, params for distribution parameters
      W_o = tf.Variable(tf.random_normal([hidden_size,output_units], stddev=0.1))
      b_o = tf.Variable(tf.constant(0.5, shape=[output_units]))


    with tf.variable_scope("Dec") as scope:
      cell_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)
      cell_dec = tf.nn.rnn_cell.MultiRNNCell([cell_dec] * num_layers)

      #Initial state
      initial_state_dec = tf.tile(h_start,[1,2*num_layers])
      PARAMS = []
      self.states = []
      state = initial_state_dec
      x_in = x_end
      x_collect = []
      x_collect.append(x_in)
      for time_step in range(sl):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell_dec(x_in, state)
        self.states.append(state)
        #Convert hidden state to offset for the next
        params_MDN = tf.nn.xw_plus_b(cell_output,W_o,b_o) # Now in [batch_size,output_units]
        PARAMS.append(params_MDN)
        x_in = x_in - params_MDN[:,:3]   #First three columns are the new x_in
        x_collect.append(x_in)

    #Prepare x_collect for extraction
    self.x_col = tf.pack(x_collect)   #in [seq_len, batch_size,crd]


    with tf.variable_scope("Loss_calc") as scope:
      ### Reconstruction loss
      PARAMS = tf.pack(PARAMS[:-1])
      PARAMS = tf.transpose(PARAMS,[1,2,0])  # Now in [batch_size, output_units,seq_len-1]
      mu1,mu2,mu3,s1,s2,s3,rho = tf.split(1,7,PARAMS)  #Each Tensor in [batch_size,seq_len-1]
      s1 = tf.exp(s1)
      s2 = tf.exp(s2)
      s3 = tf.exp(s3)
      rho = tf.tanh(rho)
      px1x2 = tf_2d_normal(xn1, xn2, mu1, mu2, s1, s2, rho)   #probability in x1x2 plane
      px3 = tf_1d_normal(xn3,mu3,s3)
      px1x2x3 = tf.mul(px1x2,px3)  #Now in [batch_size,1,seq_len-1]
      loss_seq = -tf.log(tf.maximum(px1x2x3, 1e-20)) # at the beginning, some errors are exactly zero.
      self.cost_seq = tf.reduce_mean(loss_seq)

      ### KL divergence between posterior on encoder and prior on z
      self.cost_kld = tf.reduce_mean(-0.5*tf.reduce_sum((1+z_sig_log_sq-tf.square(self.z_mu)-tf.exp(z_sig_log_sq)),1))   #KL divergence

      self.cost = self.cost_seq + self.cost_kld + self.cost_xstart

    with tf.name_scope("train") as scope:
      tvars = tf.trainable_variables()
      #We clip the gradients to prevent explosion
      grads = tf.gradients(self.cost, tvars)
      grads, _ = tf.clip_by_global_norm(grads,max_grad_norm)

      #Some decay on the learning rate
      global_step = tf.Variable(0,trainable=False)
      lr = tf.train.exponential_decay(learning_rate,global_step,1000,0.90,staircase=False)
      optimizer = tf.train.AdamOptimizer(lr)
      gradients = zip(grads, tvars)
      self.train_step = optimizer.apply_gradients(gradients,global_step=global_step)
      # The following block plots for every trainable variable
      #  - Histogram of the entries of the Tensor
      #  - Histogram of the gradient over the Tensor
      #  - Histogram of the grradient-norm over the Tensor
      self.numel = tf.constant([[0]])
      for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
          grad_values = gradient.values
        else:
          grad_values = gradient

        self.numel +=tf.reduce_sum(tf.size(variable))

        h1 = tf.histogram_summary(variable.name, variable)
        h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
        h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
    #Define one op to call all summaries
    self.merged = tf.merge_all_summaries()

