# -*- coding: utf-8 -*-
"""

1d- differentiation code
Using random points

This is correct network
"""




# -*- coding: utf-8 -*-
"""
Accurate derivative, and second derivative calculations
"""

import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def Normal_CDF(inpt):
    out=0.5*(1+math.erf(inpt/np.sqrt(2)))
    return out


class NeuralNetworkCode:
    def __init__(self,x_flat,V,Vx,layers,Losss):

        self.layers = layers
        self.Losss=Losss
        self.x_flat=x_flat
        self.V = V
        self.Vx = Vx

        self.V_tf = tf.placeholder(tf.float32, shape=[Nx,1])

        # Initialize NNs
        self.weights, self.biases, self.D1, self.D2 = self.initialize_NN(layers)

        self.V_pred, self.Vx_pred = self.net_uv(self.x_flat)

        #Define the Loss
#        self.loss = tf.reduce_mean(tf.abs(self.V_pred[pad:Nx-pad,pad:Nx-pad]-self.V[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(self.Vx_pred[pad:Nx-pad,pad:Nx-pad]-self.Vx[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))

        self.loss = tf.reduce_mean(tf.abs(self.V_pred-self.V))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))
        # Optimizers
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        D1=0.0*tf.Variable(tf.ones([Nx,1],dtype=tf.float32), dtype=tf.float32)
        D2=0.0*tf.Variable(tf.ones([Nx,1],dtype=tf.float32), dtype=tf.float32)
        return weights, biases, D1, D2
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
#        return tf.Variable(tf.ones([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, weights, biases, D1, D2):
        X=D1+x
        H=X
        num_layers = len(weights) + 1
        print(H.shape)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
#            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
#            H = tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
            print(H.shape)
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        print(H.shape)
    
        return H


    def net_uv(self, x_flat ):
        V = self.neural_net(self.weights, self.biases, self.D1, self.D2)
        V_s=tf.gradients(V,self.D1)[0]
        return V,V_s


    def callback(self, loss):
        print('Loss:', loss)


    def train(self, nIter):
        tf_dict = {self.V_tf: V}
        start_time = time.time()
        L=0
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            loss_value = self.sess.run(self.loss, tf_dict)
            Losss[L,0]=loss_value
            L=L+1
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, time: %.3e' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
    def predict(self,):
#        tf_dict = {self.Payoff_flat_tf: Payoff_flat}
        V_pred=self.sess.run(self.V_pred)
        Vx_pred=self.sess.run(self.Vx_pred)
        return V_pred, Vx_pred
    
if __name__ == "__main__": 


    layers = [1, 10, 10, 10, 10, 10, 1]

    ax=-1
    bx=1
    
    Nx=60
    pad=8

    

    
    #Defining grid points (S,t)
    x=np.zeros([Nx,1], dtype=float)
    for I in range(0,Nx):
        x[I,0] = (bx-ax)*np.random.random()-(bx-ax)/2
#        x[I,0] = ax+(bx-ax)/(Nx-1)*I
        
        
    #This is the exact solution. We will use this to test the accuracy 
    #of the predicted neural network solution.
    V=np.zeros([Nx,1],dtype=float)
    Vx=np.zeros([Nx,1],dtype=float)
    for I in range(0,Nx):
        V[I,0]=np.sin(np.pi*x[I,0])
        Vx[I,0]=np.pi*np.cos(np.pi*x[I,0])
            
    x_flat=np.zeros([Nx,1],dtype=float)
    for I in range(0,Nx):
        x_flat[I,0]=x[I,0]

    #Flattening the data i.e. creating a tile like structure
    x_flat=np.zeros([Nx,1], dtype=float)






    #This is what Nsteps does
    Nsteps=60000

    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model = NeuralNetworkCode(x_flat,V,Vx,layers,Losss)
        
    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    V_pred, Vx_pred = model.predict()


    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2)
    

    #Restructuring the data from 1-d tensor to 2-d tensor
    fig = plt.figure()
    plt.scatter(x,V_pred, color='black', linewidth=2)

    fig = plt.figure()
    plt.scatter(x,V, color='black', linewidth=2)


    fig = plt.figure()
    plt.scatter(x,Vx_pred, color='black', linewidth=2)
    
    
    fig = plt.figure()
    plt.scatter(x,Vx, color='black', linewidth=2)

