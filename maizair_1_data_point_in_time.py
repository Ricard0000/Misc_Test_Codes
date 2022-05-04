# -*- coding: utf-8 -*-
"""

2d- differentiation code
Using random points

This is correct network
"""




# -*- coding: utf-8 -*-
"""
Accurate derivative, and second derivative calculations
"""

import sys
import scipy.io
from scipy.io import savemat
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

    def __init__(self,v_flat,wv_flat,x_flat,g_flat,gt_flat,layers,Losss,vrhox_flat):

        self.layers = layers
        self.Losss=Losss
        self.v_flat=v_flat
        self.wv_flat=wv_flat
        self.x_flat=x_flat
        self.g_flat = g_flat
        self.gt_flat = gt_flat
        self.vrhox_flat=vrhox_flat

        self.g_tf = tf.placeholder(tf.float32, shape=[Nv*Nx,1])

        # Initialize NNs
        self.weights, self.biases, self.D1, self.D2, self.c1,self.c2,self.c3,self.c4 = self.initialize_NN(layers)

        self.g_pred, self.gv_pred, self.gx_pred, self.pinn = self.net_uv(self.x_flat)

        #Define the Loss
#        self.loss = tf.reduce_mean(tf.abs(self.V_pred[pad:Nx-pad,pad:Nx-pad]-self.V[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(self.Vx_pred[pad:Nx-pad,pad:Nx-pad]-self.Vx[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))

        self.loss =0.000001*tf.reduce_mean(tf.square(self.pinn))+tf.reduce_mean(tf.square(self.g_pred-self.g_tf))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))
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
        D1=0.0*tf.Variable(tf.ones([Nv*Nx,1],dtype=tf.float32), dtype=tf.float32)
        D2=0.0*tf.Variable(tf.ones([Nv*Nx,1],dtype=tf.float32), dtype=tf.float32)
        init=0.5
        c1=init*tf.Variable(1.0, dtype=tf.float32)
        c2=init*tf.Variable(1.0, dtype=tf.float32)
        c3=init*tf.Variable(1.0, dtype=tf.float32)
        c4=init*tf.Variable(1.0, dtype=tf.float32)
        return weights, biases, D1, D2,c1,c2,c3,c4
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


    def neural_net(self, weights, biases, D1, D2):
        V=D1+self.v_flat
        X=D2+self.x_flat
        H=tf.concat([V,X], 1)
#        H=D2+self.x_flat+0.0*D1
        num_layers = len(weights) + 1
        print(H.shape)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
#            H = tf.tanh(tf.add(tf.matmul(H, W), b))
#            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
            H = tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
            print(H.shape)
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        print(H.shape)
    
        return H


    def net_uv(self, x_flat ):
        gout = self.neural_net(self.weights, self.biases, self.D1, self.D2)
        g_v=tf.gradients(gout,self.D1)[0]
        g_x=tf.gradients(gout,self.D2)[0]

        #This is computing the projection term
        #Sum over 16 grid points is hardcoded as its only 16 terms
        vgx=v_flat*g_x

        Temp = (self.wv_flat[0:Nx]*vgx[0:Nx]+self.wv_flat[Nx:2*Nx]*vgx[Nx:2*Nx]+self.wv_flat[2*Nx:3*Nx]*vgx[2*Nx:3*Nx]+self.wv_flat[3*Nx:4*Nx]*vgx[3*Nx:4*Nx]+self.wv_flat[4*Nx:5*Nx]*vgx[4*Nx:5*Nx]+self.wv_flat[5*Nx:6*Nx]*vgx[5*Nx:6*Nx]+self.wv_flat[6*Nx:7*Nx]*vgx[6*Nx:7*Nx]+self.wv_flat[7*Nx:8*Nx]*vgx[7*Nx:8*Nx]+self.wv_flat[8*Nx:9*Nx]*vgx[8*Nx:9*Nx]+self.wv_flat[9*Nx:10*Nx]*vgx[9*Nx:10*Nx]+self.wv_flat[10*Nx:11*Nx]*vgx[10*Nx:11*Nx]+self.wv_flat[11*Nx:12*Nx]*vgx[11*Nx:12*Nx]+self.wv_flat[12*Nx:13*Nx]*vgx[12*Nx:13*Nx]+self.wv_flat[13*Nx:14*Nx]*vgx[13*Nx:14*Nx]+self.wv_flat[14*Nx:15*Nx]*vgx[14*Nx:15*Nx]+self.wv_flat[15*Nx:16*Nx]*vgx[15*Nx:16*Nx])/2
#        Temp = (self.wv_flat[0:Nx-1]*vgx[0:Nx-1]+self.wv_flat[Nx:2*Nx-1]*vgx[Nx:2*Nx-1]+self.wv_flat[2*Nx:3*Nx-1]*vgx[2*Nx:3*Nx-1]+self.wv_flat[3*Nx:4*Nx-1]*vgx[3*Nx:4*Nx-1]+self.wv_flat[4*Nx:5*Nx-1]*vgx[4*Nx:5*Nx-1]+self.wv_flat[5*Nx:6*Nx-1]*vgx[5*Nx:6*Nx-1]+self.wv_flat[6*Nx:7*Nx-1]*vgx[6*Nx:7*Nx-1]+self.wv_flat[7*Nx:8*Nx-1]*vgx[7*Nx:8*Nx-1]+self.wv_flat[8*Nx:9*Nx-1]*vgx[8*Nx:9*Nx-1]+self.wv_flat[9*Nx:10*Nx-1]*vgx[9*Nx:10*Nx-1]+self.wv_flat[10*Nx:11*Nx-1]*vgx[10*Nx:11*Nx-1]+self.wv_flat[11*Nx:12*Nx-1]*vgx[11*Nx:12*Nx-1]+self.wv_flat[12*Nx:13*Nx-1]*vgx[12*Nx:13*Nx-1]+self.wv_flat[13*Nx:14*Nx-1]*vgx[13*Nx:14*Nx-1]+self.wv_flat[14*Nx:15*Nx-1]*vgx[14*Nx:15*Nx-1]+self.wv_flat[15*Nx:16*Nx-1]*vgx[15*Nx:16*Nx-1])/2

        pvgx=tf.concat([Temp,Temp],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx=tf.concat([pvgx,pvgx],0)

        pinn=gt_flat+self.c1*v_flat*g_x+self.c2*pvgx+self.c3*vrhox_flat+self.c4*gout

#        pinn=gt_flat-16*v_flat*g_x+16*pvgx+16*16*vrhox_flat+16*16*gout
#        pinn=self.gt_flat+v_flat*g_x-pvgx+vrhox_flat+g
#        gt_flat-(g+v_flat*g_x+vrhox_flat)
        return gout,g_v,g_x,pinn


    def callback(self, loss):
        print('Loss:', loss)


    def train(self, nIter):
        tf_dict = {self.g_tf: g_flat}
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
        g_pred=self.sess.run(self.g_pred)
        gv_pred=self.sess.run(self.gv_pred)
        gx_pred=self.sess.run(self.gx_pred)
        c1=self.sess.run(self.c1)
        c2=self.sess.run(self.c2)
        c3=self.sess.run(self.c3)
        c4=self.sess.run(self.c4)
        return g_pred, gv_pred, gx_pred,c1,c2,c3,c4
    
if __name__ == "__main__": 
    TTdata = scipy.io.loadmat('data_mm.mat')
#    TTdata = scipy.io.loadmat('data_mm_periodic.mat')

    x = TTdata['x']
    rho_Data = TTdata['u']
    gData = TTdata['g']
    v = TTdata['v']
    wv=TTdata['wv']
    dt=TTdata['dt']
    hx=TTdata['hx']


    Nv,Nx,n=gData.shape

    xx=x

    nskip=1
    n_start=round(0.0*n/3.0)
    n_end=round(3.0*n/3.0)
    n_dat_temp=n_end-n_start
    n_dat=round(n_dat_temp/nskip)
    T_final=n*dt
    dt=nskip*dt

    layers = [2,10,10,10,1]
    hv=v[1]-v[0]

    Time=45#fitting at a random point in time
    #Since the neural network has trouble fitting
    #to all points in time
    

    rho=np.zeros([Nv,Nx],dtype=float)
    vrhox=np.zeros([Nv,Nx],dtype=float)
    for I in range(0,Nv):
        for J in range(0,Nx):
            rho[I,J]=rho_Data[J,Time]
    rhox3=(np.roll(rho,1,1)-np.roll(rho,-1,1))/(2*hx)
    for I in range(0,Nv):
        for J in range(0,Nx):
            vrhox[I,J]=v[I]*rhox3[I,J]
    vrhox_flat=np.zeros([Nv*Nx,1],dtype=float)
    L=0
    for I in range(0,Nv):
        for J in range(0,Nx):
            vrhox_flat[L,0]=vrhox[I,J]
            L=L+1
    
    g=np.zeros([Nv,Nx],dtype=float)
    gv=np.zeros([Nv,Nx],dtype=float)
    gx=np.zeros([Nv,Nx],dtype=float)
    gt=np.zeros([Nv,Nx],dtype=float)
    gv3=(np.roll(gData,1,0)-np.roll(gData,-1,0))/(2*hv)
    gv3[0,:,Time]=gv3[1,:,Time]
    gv3[Nv-1,:,Time]=gv3[Nv-2,:,Time]
    gx3=(np.roll(gData,1,1)-np.roll(gData,-1,1))/(2*hx)
#    gt3=(np.roll(gData,1,2)-np.roll(gData,-1,2))/(2*dt)
    gt3=(np.roll(gData,1,2)-np.roll(gData,-1,2))/(dt)

    for I in range(0,Nv):
        for J in range(0,Nx):
            g[I,J]=gData[I,J,Time]
            gv[I,J]=gv3[I,J,Time]
            gx[I,J]=gx3[I,J,Time]
            gt[I,J]=gt3[I,J,Time]
            
    v_flat=np.zeros([Nv*Nx,1],dtype=float)
    wv_flat=np.zeros([Nv*Nx,1],dtype=float)
    x_flat=np.zeros([Nv*Nx,1],dtype=float)
    g_flat=np.zeros([Nv*Nx,1],dtype=float)
    gt_flat=np.zeros([Nv*Nx,1],dtype=float)
    L=0
    for I in range(0,Nv):
        for J in range(0,Nx):        
            v_flat[L,0]=v[I,0]
            wv_flat[L,0]=wv[I,0]
            x_flat[L,0]=x[J,0]
            g_flat[L,0]=g[I,J]
            gt_flat[L,0]=gt[I,J]
            L=L+1

    #This is what Nsteps does
    Nsteps=250000

    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model = NeuralNetworkCode(v_flat,wv_flat,x_flat,g_flat,gt_flat,layers,Losss,vrhox_flat)
        
    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    g_pred, gv_pred, gx_pred,c1,c2,c3,c4 = model.predict()

    print(c1)
    print(c2)
    print(c3)                
    print(c4)
    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2)
    

    #Restructuring the data from 1-d tensor to 2-d tensor

    g_pred_3d=np.zeros([Nv,Nx],dtype=float)
    gv_pred_3d=np.zeros([Nv,Nx],dtype=float)
    gx_pred_3d=np.zeros([Nv,Nx],dtype=float)
    L=0
    for I in range(0,Nv):
        for J in range(0,Nx):
            g_pred_3d[I,J]=g_pred[L,0]
            gv_pred_3d[I,J]=gv_pred[L,0]
            gx_pred_3d[I,J]=gx_pred[L,0]
            L=L+1

    v_mesh,x_mesh=np.meshgrid(x,v)
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, g_pred_3d, color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, g, color='r')
    plt.title('Exact Solution',fontsize=14,fontweight='bold')


    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, gv_pred_3d, color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, gv, color='r')
    plt.title('Exact Solution',fontsize=14,fontweight='bold')



    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, gx_pred_3d, color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, -gx, color='r')
    plt.title('Exact Solution',fontsize=14,fontweight='bold')



    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, -gt, color='r')
    plt.title('Exact Solution',fontsize=14,fontweight='bold')

    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, g, color='r')
    plt.title('Exact Solution',fontsize=14,fontweight='bold')
