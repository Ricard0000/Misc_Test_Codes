# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:47:12 2021

@author: Ricardo
"""

# -*- coding: utf-8 -*-
"""
Fititng to a function of x
"""



import sys
sys.path.insert(0, '../../Utilities/')
import scipy.io

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from scipy import interpolate




def coeffsrho(W0,W1,W2,N_layers,eps):

    if Max_orders==0:
        c_rho=W0[N_layers][0]+W0[N_layers][3]*(W0[0][0,0]*W0[0][1,0])
        c_v_rho_x=W0[N_layers][1]+W0[N_layers][3]*(W0[0][0,0]*W0[0][0,1]+W0[0][0,1]*W0[0][1,0])
        c_v2_rho_xx=W0[N_layers][3]*(W0[0][0,1]*W0[0][1,1])

#        c_rho=W0[N_layers][0]+W0[N_layers][2]*(W0[0][0,0]*W0[0][1,0])
#        c_v_rho_x=W0[N_layers][1]+W0[N_layers][2]*(W0[0][0,0]*W0[0][1,1]+W0[0][0,1]*W0[0][1,0])
#        c_v2_rho_xx=W0[N_layers][2]*(W0[0][0,1]*W0[0][1,1])

    elif Max_orders==2:
        c_rho=W0[1][0]+W0[1][2]*(W0[0][0,0]*W0[0][1,0])
        c_rho=c_rho+(W1[1][0]+W1[1][2]*(W1[0][0,0]*W1[0][1,0]))/eps
        c_rho=c_rho+(W2[1][0]+W2[1][2]*(W2[0][0,0]*W2[0][1,0]))/eps**2

        c_v_rho_x=W0[1][1]+W0[1][2]*(W0[0][0,0]*W0[0][1,1]+W0[0][0,1]*W0[0][1,0])
        c_v_rho_x=c_v_rho_x+(W1[1][1]+W1[1][2]*(W1[0][0,0]*W1[0][1,1]+W1[0][0,1]*W1[0][1,0]))/eps
        c_v_rho_x=c_v_rho_x+(W2[1][1]+W2[1][2]*(W2[0][0,0]*W2[0][1,1]+W2[0][0,1]*W2[0][1,0]))/eps**2

        c_v2_rho_xx=W0[1][2]*(W0[0][0,1]*W0[0][1,1])
        c_v2_rho_xx=c_v2_rho_xx+W1[1][2]*(W1[0][0,1]*W1[0][1,1])/eps
        c_v2_rho_xx=c_v2_rho_xx+W2[1][2]*(W2[0][0,1]*W2[0][1,1])/eps**2
    else:
        print('not implemented')
    
    return c_rho,c_v_rho_x,c_v2_rho_xx

def coeffsg(W0,W1,W2,N_layers,eps):
    #Use a tree to compute the coefficients:

    def coef_g_b1(W0,N_layers):
        return W0[0][1,0]*W0[0][0,0]

    def coef_g_b2(W0,N_layers):
        Temp=W0[1][1,0]*W0[1][0,0]+W0[1][1,0]*W0[1][0,3]*coef_g_b1(W0,N_layers)
        Temp=Temp+W0[1][1,3]*W0[1][0,0]*coef_g_b1(W0,N_layers)+W0[1][1,3]*W0[1][0,3]*coef_g_b1(W0,N_layers)*coef_g_b1(W0,N_layers)
        return Temp

    def coef_g_b3(W0,N_layers):
        Temp=W0[2][1,0]*W0[2][0,0]+W0[2][1,0]*W0[2][0,3]*coef_g_b1(W0,N_layers)+W0[2][1,0]*W0[2][0,4]*coef_g_b2(W0,N_layers)
        Temp=Temp+W0[2][1,3]*W0[2][0,0]*coef_g_b1(W0,N_layers)+W0[2][1,3]*W0[2][0,3]*coef_g_b1(W0,N_layers)*coef_g_b1(W0,N_layers)
        Temp=Temp+W0[2][1,4]*W0[2][0,0]*coef_g_b2(W0,N_layers)+W0[2][1,4]*W0[2][0,4]*coef_g_b2(W0,N_layers)*coef_g_b2(W0,N_layers)
        return Temp

    def coef_adv_b1(W0,N_layers):
        return (W0[0][1,0]*W0[0][0,1]+W0[0][0,0]*W0[0][1,1])#correct
    def coef_b1_id(W0,N_layers):
        return W0[0][1,0]*W0[0][0,0]#correct
    
    def coef_b2_id(W0,N_layers):
        Temp=W0[1][1,0]*W0[1][0,0]+W0[1][1,0]*W0[1][0,3]*coef_b1_id(W0,N_layers)
        Temp=Temp+W0[1][1,3]*W0[1][0,0]*coef_b1_id(W0,N_layers)
        Temp=Temp+W0[1][1,3]*W0[1][0,3]*coef_b1_id(W0,N_layers)*coef_b1_id(W0,N_layers)
        return Temp
    
    def coef_adv_b2(W0,N_layers):
        Temp=W0[1][1,0]*(W0[1][0,1]+W0[1][0,3]*coef_adv_b1(W0,N_layers))+W0[1][1,1]*(W0[1][0,0]+W0[1][0,3]*coef_b1_id(W0,N_layers))
        Temp=Temp+W0[1][1,3]*W0[1][0,0]*coef_adv_b1(W0,N_layers)+W0[1][1,3]*W0[1][0,1]*coef_b1_id(W0,N_layers)
        Temp=Temp+W0[1][1,3]*W0[1][0,3]*(coef_b1_id(W0,N_layers)*coef_adv_b1(W0,N_layers)+coef_adv_b1(W0,N_layers)*coef_b1_id(W0,N_layers))
        return Temp#correct

    def coef_adv_b3(W0,N_layers):
        Temp=W0[2][1,0]*(W0[2][0,1]+W0[2][0,3]*coef_adv_b1(W0,N_layers)+W0[2][0,4]*coef_adv_b2(W0,N_layers))+W0[2][1,1]*(W0[2][0,0]+W0[2][0,3]*coef_b1_id(W0,N_layers)+W0[2][0,4]*coef_b2_id(W0,N_layers))

        Temp=Temp+W0[2][1,3]*W0[2][0,0]*coef_adv_b1(W0,N_layers)+W0[2][1,3]*W0[2][0,1]*coef_b1_id(W0,N_layers)

        Temp=Temp+W0[1][1,3]*W0[1][0,3]*(coef_b1_id(W0,N_layers)*coef_adv_b1(W0,N_layers)+coef_adv_b1(W0,N_layers)*coef_b1_id(W0,N_layers))
        return Temp#correct

    def coef_P_b1(W0,N_layers):
        return W0[0][1,0]*W0[0][0,2]+W0[0][1,2]*W0[0][0,0]+W0[0][1,2]*W0[0][0,2]#corrrect
    def coef_Padv_b1(W0,N_layers):
        return W0[0][1,1]*W0[0][0,2]#correct
    def coef_Padv_b2(W0,N_layers):
        Temp=W0[1][1,0]*W0[1][0,3]*coef_Padv_b1(W0,N_layers)+W0[1][0,2]*W0[1][1,1]+W0[1][1,1]*W0[1][0,3]*coef_P_b1(W0,N_layers)
#        Temp=W0[1][1,0]*W0[1][0,3]*coef_Padv_b1(W0,N_layers)+W0[1][0,2]*W0[1][1,1]+W0[1][1,1]*W0[1][0,3]*coef_P_b1(W0,N_layers)
        Temp=Temp+W0[1][1,3]*W0[1][0,0]*coef_Padv_b1(W0,N_layers)+W0[1][1,3]*W0[1][0,2]*coef_adv_b1(W0,N_layers)
        Temp=Temp+W0[1][1,3]*W0[1][0,3]*(coef_b1_id(W0,N_layers)*coef_Padv_b1(W0,N_layers)+coef_Padv_b1(W0,N_layers)*coef_b1_id(W0,N_layers)+coef_adv_b1(W0,N_layers)*coef_P_b1(W0,N_layers))
        return Temp



    if Max_orders==0:

        c_g=W0[1][0]+W0[N_layers][3]*coef_g_b1(W0,N_layers)
        c_v_g_x=W0[1][1]+W0[N_layers][3]*coef_adv_b1(W0,N_layers)
        pvgx=W0[N_layers][3]*coef_Padv_b1(W0,N_layers)
        c_v_2_g_xx=W0[N_layers][3]*(W0[0][0,1]*W0[0][1,1])



#        W0[1][1]=0.0
#        W0[1][2]=0.0
#        W0[0][1,0]=0.0
#        W0[0][1,2]=0.0
#        W0[0][0,2]=-W0[0][0,0]

#        c_g=W0[1][0]
#        c_v_g_x=W0[1][1]+W0[1][3]*(W0[0][0,0]*W0[0][1,1])
#        pvgx=W0[1][3]*(W0[0][0,2]*W0[0][1,1])
#        c_v_2_g_xx=W0[1][3]*(W0[0][0,1]*W0[0][1,1])
    elif Max_orders==2:
        c_g=(W0[1][0]+W0[N_layers][3]*coef_g_b1(W0,N_layers))+(W1[1][0]+W1[N_layers][3]*coef_g_b1(W1,N_layers))/eps+(W2[1][0]+W2[N_layers][3]*coef_g_b1(W2,N_layers))/eps**2
        c_v_g_x=(W0[1][1]+W0[N_layers][3]*coef_adv_b1(W0,N_layers))+(W1[1][1]+W1[N_layers][3]*coef_adv_b1(W1,N_layers))/eps+(W2[1][1]+W2[N_layers][3]*coef_adv_b1(W2,N_layers))/eps**2
        pvgx=(W0[N_layers][3]*coef_Padv_b1(W0,N_layers))+(W1[N_layers][3]*coef_Padv_b1(W1,N_layers))/eps+(W2[N_layers][3]*coef_Padv_b1(W2,N_layers))/eps**2
        c_v_2_g_xx=W0[N_layers][3]*(W0[0][0,1]*W0[0][1,1])+W1[N_layers][3]*(W1[0][0,1]*W1[0][1,1])/eps+W2[N_layers][3]*(W2[0][0,1]*W2[0][1,1])/eps**2


#        c_g=W0[1][0]+W1[1][0]/eps+W2[1][0]/eps**2
#        c_v_g_x=W0[1][1]+W0[1][3]*(W0[0][0,0]*W0[0][1,1])+(W1[1][1]+W1[1][3]*(W1[0][0,0]*W1[0][1,1]))/eps+(W2[1][1]+W2[1][3]*(W2[0][0,0]*W2[0][1,1]))/eps**2
#        pvgx=W0[1][3]*(W0[0][0,2]*W0[0][1,1])+(W1[1][3]*(W1[0][0,2]*W1[0][1,1]))/eps+(W2[1][3]*(W2[0][0,2]*W2[0][1,1]))/eps**2
#        c_v_2_g_xx=W0[1][3]*(W0[0][0,1]*W0[0][1,1])+(W1[1][3]*(W1[0][0,1]*W1[0][1,1]))/eps+(W2[1][3]*(W2[0][0,1]*W2[0][1,1]))/eps**2
    else:
      print('no')
  #      c_g=W0[N_layers][0]
  #      c_v_g_x=W0[N_layers][1]+W0[N_layers][3]*(W0[0][0,0]*W0[0][1,1])+W0[N_layers][4]*(W0[1][0,0]*W0[1][1,1]+W0[1][1,0]*W0[0][0,0]*W0[0][1,1]*W0[1][1,3])
  #      pvgx=W0[N_layers][3]*(W0[0][1,1]*W0[0][0,2])+W0[N_layers][4]*(W0[1][0,2]*W0[1][1,1]+W0[1][1,0]*W0[0][0,2]*W0[0][1,1]*W0[1][1,3]+W0[1][1,2]*W0[0][0,0]*W0[0][1,1]*W0[1][1,3])
  #      c_v_2_g_xx=W0[N_layers][3]*(W0[0][0,1]*W0[0][1,1])
    return c_g,c_v_g_x,pvgx,c_v_2_g_xx


class PDE_CLASSIFY:
    def __init__(self,g,rho_extend,n_dat,N_layers,Nx,dt,hx,max_epsilon,min_epsilon,Max_orders,Losss,x1,x2,x3,x4,x5):
        self.g=g
        self.rho_extend=rho_extend
        
        self.Losss=Losss

        self.max_epsilon=max_epsilon
        self.min_epsilon=min_epsilon
        self.Max_orders=Max_orders
        
        self.Nx=Nx
        self.dt=dt
        self.N_layers=N_layers
        self.n_dat=n_dat
        self.hx=hx
        self.x1=x1
        self.x2=x2
        self.x3=x3
        self.x4=x4
        self.x5=x5
        
        self.g_tf = tf.placeholder(tf.float32, shape=[Nv,Nx,n_dat])
        self.rho_extend_tf = tf.placeholder(tf.float32, shape=[Nv,Nx,n_dat])

        self.w_110,self.w_120,self.w_111,self.w_121,self.w_112,self.w_122,self.w_eps, self.wstiffg2, self.sigmaS, self.sigmaA, self.cont_const,self.a0,self.b0,self.c0,self.a1,self.b1,self.c1,self.a2,self.b2,self.c2,self.a3,self.b3,self.c3,self.a4,self.b4,self.c4= self.initialize_NNg(N_layers)
        
        # tf Graphs
        self.g_pred,self.w_110_tf, self.w_120_tf,self.w_111_tf, self.w_121_tf,self.w_112_tf, self.w_122_tf,self.w_eps_tf,self.wstiff_g2_tf,self.g_innerproduct,self.rho_pred,self.eps_pred,self.sigmaS_tf,self.sigmaA_tf,self.cont0,self.cont1,self.cont2,self.cont3,self.cont4,self.diff1,self.diff2,self.diff3,self.diff4 = self.net_uv(self.g_tf,self.rho_extend_tf)
        # Loss

        gamma=0.000001
#        gamma=0.0001
#        self.loss = tf.reduce_mean(tf.abs(self.g_pred))
        #setting sigmaS[0]=1 for degree of freedom sake
        self.loss = tf.reduce_mean(tf.abs(self.g_pred))+tf.reduce_mean(tf.abs(self.sigmaS[0]-0))+gamma*(tf.reduce_mean(tf.abs(self.cont0))+tf.reduce_mean(tf.abs(self.cont1))+tf.reduce_mean(tf.abs(self.cont2))+tf.reduce_mean(tf.abs(self.cont3))+tf.reduce_mean(tf.abs(self.cont4))+tf.reduce_mean(tf.abs(self.diff1))+tf.reduce_mean(tf.abs(self.diff2))+tf.reduce_mean(tf.abs(self.diff3))+tf.reduce_mean(tf.abs(self.diff4)))#+tf.reduce_mean(tf.abs(self.sigmaS[1]-(1+100*x[1]**2)))+tf.reduce_mean(tf.abs(self.sigmaS[Nx-2]-(1+100*x[Nx-2]**2)))+tf.reduce_mean(tf.abs(self.sigmaS[Nx-1]-(1+100*x[Nx-1]**2)))#+0.000001*tf.reduce_mean(tf.abs(self.diff[1:Nx-1]))#+tf.reduce_mean(tf.abs(self.sigmaS[Nx-1]-100))#+gamma*tf.reduce_mean(tf.abs(self.g_innerproduct))# +gamma*(tf.reduce_mean(tf.abs(self.w_110_tf[0]))+tf.reduce_mean(tf.abs(self.w_120_tf[0]))+tf.reduce_mean(tf.abs(self.w_111_tf[0]))+tf.reduce_mean(tf.abs(self.w_121_tf[0]))+tf.reduce_mean(tf.abs(self.w_112_tf[0]))+tf.reduce_mean(tf.abs(self.w_122_tf[0])))+gamma*(tf.reduce_mean(tf.abs(self.w_110_tf[1]))+tf.reduce_mean(tf.abs(self.w_120_tf[1]))+tf.reduce_mean(tf.abs(self.w_111_tf[1]))+tf.reduce_mean(tf.abs(self.w_121_tf[1]))+tf.reduce_mean(tf.abs(self.w_112_tf[1]))+tf.reduce_mean(tf.abs(self.w_122_tf[1])))
#        self.loss = tf.reduce_mean(tf.abs(self.g_pred))+gamma*tf.reduce_mean(tf.abs(self.g_innerproduct-projection_term[:,1:n_dat-1]))# +gamma*(tf.reduce_mean(tf.abs(self.w_110_tf[0]))+tf.reduce_mean(tf.abs(self.w_120_tf[0]))+tf.reduce_mean(tf.abs(self.w_111_tf[0]))+tf.reduce_mean(tf.abs(self.w_121_tf[0]))+tf.reduce_mean(tf.abs(self.w_112_tf[0]))+tf.reduce_mean(tf.abs(self.w_122_tf[0])))+gamma*(tf.reduce_mean(tf.abs(self.w_110_tf[1]))+tf.reduce_mean(tf.abs(self.w_120_tf[1]))+tf.reduce_mean(tf.abs(self.w_111_tf[1]))+tf.reduce_mean(tf.abs(self.w_121_tf[1]))+tf.reduce_mean(tf.abs(self.w_112_tf[1]))+tf.reduce_mean(tf.abs(self.w_122_tf[1])))

 #       self.loss =tf.reduce_mean(tf.abs(self.g_innerproduct))#+1.0/(gamma*(tf.reduce_mean(tf.abs(self.w_110_tf[0]))+tf.reduce_mean(tf.abs(self.w_120_tf[0]))+tf.reduce_mean(tf.abs(self.w_111_tf[0]))+tf.reduce_mean(tf.abs(self.w_121_tf[0]))+tf.reduce_mean(tf.abs(self.w_112_tf[0]))+tf.reduce_mean(tf.abs(self.w_122_tf[0])))+gamma*(tf.reduce_mean(tf.abs(self.w_110_tf[1]))+tf.reduce_mean(tf.abs(self.w_120_tf[1]))+tf.reduce_mean(tf.abs(self.w_111_tf[1]))+tf.reduce_mean(tf.abs(self.w_121_tf[1]))+tf.reduce_mean(tf.abs(self.w_112_tf[1]))+tf.reduce_mean(tf.abs(self.w_122_tf[1]))))

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


    def initialize_NNg(self, N_layers):

        if Max_orders==0:
            w_110=[]
            w_120=[]
            w_111=[]
            w_121=[]
            w_112=[]
            w_122=[]
            initializeit=1.5
            for I in range(0,N_layers):
                #The variables defined here is equation 3.6 w_{i,j}
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_110.append(W)
                #of course the wieghts for the last layer is of size $1\times(2+N_layers)$
                #See last equation in 3.9
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_110.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_120.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_120.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_111.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_111.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_121.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_121.append(W_last_g)
            
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_112.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_112.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_122.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_122.append(W_last_g)




        elif Max_orders==2:
            w_110=[]
            w_120=[]
            w_111=[]
            w_121=[]
            w_112=[]
            w_122=[]
            initializeit=1.5
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_110.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_110.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_120.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_120.append(W_last_g)
            
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_111.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_111.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_121.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_121.append(W_last_g)
            
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_112.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_112.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_122.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_122.append(W_last_g)            
        else:
            print('Not implemented')
        w_eps=tf.Variable(1.0,dtype=tf.float32)
        wstiffg2=initializeit*tf.Variable(1.0,dtype=tf.float32)
        sigmaS=initializeit*tf.Variable(tf.ones([Nx,1],dtype=tf.float32),dtype=tf.float32)
        sigmaA=initializeit*tf.Variable(tf.ones([Nx,1],dtype=tf.float32),dtype=tf.float32)
        cont_const=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        a0=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        b0=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        c0=2*initializeit*tf.Variable(1.0,dtype=tf.float32)

        a1=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        b1=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        c1=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        
        a2=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        b2=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        c2=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        
        a3=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        b3=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        c3=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        
        a4=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        b4=2*initializeit*tf.Variable(1.0,dtype=tf.float32)
        c4=2*initializeit*tf.Variable(1.0,dtype=tf.float32)


        return w_110,w_120,w_111,w_121,w_112,w_122,w_eps,wstiffg2,sigmaS,sigmaA,cont_const,a0,b0,c0,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4


    def RNNg(self,g,N_layers,w0,w1,w2,eps):

        if Max_orders==0:
            W0=w0[0]
            WL0=w0[N_layers]

            Linear_As=WL0[0]*g+WL0[1]*self.advection(g,vextend)+WL0[2]*self.projection(g,wvextend)
            Right=W0[1,0]*g+W0[1,1]*self.advection(g,vextend)+W0[1,2]*self.projection(g,wvextend)
            B1=W0[0,0]*Right+W0[0,1]*self.advection(Right,vextend)+W0[0,2]*self.projection(Right,wvextend)
            F_rhs_g=Linear_As+WL0[3]*B1
        else:
            W0=w0[0]
            WL0=w0[N_layers]
            W1=w1[0]
            WL1=w1[N_layers]
            W2=w2[0]
            WL2=w2[N_layers]


            Linear_As0=(WL0[0])*g+WL0[1]*self.advection(g,vextend)+WL0[2]*self.projection(g,wvextend)
            Right0=W0[1,0]*g+W0[1,1]*self.advection(g,vextend)+W0[1,2]*self.projection(g,wvextend)
            B0=W0[0,0]*Right0+W0[0,1]*self.advection(Right0,vextend)+W0[0,2]*self.projection(Right0,wvextend)

            Linear_As1=(WL1[0])*g+WL1[1]*self.advection(g,vextend)+WL1[2]*self.projection(g,wvextend)
            Right1=W1[1,0]*g+W1[1,1]*self.advection(g,vextend)+W1[1,2]*self.projection(g,wvextend)
            B1=W1[0,0]*Right1+W1[0,1]*self.advection(Right1,vextend)+W1[0,2]*self.projection(Right1,wvextend)

            Linear_As2=(WL2[0])*g+WL2[1]*self.advection(g,vextend)+WL2[2]*self.projection(g,wvextend)
            Right2=W2[1,0]*g+W2[1,1]*self.advection(g,vextend)+W2[1,2]*self.projection(g,wvextend)
            B2=W2[0,0]*Right2+W2[0,1]*self.advection(Right2,vextend)+W2[0,2]*self.projection(Right2,wvextend)

            F_rhs_g=Linear_As0+WL0[3]*B0+(Linear_As1+WL1[3]*B1)/eps+(Linear_As2+WL2[3]*B2)/eps**2
        return F_rhs_g,w0,w1,w2


    def RNNrho(self,rho_extend,N_layers,w0,w1,w2,eps):
        if Max_orders==0:
            W0=w0[0]
            WL0=w0[N_layers]
            Linear_As=WL0[0]*rho_extend+WL0[1]*self.advection(rho_extend,vextend)
            Right=W0[1,0]*rho_extend+W0[1,1]*self.advection(rho_extend,vextend)
            B1=W0[0,0]*Right+W0[0,1]*self.advection(Right,vextend)
            F_rhs_rho=Linear_As+WL0[3]*B1
        else:
            print('cant use yet')
        return F_rhs_rho,w0,w1,w2



    def firstderiv(self,g):
        out=(tf.roll(g,1,axis=1)-tf.roll(g,-1,axis=1))/(2.0*hx)
        return out

    def secondderiv(self,g):
        out=(tf.roll(g,1,axis=1)-2.0*g+tf.roll(g,-1,axis=1))/(hx*hx)
        return out

    def rhoOP(self,rhoextend,g):
        rhog=rhoextend*g
        return rhog

    def gOP(self,g,f):
        gf=g*f
        return gf

    def advection(self,g,vextend):#This uses central order difference to compute advection. In future code,
        #higher ordermethods will be considered. For demonstration purpose, this is accurate enough.
        advec=vextend*(tf.roll(g,1,axis=1)-tf.roll(g,-1,axis=1))/(2*hx)
        return advec

    def extend_2d_to_3d(self,g):
        reduced=g
        vv=tf.concat([reduced,reduced],0)
        for I in range(0,Nv-2):
            vv=tf.concat([vv,reduced],0)
        prj=tf.reshape(vv,[Nv,Nx,n_dat])
        return prj


    def projection(self,g,wvextend):
        s=wvextend*g
        reduced=tf.math.reduce_sum(s/2.0,0)
        vv=tf.concat([reduced,reduced],0)
        for I in range(0,Nv-2):
            vv=tf.concat([vv,reduced],0)
        prj=tf.reshape(vv,[Nv,Nx,n_dat])
        return prj

    def projection2(self,g,wvextend):
        s=wvextend*g
        reduced=tf.math.reduce_sum(s/2.0,0)
        vv=tf.concat([reduced,reduced],0)
        for I in range(0,Nv-2):
            vv=tf.concat([vv,reduced],0)
        prj=tf.reshape(vv,[Nv,Nx,n_dat-2])
        return prj


    def B_op(self,g,i,w,vextend,wvextend):
        if i==0:
            sr=w[0][1,1]*self.advection(g,vextend)
            s=w[0][0,0]*sr+w[0][0,1]*self.advection(sr,vextend)+w[0][0,2]*self.projection(sr,wvextend)
        else:
            sr=w[i][1,1]*self.advection(g,vextend)
            sout=w[i][0,0]*sr+w[i][0,1]*self.advection(sr,vextend)+w[i][0,2]*self.projection(sr,wvextend)
            for I in range(0,i):
                sr=w[i][1,3+I]*self.B_op(g,I,w,vextend,wvextend)+sr
            for I in range(0,i):
                sout=w[i][0,3+I]*self.B_op(sr,I,w,vextend,wvextend)+sout
            s=sout
        return s


    def net_uv(self,g,rho_extend):

        w_eps=self.w_eps
        eps=(tf.math.tanh(w_eps) + 1)*(max_epsilon-min_epsilon )/2+min_epsilon
        print('7')
        if Max_orders==0:
            F112, w_110, w_111,w_112 = self.RNNg(g,self.N_layers,self.w_110,self.w_111,self.w_112,eps)
            F122, w_120, w_121,w_122 = self.RNNrho(rho_extend,self.N_layers,self.w_120,self.w_121,self.w_122,eps)
            w_111=self.w_111*0
            w_112=self.w_112*0
            w_121=self.w_121*0
            w_122=self.w_122*0
            temp=F112+F122
        else:
            F112,w_110,w_111,w_112 = self.RNNg(g,self.N_layers,self.w_110,self.w_111,self.w_112,eps)
            F122,w_120,w_121,w_122 = self.RNNrho(rho_extend,self.N_layers,self.w_120,self.w_121,self.w_122,eps)
            temp=F112+F122
        wstiffg2=self.wstiffg2
        sigmaS=self.sigmaS
        sigmaA=self.sigmaA

        
        #Set up sigma S before extension:
        sigmaS1=self.a0*x1**2+self.b0*x1+self.c0
        sigmaS2=self.a1*x2**2+self.b1*x2+self.c1
        sigmaS3=self.a2*x3**2+self.b2*x3+self.c2
        sigmaS4=self.a3*x4**2+self.b3*x4+self.c3
        sigmaS5=self.a4*x5**2+self.b4*x5+self.c4


        #Set up continutity
        cont0=self.a0*x1[0]**2+self.b0*x1[0]+self.c0-0
        cont1=self.a0*x2[0]**2+self.b0*x2[0]+self.c0-(self.a1*x2[0]**2+self.b1*x2[0]+self.c1)
        cont2=self.a1*x3[0]**2+self.b1*x3[0]+self.c1-(self.a2*x3[0]**2+self.b2*x3[0]+self.c2)
        cont3=self.a2*x4[0]**2+self.b2*x4[0]+self.c2-(self.a3*x4[0]**2+self.b3*x4[0]+self.c3)
        cont4=self.a3*x5[0]**2+self.b3*x5[0]+self.c3-(self.a4*x5[0]**2+self.b4*x5[0]+self.c4)



        #Set up continutity of derivatives
        diff1=2*self.a0*x2[0]+self.b0-(2*self.a1*x2[0]+self.b1)
        diff2=2*self.a1*x3[0]+self.b1-(2*self.a2*x3[0]+self.b2)
        diff3=2*self.a2*x4[0]+self.b2-(2*self.a3*x4[0]+self.b3)
        diff4=2*self.a3*x5[0]+self.b3-(2*self.a4*x5[0]+self.b4)
        
        

        sigmaS=tf.concat([sigmaS1,sigmaS2],0)
        sigmaS=tf.concat([sigmaS,sigmaS3],0)
        sigmaS=tf.concat([sigmaS,sigmaS4],0)
        sigmaS=tf.concat([sigmaS,sigmaS5],0)
        
        
        """
        cont1=0
        cont2=0
        cont3=0
        cont4=0
        diff1=0
        diff2=0
        diff3=0
        diff4=0
        """

        Stemp=tf.linalg.matmul(sigmaS,tf.ones([1,n_dat],dtype=tf.float32))
        sigmaS_extend=self.extend_2d_to_3d(Stemp)
#        K_g=g[:,:,1:n_dat-1]-g[:,:,0:n_dat-2]-dt*(temp[:,:,0:n_dat-2])+dt*sigmaS_extend[:,:,0:n_dat-2]*g[:,:,0:n_dat-2]
        K_g=g[:,:,2:n_dat-1]-g[:,:,0:n_dat-3]-1*dt*(temp[:,:,2:n_dat-1]+temp[:,:,0:n_dat-3])+dt*(sigmaS_extend[:,:,0:n_dat-3]*g[:,:,0:n_dat-3]+sigmaS_extend[:,:,2:n_dat-1]*g[:,:,2:n_dat-1])/eps
#        K_g=g[:,:,1:n_dat-1]*(1+sigmaS*dt/eps**2)-g[:,:,0:n_dat-2]*(1-sigmaA*dt)-dt*temp[:,:,0:n_dat-2]  #This is IMEX1
        K_rh=0*g
        tmepg=0*g[:,:,0:n_dat-2]
#        tmepg=(g[:,:,0:n_dat-2]*(1-sigmaA*dt)+dt*temp[:,:,0:n_dat-2])/(1+sigmaS*dt/eps**2)
#        diff_sigmaS=(tf.roll(sigmaS,0,0)-tf.roll(sigmaS,1,0))/(hx)
#        diff_sigmaS=(tf.roll(diff_sigmaS,0,0)-tf.roll(diff_sigmaS,1,0))/(hx)
#        diff_sigmaS=(tf.roll(sigmaS,1,axis=0)-2*sigmaS+tf.roll(sigmaS,-1,axis=0))/(hx**2)
        diff_sigmaS=(-1/12*tf.roll(sigmaS,2,axis=0)+4/3*tf.roll(sigmaS,1,axis=0)-5/2*sigmaS+4/3*tf.roll(sigmaS,-1,axis=0)-1/12*tf.roll(sigmaS,-2,axis=0))/(hx**2)

#        diff_sigmaS=(tf.roll(diff_sigmaS,1,0)-2*diff_sigmaS+tf.roll(diff_sigmaS,-1,0))
#        diff_sigmaS=(-0.5*tf.roll(sigmaS,2,0)+tf.roll(sigmaS,1,0)-tf.roll(sigmaS,-1,0)+0.5*tf.roll(sigmaS,-2,0))#/(hx**3)
        Frhs_innerproduct=tf.math.reduce_sum(wvextend[:,:,1:n_dat-1]*tmepg/2.0,0)
        return K_g,w_110,w_120,w_111,w_121,w_112,w_122,w_eps,wstiffg2,Frhs_innerproduct,K_rh,eps,sigmaS,sigmaA,cont0,cont1,cont2,cont3,cont4,diff1,diff2,diff3,diff4

    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        tf_dict = {self.g_tf: self.g,
                   self.rho_extend_tf: self.rho_extend}        
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
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()

    def predictrho(self,g):
        tf_dict = {self.g_tf: self.g,
                   self.rho_extend_tf: self.rho_extend}
        w_110=self.sess.run(self.w_110_tf)
        w_120=self.sess.run(self.w_120_tf)
        w_111=self.sess.run(self.w_111_tf)
        w_121=self.sess.run(self.w_121_tf)
        w_112=self.sess.run(self.w_112_tf)
        w_122=self.sess.run(self.w_122_tf)        
        w_eps=self.sess.run(self.w_eps_tf)
        wstiff_g2=self.sess.run(self.wstiff_g2_tf)
        sigmaS=self.sess.run(self.sigmaS_tf)
        sigmaA=self.sess.run(self.sigmaA_tf)
        const=self.sess.run(self.cont_const)
#        gg=self.sess.run(self.g_pred)
        return w_110,w_120,w_111,w_121,w_112,w_122,w_eps,wstiff_g2,sigmaS,sigmaA,const

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

    N_layers=1
    #This is the multiscale setting
    #Max_orders=0 means no epsilon dependence, 
    #Max_orders=2 means we have slow, medium, and fast scales.
    Max_orders=0
         
    choose_order=1
    
    if choose_order==1:
        g=np.zeros([Nv,Nx,n_dat], dtype=float)
        rho=np.zeros([Nx,n_dat], dtype=float)
        rho_extend=np.zeros([Nv,Nx,n_dat], dtype=float)
    else:
        g=np.zeros([Nv,Nx,n_dat], dtype=float)

    for K in range(0,n_dat):
        for I in range(0,Nv):
            for J in range(0,Nx):
                g[I,J,K]=gData[I,J,nskip*(K)+n_start]
    
    for K in range(0,n_dat):
        for J in range(0,Nx):
            rho[J,K]=rho_Data[J,nskip*(K)+n_start]
            
    wv=np.float32(wv)
    
    rho = np.float32(rho)

    vextend=np.zeros([Nv,Nx,n_dat],dtype=float)
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,n_dat):
                vextend[I,J,K]=v[I,0]
                
    wvextend=np.zeros([Nv,Nx,n_dat],dtype=float)
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,n_dat):
                wvextend[I,J,K]=wv[I,0]

    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,n_dat):
                rho_extend[I,J,K]=rho[J,K]

    vextend=np.zeros([Nv,Nx,n_dat],dtype=float)
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,n_dat):
                vextend[I,J,K]=v[I,0]

    xv=np.zeros([Nv,1],dtype=float)
    for I in range(0,Nv):
        xv[I,0]=-1.0+2.0/(Nv-1)*(I)

    v=np.float32(v)

    max_epsilon=1
    min_epsilon=0.0001

    Nsteps=40000
    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    projection_term=np.zeros([Nx,n_dat],dtype=float)
    temporary=np.zeros([Nv,Nx,n_dat],dtype=float)
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,n_dat):
                temporary[I,J,K]=wvextend[I,J,K]*g[I,J,K]

    for J in range(0,Nx):
        for K in range(0,n_dat):
            projection_term[J,K]=np.sum(temporary[:,J,K])

#    x1=np.linspace(0,1.0-hx,Nx)
    x1=np.linspace(0,0.2-hx,int(Nx/5))
    x2=np.linspace(0.2,0.4-hx,int(Nx/5))
    x3=np.linspace(0.4,0.6-hx,int(Nx/5))
    x4=np.linspace(0.6,0.8-hx,int(Nx/5))
    x5=np.linspace(0.8,1-hx,int(Nx/5))
    
    modelrho = PDE_CLASSIFY(g,rho_extend,n_dat,N_layers,Nx,dt,hx,max_epsilon,min_epsilon,Max_orders,Losss,x1,x2,x3,x4,x5)

    start_time = time.time()
    modelrho.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    w_110,w_120,w_111,w_121,w_112,w_122,w_eps,wstiffg2,sigmaS,sigmaA,const= modelrho.predictrho(g)

    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2, label='Log of Loss vs Number of Iterations')
    plt.show()
#    plt.legend()
#    fig.savefig('LossBDF2_example1.png')
#    savemat('Loss1.mat',{'L_domain':Losss_domain, 'Loss_y':LLosss})





    u_pred_ndat=np.zeros([Nx,1],dtype=float)

    eps=(np.tanh(w_eps) + 1)*(max_epsilon-min_epsilon)/2+min_epsilon


    c_g,c_v_g_x,pvgx,c_v_2_g_xx=coeffsg(w_110,w_111,w_112,N_layers,eps)
    c_rho,c_v_rho_x,c_v2_rho_xx=coeffsrho(w_120,w_121,w_122,N_layers,eps)

    cg2=c_g
    c_g=c_g
#    c_g=c_g-sigmaS/eps**2-sigmaA

    print('The learned IMEX1 G-equation is:')
    print(c_g,'g +',c_v_g_x,'v*g_x+',pvgx,'<vg_x>')
    print(c_rho,'rho +',c_v_rho_x,'v*rho_x+')

    ep=np.int(1)
    ep2=ep**2

    c1=abs(c_g)-ep2
    c2=abs(c_v_g_x)-ep
    c3=abs(pvgx)-ep
    c4=abs(c_v_rho_x)-ep2
    c5=abs(c_v_2_g_xx)
    c6=abs(c_rho)
    c7=abs(c_v2_rho_xx)
#    error=(abs(c1)+abs(c2)+abs(c3)+abs(c4)+abs(c5)+abs(c6)+abs(c7))/(2*ep2+2*ep)*100
    error=(abs(c1)+abs(c2)+abs(c3)+abs(c4))/(2*ep2+2*ep)*100
    
    for I in range(-10,10):
        if (1<=round(abs(c1/10**I))<10):
            J1=I
    for I in range(-10,10):
        if (1<=round(abs(c2/10**I))<10):
            J2=I
    for I in range(-10,10):
        if (1<=round(abs(c3/10**I))<10):
            J3=I
    for I in range(-10,10):
        if (1<=round(abs(c4/10**I))<10):
            J4=I


    #This is latex code for generation of data tables(You need to install all latex packages)
    #Copy paste the commented section into Latex. Inside these tables
    #Copy and paste the outputs of line 1 and line 2 below in the latex table
    """
    \begin{table}[t]
    \label{Tab:IMEX1methods}
    \vskip 0.15in
    \begin{center}
    \begin{scriptsize}
    \begin{sc}
    \begin{tabular}{lccr}
    \toprule
    $\veps$ & Multiscale & learned $g$-equation using IMEX1 Scheme & Error \\
    \midrule
    """
    #line 1
    print('1/',ep,' & No & $\partial_{t}g=-(',ep,'^2-{\color{red}',c1/10**J1,'\cdot 10^{',J1,'}}) g -(',ep,'-{\color{red}',c2/10**J2,'\cdot 10 ^{',J2,'}}) v\cdot \partial_{x}g$  & \\\\')
    #line 2
    print('& & $+ (',ep,'-{\color{red}',c3/10**J3,'\cdot 10^{',J3,'}})\langle v\partial_{x}g\\rangle -(',ep2,'-{\color{red}',c4/10**J4,'\cdot 10^{',J4,'}})v\cdot\partial_{x}\\rho+\cdots$ & {',error,'\%}\\\\')
    
    """    
    \bottomrule
    \end{tabular}
    \end{sc}
    \end{scriptsize}
    \end{center}\caption{Learned $g$-equation using the DC-RNN algorithm based on Forward-Euler (FE) and IMEX schemes.}
    \vskip -0.1in
    \end{table}
    """
    
#    SaveLoss256 = scipy.io.loadmat('Loss256nomsfwrd.mat')

#    savemat('W_bdf0',{'weightsg0':weightsg0})
#    savemat('W_bdf1',{'weightsg1':weightsg1})
#    savemat('W_stiff_bdf',{'wstiffg':wstiffg})
#    savemat('W_stiff_bdf2',{'wstiffg2':wstiffg2})



    axiss=np.linspace(0,1,Nx)
    actual=4+100*axiss**2
    fig = plt.figure()
    plt.scatter(axiss,actual, color='black', linewidth=2, label='sigma S')
    plt.show()
    
    
    
    adiff=(actual-np.roll(actual,1,0))/hx
    fig = plt.figure()
    plt.scatter(axiss[1:Nx],adiff[0,1:Nx], color='black', linewidth=2, label='sigma S')
    plt.show()
    
    
    fig = plt.figure()
    plt.scatter(axiss,sigmaS/eps-c_g+1, color='red', linewidth=1, label='$\sigma_S$ Predicted')
    plt.plot(axiss,actual, color='black',linewidth=4, label='$\sigma_S$ Exact')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Continuous')
    plt.legend()
    plt.show()    
    
    
    diff=(np.roll(sigmaS,1,0)-np.roll(sigmaS,-1,0))/(2*hx)
    fig = plt.figure()
    plt.scatter(axiss,diff, color='black', linewidth=2, label='sigma S')
    plt.show()

#    xsu=np.zeros([Nx],dtype=float)
#    for I in range(0,Nx):
#        xsu[I]=1/(Nx-1)*(I-1)
    
#    fig = plt.figure()
#    plt.scatter(xsu,projection_term[:,0])






