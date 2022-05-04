# -*- coding: utf-8 -*-
"""
Multiscale - hits idea
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




def coeffsrho(W0,W1,W2,W3,W4,W5,N_layers,eps):

    if Max_orders==0:
        c_rho=W0[N_layers][0]+W0[N_layers][3]*(W0[0][0,0]*W0[0][1,0])
        c_rho=c_rho+W1[N_layers][0]*2+W1[N_layers][3]*(W1[0][0,0]*W1[0][1,0])*2
        c_rho=c_rho+W2[N_layers][0]*4+W2[N_layers][3]*(W2[0][0,0]*W2[0][1,0])*4
        c_rho=c_rho+W3[N_layers][0]*8+W3[N_layers][3]*(W3[0][0,0]*W3[0][1,0])*8      
        c_rho=c_rho+W4[N_layers][0]*16+W4[N_layers][3]*(W4[0][0,0]*W4[0][1,0])*16      
        c_rho=c_rho+W5[N_layers][0]*32+W5[N_layers][3]*(W5[0][0,0]*W5[0][1,0])*32
                
        c_v_rho_x=W0[N_layers][1]+W0[N_layers][3]*(W0[0][0,0]*W0[0][0,1]+W0[0][0,1]*W0[0][1,0])
        c_v_rho_x=c_v_rho_x+(W1[N_layers][1]+W1[N_layers][3]*(W1[0][0,0]*W1[0][0,1]+W1[0][0,1]*W1[0][1,0]))*2
        c_v_rho_x=c_v_rho_x+(W2[N_layers][1]+W2[N_layers][3]*(W2[0][0,0]*W2[0][0,1]+W2[0][0,1]*W2[0][1,0]))*4
        c_v_rho_x=c_v_rho_x+(W3[N_layers][1]+W3[N_layers][3]*(W3[0][0,0]*W3[0][0,1]+W3[0][0,1]*W3[0][1,0]))*8
        c_v_rho_x=c_v_rho_x+(W4[N_layers][1]+W4[N_layers][3]*(W4[0][0,0]*W4[0][0,1]+W4[0][0,1]*W4[0][1,0]))*16
        c_v_rho_x=c_v_rho_x+(W5[N_layers][1]+W5[N_layers][3]*(W5[0][0,0]*W5[0][0,1]+W5[0][0,1]*W5[0][1,0]))*32
        
        c_v2_rho_xx=W0[N_layers][3]*(W0[0][0,1]*W0[0][1,1])
        c_v2_rho_xx=c_v2_rho_xx+(W1[N_layers][3]*(W1[0][0,1]*W1[0][1,1]))*2
        c_v2_rho_xx=c_v2_rho_xx+(W2[N_layers][3]*(W2[0][0,1]*W2[0][1,1]))*4
        c_v2_rho_xx=c_v2_rho_xx+(W3[N_layers][3]*(W3[0][0,1]*W3[0][1,1]))*8
        c_v2_rho_xx=c_v2_rho_xx+(W4[N_layers][3]*(W4[0][0,1]*W4[0][1,1]))*16
        c_v2_rho_xx=c_v2_rho_xx+(W5[N_layers][3]*(W5[0][0,1]*W5[0][1,1]))*32

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

def coeffsg(W0,W1,W2,W3,W4,W5,N_layers,eps):
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
        c_g=c_g+(W1[1][0]+W1[N_layers][3]*coef_g_b1(W1,N_layers))*2
        c_g=c_g+(W2[1][0]+W2[N_layers][3]*coef_g_b1(W2,N_layers))*4
        c_g=c_g+(W3[1][0]+W3[N_layers][3]*coef_g_b1(W3,N_layers))*8
        c_g=c_g+(W4[1][0]+W4[N_layers][3]*coef_g_b1(W4,N_layers))*16
        c_g=c_g+(W5[1][0]+W5[N_layers][3]*coef_g_b1(W5,N_layers))*32
        
        c_v_g_x=W0[1][1]+W0[N_layers][3]*coef_adv_b1(W0,N_layers)
        c_v_g_x=c_v_g_x+(W1[1][1]+W1[N_layers][3]*coef_adv_b1(W1,N_layers))*2
        c_v_g_x=c_v_g_x+(W2[1][1]+W2[N_layers][3]*coef_adv_b1(W2,N_layers))*4
        c_v_g_x=c_v_g_x+(W3[1][1]+W3[N_layers][3]*coef_adv_b1(W3,N_layers))*8
        c_v_g_x=c_v_g_x+(W4[1][1]+W4[N_layers][3]*coef_adv_b1(W4,N_layers))*16
        c_v_g_x=c_v_g_x+(W5[1][1]+W5[N_layers][3]*coef_adv_b1(W5,N_layers))*32
        
        pvgx=W0[N_layers][3]*coef_Padv_b1(W0,N_layers)
        pvgx=pvgx+(W1[N_layers][3]*coef_Padv_b1(W1,N_layers))*2
        pvgx=pvgx+(W2[N_layers][3]*coef_Padv_b1(W2,N_layers))*4
        pvgx=pvgx+(W3[N_layers][3]*coef_Padv_b1(W3,N_layers))*8
        pvgx=pvgx+(W4[N_layers][3]*coef_Padv_b1(W4,N_layers))*16
        pvgx=pvgx+(W5[N_layers][3]*coef_Padv_b1(W5,N_layers))*32
        
        c_v_2_g_xx=W0[N_layers][3]*(W0[0][0,1]*W0[0][1,1])
        c_v_2_g_xx=c_v_2_g_xx+(W1[N_layers][3]*(W1[0][0,1]*W1[0][1,1]))*2
        c_v_2_g_xx=c_v_2_g_xx+(W2[N_layers][3]*(W2[0][0,1]*W2[0][1,1]))*4
        c_v_2_g_xx=c_v_2_g_xx+(W3[N_layers][3]*(W3[0][0,1]*W3[0][1,1]))*8
        c_v_2_g_xx=c_v_2_g_xx+(W4[N_layers][3]*(W4[0][0,1]*W4[0][1,1]))*16
        c_v_2_g_xx=c_v_2_g_xx+(W5[N_layers][3]*(W5[0][0,1]*W5[0][1,1]))*32
        

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
    def __init__(self,g,rho_extend,n_dat,N_layers,Nx,dt,hx,max_epsilon,min_epsilon,Max_orders,Losss):
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
        
        self.g_tf = tf.placeholder(tf.float32, shape=[Nv,Nx,n_dat])
        self.rho_extend_tf = tf.placeholder(tf.float32, shape=[Nv,Nx,n_dat])

        self.w_110,self.w_120,self.w_111,self.w_121,self.w_112,self.w_122,self.w_113,self.w_123,self.w_114,self.w_124,self.w_115,self.w_125,self.w_eps, self.wstiffg2, self.sigmaS, self.sigmaA= self.initialize_NNg(N_layers)
        
        # tf Graphs
        self.K32,self.K161,self.K162,self.K81,self.K82,self.K41,self.K42,self.K21,self.K22,self.K11,self.K12,self.w_110_tf, self.w_120_tf,self.w_111_tf, self.w_121_tf,self.w_112_tf, self.w_122_tf,self.w_113_tf, self.w_123_tf,self.w_114_tf, self.w_124_tf,self.w_115_tf, self.w_125_tf,self.w_eps_tf,self.wstiff_g2_tf,self.g_innerproduct,self.rho_pred,self.eps_pred,self.sigmaS_tf,self.sigmaA_tf = self.net_uv(self.g_tf,self.rho_extend_tf)

#        self.K32,self.K161,self.K162,self.K81,self.K82,self.w_110_tf, self.w_120_tf,self.w_111_tf, self.w_121_tf,self.w_112_tf, self.w_122_tf,self.w_113_tf, self.w_123_tf,self.w_114_tf, self.w_124_tf,self.w_115_tf, self.w_125_tf,self.w_eps_tf,self.wstiff_g2_tf,self.g_innerproduct,self.rho_pred,self.eps_pred,self.sigmaS_tf,self.sigmaA_tf = self.net_uv(self.g_tf,self.rho_extend_tf)

        # Loss

        gamma=1
#        gamma=0.0001
#        self.loss = tf.reduce_mean(tf.abs(self.g_pred))
        self.loss = tf.reduce_mean(tf.abs(self.K32))+tf.reduce_mean(tf.abs(self.K161))+tf.reduce_mean(tf.abs(self.K162))+tf.reduce_mean(tf.abs(self.K81))+tf.reduce_mean(tf.abs(self.K82))+tf.reduce_mean(tf.abs(self.K41))+tf.reduce_mean(tf.abs(self.K42))+tf.reduce_mean(tf.abs(self.K21))+tf.reduce_mean(tf.abs(self.K22))+tf.reduce_mean(tf.abs(self.K11))+tf.reduce_mean(tf.abs(self.K12))
#        tf.reduce_mean(tf.abs(self.g_pred11))+tf.reduce_mean(tf.abs(self.g_pred12))#+gamma*tf.reduce_mean(tf.abs(self.g_innerproduct))# +gamma*(tf.reduce_mean(tf.abs(self.w_110_tf[0]))+tf.reduce_mean(tf.abs(self.w_120_tf[0]))+tf.reduce_mean(tf.abs(self.w_111_tf[0]))+tf.reduce_mean(tf.abs(self.w_121_tf[0]))+tf.reduce_mean(tf.abs(self.w_112_tf[0]))+tf.reduce_mean(tf.abs(self.w_122_tf[0])))+gamma*(tf.reduce_mean(tf.abs(self.w_110_tf[1]))+tf.reduce_mean(tf.abs(self.w_120_tf[1]))+tf.reduce_mean(tf.abs(self.w_111_tf[1]))+tf.reduce_mean(tf.abs(self.w_121_tf[1]))+tf.reduce_mean(tf.abs(self.w_112_tf[1]))+tf.reduce_mean(tf.abs(self.w_122_tf[1])))
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

            w_113=[]
            w_123=[]
            
            w_114=[]
            w_124=[]
            
            w_115=[]
            w_125=[]

            initializeit=0.5
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

            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_113.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_113.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_123.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_123.append(W_last_g)            

            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_114.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_114.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_124.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_124.append(W_last_g)            

            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_115.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_115.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_125.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_125.append(W_last_g)            


        elif Max_orders==2:
            w_110=[]
            w_120=[]
            w_111=[]
            w_121=[]
            w_112=[]
            w_122=[]
            w_113=[]
            w_123=[]
            w_114=[]
            w_124=[]
            w_115=[]
            w_125=[]

            initializeit=0.5
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

            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_113.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_113.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_123.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_123.append(W_last_g)            

            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_114.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_114.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_124.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_124.append(W_last_g)

            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_115.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_115.append(W_last_g)
            for I in range(0,N_layers):
                W=initializeit*tf.Variable(tf.ones([2,3+I],dtype=tf.float32),dtype=tf.float32)
                w_125.append(W)
            W_last_g=initializeit*tf.Variable(tf.ones([3+N_layers], dtype=tf.float32), dtype=tf.float32)
            w_125.append(W_last_g)            

        else:
            print('Not implemented')
        w_eps=tf.Variable(1.0,dtype=tf.float32)
        wstiffg2=initializeit*tf.Variable(1.0,dtype=tf.float32)
        sigmaS=initializeit*tf.Variable(1.0,dtype=tf.float32)        
        sigmaA=initializeit*tf.Variable(1.0,dtype=tf.float32)
        return w_110,w_120,w_111,w_121,w_112,w_122,w_113,w_123,w_114,w_124,w_115,w_125,w_eps,wstiffg2,sigmaS,sigmaA


    def RNNg(self,g,N_layers,w0,eps):

        W0=w0[0]
        WL0=w0[N_layers]
        
        Linear_As0=(WL0[0])*g+WL0[1]*self.advection(g,vextend)+WL0[2]*self.projection(g,wvextend)
        Right0=W0[1,0]*g+W0[1,1]*self.advection(g,vextend)+W0[1,2]*self.projection(g,wvextend)
        B0=W0[0,0]*Right0+W0[0,1]*self.advection(Right0,vextend)+W0[0,2]*self.projection(Right0,wvextend)

        F_rhs_g=eps*Linear_As0+eps*WL0[3]*B0

        """
        W0=w0[0]
        WL0=w0[N_layers]
        W1=w1[0]
        WL1=w1[N_layers]
        W2=w2[0]
        WL2=w2[N_layers]
        W3=w3[0]
        WL3=w3[N_layers]
        W4=w4[0]
        WL4=w4[N_layers]
        W5=w5[0]
        WL5=w5[N_layers]
            
        Linear_As0=(WL0[0])*g+WL0[1]*self.advection(g,vextend)+WL0[2]*self.projection(g,wvextend)
        Right0=W0[1,0]*g+W0[1,1]*self.advection(g,vextend)+W0[1,2]*self.projection(g,wvextend)
        B0=W0[0,0]*Right0+W0[0,1]*self.advection(Right0,vextend)+W0[0,2]*self.projection(Right0,wvextend)

        Linear_As1=(WL1[0])*g+WL1[1]*self.advection(g,vextend)+WL1[2]*self.projection(g,wvextend)
        Right1=W1[1,0]*g+W1[1,1]*self.advection(g,vextend)+W1[1,2]*self.projection(g,wvextend)
        B1=W1[0,0]*Right1+W1[0,1]*self.advection(Right1,vextend)+W1[0,2]*self.projection(Right1,wvextend)

        Linear_As2=(WL2[0])*g+WL2[1]*self.advection(g,vextend)+WL2[2]*self.projection(g,wvextend)
        Right2=W2[1,0]*g+W2[1,1]*self.advection(g,vextend)+W2[1,2]*self.projection(g,wvextend)
        B2=W2[0,0]*Right2+W2[0,1]*self.advection(Right2,vextend)+W2[0,2]*self.projection(Right2,wvextend)

        Linear_As3=(WL3[0])*g+WL3[1]*self.advection(g,vextend)+WL3[2]*self.projection(g,wvextend)
        Right3=W3[1,0]*g+W3[1,1]*self.advection(g,vextend)+W3[1,2]*self.projection(g,wvextend)
        B3=W3[0,0]*Right3+W3[0,1]*self.advection(Right3,vextend)+W3[0,2]*self.projection(Right3,wvextend)

        Linear_As4=(WL4[0])*g+WL4[1]*self.advection(g,vextend)+WL4[2]*self.projection(g,wvextend)
        Right4=W4[1,0]*g+W4[1,1]*self.advection(g,vextend)+W4[1,2]*self.projection(g,wvextend)
        B4=W4[0,0]*Right4+W4[0,1]*self.advection(Right4,vextend)+W4[0,2]*self.projection(Right4,wvextend)

        Linear_As5=(WL5[0])*g+WL5[1]*self.advection(g,vextend)+WL5[2]*self.projection(g,wvextend)
        Right5=W5[1,0]*g+W5[1,1]*self.advection(g,vextend)+W5[1,2]*self.projection(g,wvextend)
        B5=W5[0,0]*Right5+W5[0,1]*self.advection(Right5,vextend)+W5[0,2]*self.projection(Right5,wvextend)

        F_rhs_g=Linear_As0+2*Linear_As1+4*Linear_As2+8*Linear_As3+16*Linear_As4+32*Linear_As5
        F_rhs_g=F_rhs_g+WL0[3]*B0+2*WL1[3]*B1+4*WL2[3]*B2+8*WL3[3]*B3+16*WL4[3]*B4+32*WL5[3]*B5
        """
        return F_rhs_g,w0


    def RNNrho(self,rho_extend,N_layers,w0,eps):
        W0=w0[0]
        WL0=w0[N_layers]
        
        Linear_As0=WL0[0]*rho_extend+WL0[1]*self.advection(rho_extend,vextend)
        Right0=W0[1,0]*rho_extend+W0[1,1]*self.advection(rho_extend,vextend)
        B0=W0[0,0]*Right0+W0[0,1]*self.advection(Right0,vextend)

        F_rhs_rho=eps*Linear_As0+eps*WL0[3]*B0

        """
        W0=w0[0]
        WL0=w0[N_layers]
        W1=w1[0]
        WL1=w1[N_layers]
        W2=w2[0]
        WL2=w2[N_layers]
        W3=w3[0]
        WL3=w3[N_layers]
        W4=w4[0]
        WL4=w4[N_layers]
        W5=w5[0]
        WL5=w5[N_layers]
            
        Linear_As0=WL0[0]*rho_extend+WL0[1]*self.advection(rho_extend,vextend)
        Right0=W0[1,0]*rho_extend+W0[1,1]*self.advection(rho_extend,vextend)
        B0=W0[0,0]*Right0+W0[0,1]*self.advection(Right0,vextend)

        Linear_As1=WL1[0]*rho_extend+WL1[1]*self.advection(rho_extend,vextend)
        Right1=W1[1,0]*rho_extend+W1[1,1]*self.advection(rho_extend,vextend)
        B1=W1[0,0]*Right0+W1[0,1]*self.advection(Right1,vextend)
        
        Linear_As2=WL2[0]*rho_extend+WL2[1]*self.advection(rho_extend,vextend)
        Right2=W2[1,0]*rho_extend+W2[1,1]*self.advection(rho_extend,vextend)
        B2=W2[0,0]*Right2+W2[0,1]*self.advection(Right2,vextend)

        Linear_As3=WL3[0]*rho_extend+WL3[1]*self.advection(rho_extend,vextend)
        Right3=W3[1,0]*rho_extend+W3[1,1]*self.advection(rho_extend,vextend)
        B3=W3[0,0]*Right3+W3[0,1]*self.advection(Right3,vextend)

        Linear_As4=WL4[0]*rho_extend+WL4[1]*self.advection(rho_extend,vextend)
        Right4=W4[1,0]*rho_extend+W4[1,1]*self.advection(rho_extend,vextend)
        B4=W4[0,0]*Right4+W4[0,1]*self.advection(Right4,vextend)

        Linear_As5=WL5[0]*rho_extend+WL5[1]*self.advection(rho_extend,vextend)
        Right5=W5[1,0]*rho_extend+W5[1,1]*self.advection(rho_extend,vextend)
        B5=W5[0,0]*Right5+W5[0,1]*self.advection(Right5,vextend)


        F_rhs_rho=Linear_As0+2*Linear_As1+4*Linear_As2+8*Linear_As3+16*Linear_As4+32*Linear_As5
        F_rhs_rho=F_rhs_rho+WL0[3]*B0+2*WL1[3]*B1+4*WL2[3]*B2+8*WL3[3]*B3+16*WL4[3]*B4+32*WL5[3]*B5
        """ 
        
        return F_rhs_rho,w0



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

#        F110, w_110,w_111,w_112,w_113,w_114,w_115 = self.RNNg(g,self.N_layers,self.w_110,self.w_111,self.w_112,self.w_113,self.w_114,self.w_115,eps)
#        F120, w_120,w_121,w_122,w_123,w_124,w_125 = self.RNNrho(rho_extend,self.N_layers,self.w_120,self.w_121,self.w_122,self.w_123,self.w_124,self.w_125,eps)

        F110, w_110 = self.RNNg(g,self.N_layers,self.w_110,1)
        F120, w_120 = self.RNNrho(rho_extend,self.N_layers,self.w_120,1)

        F111, w_111 = self.RNNg(g,self.N_layers,self.w_111,2)
        F121, w_121 = self.RNNrho(rho_extend,self.N_layers,self.w_121,2)

        F112, w_112 = self.RNNg(g,self.N_layers,self.w_112,4)
        F122, w_122 = self.RNNrho(rho_extend,self.N_layers,self.w_122,4)

        F113, w_113 = self.RNNg(g,self.N_layers,self.w_113,8)
        F123, w_123 = self.RNNrho(rho_extend,self.N_layers,self.w_123,8)

        F114, w_114 = self.RNNg(g,self.N_layers,self.w_114,16)
        F124, w_124 = self.RNNrho(rho_extend,self.N_layers,self.w_124,16)

        F115, w_115 = self.RNNg(g,self.N_layers,self.w_115,32)
        F125, w_125 = self.RNNrho(rho_extend,self.N_layers,self.w_125,32)
#        w_111=self.w_111
#        w_112=self.w_112
#        w_121=self.w_121
#        w_122=self.w_122

#        temp=F112+F122
        wstiffg2=self.wstiffg2
        sigmaS=self.sigmaS
        sigmaA=self.sigmaA
#        Hierarchy of propagation
#        K_g11=g[:,:,27]-g[:,:,0]-27*dt*(temp[:,:,0]) 
#        K_g12=g[:,:,54]-g[:,:,27]-27*dt*(temp[:,:,27])

        temp0=F110+F120
        temp1=F111+F121
        temp2=F112+F122
        temp3=F113+F123
        temp4=F114+F124
        temp5=F115+F125
        
        K32=g[:,:,32]-g[:,:,0]-1*dt*(temp5[:,:,0])
        
        K161=g[:,:,16]-g[:,:,0]-1*dt*(temp4[:,:,0])
        K162=g[:,:,32]-g[:,:,16]-1*dt*(temp4[:,:,16])
        
        K81=g[:,:,8]-g[:,:,0]-1*dt*(temp3[:,:,0])
        K82=g[:,:,16]-g[:,:,8]-1*dt*(temp3[:,:,8])

        K41=g[:,:,4]-g[:,:,0]-1*dt*(temp2[:,:,0])
        K42=g[:,:,8]-g[:,:,4]-1*dt*(temp2[:,:,4])

        K21=g[:,:,2]-g[:,:,0]-1*dt*(temp1[:,:,0])
        K22=g[:,:,4]-g[:,:,2]-1*dt*(temp1[:,:,2])

        K11=g[:,:,1]-g[:,:,0]-1*dt*(temp0[:,:,0])
        K12=g[:,:,2]-g[:,:,1]-1*dt*(temp0[:,:,1])
#        K_g11=g[:,:,4]-g[:,:,2]-dt*(temp[:,:,0]) 
#        K_g12=g[:,:,2]-g[:,:,1]-dt*(temp[:,:,1])
#        K_g=g[:,:,1:n_dat-1]-g[:,:,0:n_dat-2]-dt*(temp[:,:,0:n_dat-2]) 
#        K_g=g[:,:,1:n_dat-1]*(1+sigmaS*dt/eps**2)-g[:,:,0:n_dat-2]*(1-sigmaA*dt)-dt*temp[:,:,0:n_dat-2]  #This is IMEX1
        K_rh=0*g
        tmepg=0*g[:,:,0:n_dat-2]
#        tmepg=(g[:,:,0:n_dat-2]*(1-sigmaA*dt)+dt*temp[:,:,0:n_dat-2])/(1+sigmaS*dt/eps**2)
        Frhs_innerproduct=0.0
#        Frhs_innerproduct=tf.math.reduce_sum(wvextend[:,:,1:n_dat-1]*tmepg/2.0,0)
#        return K32,K161,K162,K81,K82,w_110,w_120,w_111,w_121,w_112,w_122,w_eps,wstiffg2,Frhs_innerproduct,K_rh,eps,sigmaS,sigmaA
        return K32,K161,K162,K81,K82,K41,K42,K21,K22,K11,K12,w_110,w_120,w_111,w_121,w_112,w_122,w_113,w_123,w_114,w_124,w_115,w_125,w_eps,wstiffg2,Frhs_innerproduct,K_rh,eps,sigmaS,sigmaA
    
    
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
        w_113=self.sess.run(self.w_113_tf)
        w_123=self.sess.run(self.w_123_tf)
        w_114=self.sess.run(self.w_114_tf)
        w_124=self.sess.run(self.w_124_tf)
        w_115=self.sess.run(self.w_115_tf)
        w_125=self.sess.run(self.w_125_tf)

        w_eps=self.sess.run(self.w_eps_tf)
        wstiff_g2=self.sess.run(self.wstiff_g2_tf)
        sigmaS=self.sess.run(self.sigmaS_tf)
        sigmaA=self.sess.run(self.sigmaA_tf)
#        gg=self.sess.run(self.g_pred)
        
        return w_110,w_120,w_111,w_121,w_112,w_122,w_113,w_123,w_114,w_124,w_115,w_125,w_eps,wstiff_g2,sigmaS,sigmaA

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

    Nsteps=12000
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

    modelrho = PDE_CLASSIFY(g,rho_extend,n_dat,N_layers,Nx,dt,hx,max_epsilon,min_epsilon,Max_orders,Losss)

    start_time = time.time()
    modelrho.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    w_110,w_120,w_111,w_121,w_112,w_122,w_113,w_123,w_114,w_124,w_115,w_125,w_eps,wstiffg2,sigmaS,sigmaA= modelrho.predictrho(g)

    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2, label='Log of Loss vs Number of Iterations')
#    plt.show()
#    plt.legend()
#    fig.savefig('LossBDF2_example1.png')
#    savemat('Loss1.mat',{'L_domain':Losss_domain, 'Loss_y':LLosss})





    u_pred_ndat=np.zeros([Nx,1],dtype=float)

    eps=(np.tanh(w_eps) + 1)*(max_epsilon-min_epsilon)/2+min_epsilon


    c_g,c_v_g_x,pvgx,c_v_2_g_xx=coeffsg(w_110,w_111,w_112,w_113,w_114,w_115,N_layers,eps)
    c_rho,c_v_rho_x,c_v2_rho_xx=coeffsrho(w_120,w_121,w_122,w_123,w_124,w_125,N_layers,eps)

    cg2=c_g
    c_g=c_g
#    c_g=c_g-sigmaS/eps**2-sigmaA

    print('The learned IMEX1 G-equation is:')
    print(c_g,'g +',c_v_g_x,'v*g_x+',pvgx,'<vg_x>')
    print(c_rho,'rho +',c_v_rho_x,'v*rho_x+')

    ep=np.int(16)
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










#    xsu=np.zeros([Nx],dtype=float)
#    for I in range(0,Nx):
#        xsu[I]=1/(Nx-1)*(I-1)
    
#    fig = plt.figure()
#    plt.scatter(xsu,projection_term[:,0])






