

import sys
sys.path.insert(0, '../../Utilities/')
import scipy.io
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow_probability as tfp
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from scipy import interpolate



#Set random seed.
global_seed = 42
N_chains = 5
np.random.seed(global_seed)
seeds = np.random.randint(0, 429, size=N_chains)


def coeffsg(ww,N_layers,eps):
    c_g=0
    pvgx=0
    c_v_2_g_xx=0
    if N_layers==1:
        for I in range(0,Max_orders):
            c_g=c_g+0.0
            c_v_g_x=(ww[:,0]+ww[:,1])
            pvgx=pvgx
            c_v_2_g_xx=0
    if N_layers==2:
        W1=ww[0]
        W2=ww[1]
        for I in range(0,Max_orders+1):
            c_g=c_g+0.0
            c_v_g_x=c_v_g_x+(W1[:,0]+W1[:,1])/eps**I
            pvgx=pvgx+(W2[:,4]+W2[:,5]+W2[:,6]+W2[:,7:,])*(W1[:,0]+W1[:,1])/eps**I
            c_v_2_g_xx=0
    elif N_layers==3:
        W1=ww[0]
        W2=ww[1]
        W3=ww[2]
        for I in range(0,Max_orders+1):
            c_g=c_g+0.0
            c_v_g_x=c_v_g_x+(W1[0,I]+W1[1,I])/eps**I
            temp=(W2[4,I]+W2[5,I]+W2[6,I]+W2[7,I])*(W1[0,I]+W1[1,I])/eps**I
            pvgx=pvgx+temp+temp*(W3[8,I]+W3[9,I]+W3[10,I]+W3[11,I]+W3[12,I]+W3[13,I]+W3[14,I]+W3[15,I])
            c_v_2_g_xx=0
    return c_g,c_v_g_x,pvgx,c_v_2_g_xx


def coeffsrho(ww,N_layers,eps):
    #For now N_layers is 1
    c_rho=0
    c_v_rho_x=0
    pvrhox=0
    if N_layers==1:
        for I in range(0,Max_orders):
            c_rho=c_rho+0.0
            c_v_rho_x=(ww[:,0]+ww[:,1])
            pvrhox=pvrhox
            c_v_2_rho_xx=0
    if N_layers==2:
        W=ww[0]
        for I in range(0,Max_orders+1):
            c_rho=0.0
            c_v_rho_x=c_v_rho_x+(W[0,I]+W[1,I])/eps**I
            c_v2_rho_xx=0.0
    elif N_layers==3:
        W1=ww[0]
        W2=ww[0]
        for I in range(0,Max_orders+1):
            c_rho=0.0
            c_v_rho_x=c_v_rho_x+(W1[0,I]+W1[1,I])/eps**I
            c_v2_rho_xx=0.0

    return c_rho,c_v_rho_x,pvrhox


class PDE_CLASSIFY:
    def __init__(self,rho_extend,g,n_dat,N_layers,Nv,Nx,dt,hx,vextend,wvextend,max_epsilon,min_epsilon,Max_orders,Losss,x1,x2,x3,x4,x5):
        self.rho_extend = rho_extend
        self.vextend = vextend
        self.wvextend = wvextend
        self.g = g
        self.Losss=Losss

        self.max_epsilon=max_epsilon
        self.min_epsilon=min_epsilon
        self.Max_orders=Max_orders

        self.Nv=Nv
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

        self.weights_g,self.weights_rho,self.w_eps, self.sigmaS, self.sigmaA, self.a0,self.b0,self.c0,self.a1,self.b1,self.c1,self.a2,self.b2,self.c2,self.a3,self.b3,self.c3,self.a4,self.b4,self.c4, self.a0r,self.b0r,self.c0r,self.a1r,self.b1r,self.c1r,self.a2r,self.b2r,self.c2r,self.a3r,self.b3r,self.c3r,self.a4r,self.b4r,self.c4r = self.initialize_NNg(N_layers,Max_orders)

        self.rho_extend_tf = tf.placeholder(tf.float32, shape=[None, None, self.rho_extend.shape[2]])

        self.g_tf = tf.placeholder(tf.float32, shape=[None, None, self.g.shape[2]])

        self.vextend_tf = tf.placeholder(tf.float32, shape=[None, None, self.vextend.shape[2]])
        self.wvextend_tf = tf.placeholder(tf.float32, shape=[None, None, self.wvextend.shape[2]])

        # tf GraphsK_g,weights_g,weights_rho,w_eps,sigmaS,sigmaA

        self.g_pred,self.weights_g_tf, self.weights_rho_tf,self.w_eps_tf,self.sigmaS_tf,self.sigmaA_tf, self.wS_tf,self.cont0,self.cont1,self.cont2,self.cont3,self.cont4,self.diff1,self.diff2,self.diff3,self.diff4, self.wSr_tf,self.cont0r,self.cont1r,self.cont2r,self.cont3r,self.cont4r,self.diff1r,self.diff2r,self.diff3r,self.diff4r  = self.net_uv(self.rho_extend_tf,self.g_tf,self.vextend_tf,self.wvextend_tf)
        # Loss

        gamma=0.000001
#        self.loss = tf.reduce_mean(tf.abs(self.g_pred))+gamma*(tf.reduce_mean(tf.abs(self.cont1[0]))+tf.reduce_mean(tf.abs(self.cont2[0]))+tf.reduce_mean(tf.abs(self.cont3[0]))+tf.reduce_mean(tf.abs(self.cont4[0])))#+tf.reduce_mean(tf.abs(self.diff1[0]))+tf.reduce_mean(tf.abs(self.diff2[0]))+tf.reduce_mean(tf.abs(self.diff3[0]))+tf.reduce_mean(tf.abs(self.diff4[0])))

        self.loss = tf.reduce_mean(tf.abs(self.g_pred))+gamma*(tf.reduce_mean(tf.abs(self.cont1[0]))+tf.reduce_mean(tf.abs(self.cont2[0]))+tf.reduce_mean(tf.abs(self.cont3[0]))+tf.reduce_mean(tf.abs(self.cont4[0])))   +gamma*(tf.reduce_mean(tf.abs(self.cont1r[0]))+tf.reduce_mean(tf.abs(self.cont2r[0]))+tf.reduce_mean(tf.abs(self.cont3r[0]))+tf.reduce_mean(tf.abs(self.cont4r[0])))


#        self.loss = tf.reduce_mean(tf.abs(self.g_pred))#+gamma*(tf.reduce_mean(tf.abs(self.weights_g_tf[0]))+tf.reduce_mean(tf.abs(self.weights_rho_tf[0])))+gamma*(tf.reduce_mean(tf.abs(self.weights_g_tf[1]))+tf.reduce_mean(tf.abs(self.weights_rho_tf[1])))+gamma*(tf.reduce_mean(tf.abs(self.weights_g_tf[2]))+tf.reduce_mean(tf.abs(self.weights_rho_tf[2])))

        #This is for 2-layers
#        self.loss = tf.reduce_mean(tf.abs(self.g_pred))+gamma*(tf.reduce_mean(tf.abs(self.weights_g_tf[0]))+tf.reduce_mean(tf.abs(self.weights_rho_tf[0])))+gamma*(tf.reduce_mean(tf.abs(self.weights_g_tf[1]))+tf.reduce_mean(tf.abs(self.weights_rho_tf[1])))

    # Optimizers
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')


        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)



        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def initialize_NNg(self,N_layers,Max_orders):

        weights_g=[]
        weights_rho=[]

        a0=[]
        b0=[]
        c0=[]

        a1=[]
        b1=[]
        c1=[]

        a2=[]
        b2=[]
        c2=[]

        a3=[]
        b3=[]
        c3=[]
        
        a4=[]
        b4=[]
        c4=[]







        a0r=[]
        b0r=[]
        c0r=[]

        a1r=[]
        b1r=[]
        c1r=[]

        a2r=[]
        b2r=[]
        c2r=[]

        a3r=[]
        b3r=[]
        c3r=[]
        
        a4r=[]
        b4r=[]
        c4r=[]






        
        initializeit=0.25
        init=np.zeros([N_layers],dtype=float)
        for I in range(0,N_layers):
            init[I]=abs(np.random.normal())
        init=np.flip(np.sort(init))
        init=np.float32(init)
        for I in range(0,N_layers):
            W=init[I]*tf.Variable(tf.ones([2**(I+2),Max_orders+1],dtype=tf.float32),dtype=tf.float32)
            weights_g.append(W)
            W=init[I]*tf.Variable(tf.ones([2**(I+2),Max_orders+1],dtype=tf.float32),dtype=tf.float32)
            weights_rho.append(W)
            
            t1=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t2=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t3=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)

            a0.append(t1)
            b0.append(t2)
            c0.append(t3)

            t4=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t5=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t6=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)


            a1.append(t4)
            b1.append(t5)
            c1.append(t6)

            t7=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t8=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t9=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)

            a2.append(t7)
            b2.append(t8)
            c2.append(t9)


            t10=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t11=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t12=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)

            a3.append(t10)
            b3.append(t11)
            c3.append(t12)

            t13=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t14=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t15=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)

            a4.append(t13)
            b4.append(t14)
            c4.append(t15)










            t16=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t17=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t18=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)

            a0r.append(t1)
            b0r.append(t17)
            c0r.append(t18)

            t19=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t20=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t21=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)


            a1r.append(t19)
            b1r.append(t20)
            c1r.append(t21)

            t22=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t23=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t24=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)

            a2r.append(t22)
            b2r.append(t23)
            c2r.append(t24)


            t25=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t26=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t27=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)

            a3r.append(t25)
            b3r.append(t26)
            c3r.append(t27)

            t28=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t29=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)
            t30=2*init[I]*tf.Variable(tf.ones([2**(I+2)],dtype=tf.float32),dtype=tf.float32)

            a4r.append(t28)
            b4r.append(t29)
            c4r.append(t30)









        w_eps=0.05*tf.Variable(1.0,dtype=tf.float32)
        sigmaS=initializeit*tf.Variable(1.0,dtype=tf.float32)
        sigmaA=initializeit*tf.Variable(1.0,dtype=tf.float32)

        return weights_g,weights_rho,w_eps,sigmaS,sigmaA,a0,b0,c0,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a0r,b0r,c0r,a1r,b1r,c1r,a2r,b2r,c2r,a3r,b3r,c3r,a4r,b4r,c4r



    def RNNg(self,g,w,Max_orders,N_layers,eps):
######################
### G-RNN
######################
        s=g*0.0
        for K in range(0,Max_orders):
            print('0')
            out=g*0.0
            F_rhs_g=g
            for I in range(0,N_layers):
                
                print('1')
                w1=tf.linalg.matmul(self.x1,self.b0)+self.c0
                w2=tf.linalg.matmul(self.x2,self.b1)+self.c1
                w3=tf.linalg.matmul(self.x3,self.b2)+self.c2
                w4=tf.linalg.matmul(self.x4,self.b3)+self.c3
                w5=tf.linalg.matmul(self.x5,self.b4)+self.c4
                print('2')
                #Set up continutity
                cont0=self.b0*self.x1[0]+self.c0-0.0
                cont1=self.b0*self.x2[0]+self.c0-(self.b1*self.x2[0]+self.c1)
                cont2=self.b1*self.x3[0]+self.c1-(self.b2*self.x3[0]+self.c2)
                cont3=self.b2*self.x4[0]+self.c2-(self.b3*self.x4[0]+self.c3)
                cont4=self.b3*self.x5[0]+self.c3-(self.b4*self.x5[0]+self.c4)
                print('3')
                #Set up continutity of derivatives
                diff1=0.0#2*self.a0*self.x2[0]+self.b0-(2*self.a1*self.x2[0]+self.b1)
                diff2=0.0#2*self.a1*self.x3[0]+self.b1-(2*self.a2*self.x3[0]+self.b2)
                diff3=0.0#2*self.a2*self.x4[0]+self.b2-(2*self.a3*self.x4[0]+self.b3)
                diff4=0.0#2*self.a3*self.x5[0]+self.b3-(2*self.a4*self.x5[0]+self.b4)
                print('4')                
                wS=tf.concat([w1,w2],0)
                wS=tf.concat([wS,w3],0)
                wS=tf.concat([wS,w4],0)
                wS=tf.concat([wS,w5],0)
                print('5')
                

                """
                print('1')
                w1=tf.linalg.matmul(self.x1**2,self.a0)+tf.linalg.matmul(self.x1,self.b0)+self.c0
                w2=tf.linalg.matmul(self.x2**2,self.a1)+tf.linalg.matmul(self.x2,self.b1)+self.c1
                w3=tf.linalg.matmul(self.x3**2,self.a2)+tf.linalg.matmul(self.x3,self.b2)+self.c2
                w4=tf.linalg.matmul(self.x4**2,self.a3)+tf.linalg.matmul(self.x4,self.b3)+self.c3
                w5=tf.linalg.matmul(self.x5**2,self.a4)+tf.linalg.matmul(self.x5,self.b4)+self.c4
                print('2')
                #Set up continutity
                cont0=self.a0*self.x1[0]**2+self.b0*self.x1[0]+self.c0-0.0
                cont1=self.a0*self.x2[0]**2+self.b0*self.x2[0]+self.c0-(self.a1*self.x2[0]**2+self.b1*self.x2[0]+self.c1)
                cont2=self.a1*self.x3[0]**2+self.b1*self.x3[0]+self.c1-(self.a2*self.x3[0]**2+self.b2*self.x3[0]+self.c2)
                cont3=self.a2*self.x4[0]**2+self.b2*self.x4[0]+self.c2-(self.a3*self.x4[0]**2+self.b3*self.x4[0]+self.c3)
                cont4=self.a3*self.x5[0]**2+self.b3*self.x5[0]+self.c3-(self.a4*self.x5[0]**2+self.b4*self.x5[0]+self.c4)
                print('3')
                #Set up continutity of derivatives
                diff1=2*self.a0*self.x2[0]+self.b0-(2*self.a1*self.x2[0]+self.b1)
                diff2=2*self.a1*self.x3[0]+self.b1-(2*self.a2*self.x3[0]+self.b2)
                diff3=2*self.a2*self.x4[0]+self.b2-(2*self.a3*self.x4[0]+self.b3)
                diff4=2*self.a3*self.x5[0]+self.b3-(2*self.a4*self.x5[0]+self.b4)
                print('4')                
                wS=tf.concat([w1,w2],0)
                wS=tf.concat([wS,w3],0)
                wS=tf.concat([wS,w4],0)
                wS=tf.concat([wS,w5],0)
                print('5')
                """





                s1=tf.reduce_sum(wS[:,0:int((2**(I+2))/2)],1)
                s2=tf.reduce_sum(wS[:,int((2**(I+2))/2):int(2**(I+2))],1)
                s1=tf.expand_dims(s1,1)
                s2=tf.expand_dims(s2,1)

                Stemp1=tf.linalg.matmul(s1,tf.ones([1,n_dat],dtype=tf.float32))
                Stemp2=tf.linalg.matmul(s2,tf.ones([1,n_dat],dtype=tf.float32))

                      
                
                wS_extend1=self.extend_2d_to_3d(Stemp1)
                wS_extend2=self.extend_2d_to_3d(Stemp2)

                F_rhs_g=wS_extend1*self.advection(F_rhs_g,vextend)+wS_extend2*self.projection(F_rhs_g,wvextend)
                out=F_rhs_g+out
                
            s=out/eps**K+s


        """
        F_rhs_g=g
        out=g*0.0
        for I in range(0,N_layers):
            W=w[I]
            s1=tf.reduce_sum(W[0:int((2**(I+2))/2),0])
            s2=tf.reduce_sum(W[int((2**(I+2))/2):int(2**(I+2)),0])
            F_rhs_g=s1*self.advection(F_rhs_g,vextend)+s2*self.projection(F_rhs_g,wvextend)
            out=F_rhs_g+out
        """
        return s,w,wS,cont0,cont1,cont2,cont3,cont4,diff1,diff2,diff3,diff4

    def RNNrho(self,rho_extend,w,Max_orders,N_layers,eps):
###################   
### RHO RNN
###################
        s=rho_extend*0.0
        for K in range(0,Max_orders):
            print('0')
            out=rho_extend*0.0
            F_rhs_rho=rho_extend
            for I in range(0,N_layers):

                print('1')
                w1=tf.linalg.matmul(self.x1,self.b0r)+self.c0r
                w2=tf.linalg.matmul(self.x2,self.b1r)+self.c1r
                w3=tf.linalg.matmul(self.x3,self.b2r)+self.c2r
                w4=tf.linalg.matmul(self.x4,self.b3r)+self.c3r
                w5=tf.linalg.matmul(self.x5,self.b4r)+self.c4r
                print('2')
                #Set up continutity
                cont0=self.b0r*self.x1[0]+self.c0r-0.0
                cont1=self.b0r*self.x2[0]+self.c0r-(self.b1r*self.x2[0]+self.c1r)
                cont2=self.b1r*self.x3[0]+self.c1r-(self.b2r*self.x3[0]+self.c2r)
                cont3=self.b2r*self.x4[0]+self.c2r-(self.b3r*self.x4[0]+self.c3r)
                cont4=self.b3r*self.x5[0]+self.c3r-(self.b4r*self.x5[0]+self.c4r)
                print('3')
                #Set up continutity of derivatives
                diff1=0.0#2*self.a0*self.x2[0]+self.b0-(2*self.a1*self.x2[0]+self.b1)
                diff2=0.0#2*self.a1*self.x3[0]+self.b1-(2*self.a2*self.x3[0]+self.b2)
                diff3=0.0#2*self.a2*self.x4[0]+self.b2-(2*self.a3*self.x4[0]+self.b3)
                diff4=0.0#2*self.a3*self.x5[0]+self.b3-(2*self.a4*self.x5[0]+self.b4)
                print('4')                
                wS=tf.concat([w1,w2],0)
                wS=tf.concat([wS,w3],0)
                wS=tf.concat([wS,w4],0)
                wS=tf.concat([wS,w5],0)
                print('5')


                s1=tf.reduce_sum(wS[:,0:int((2**(I+2))/2)],1)
                s2=tf.reduce_sum(wS[:,int((2**(I+2))/2):int(2**(I+2))],1)
                s1=tf.expand_dims(s1,1)
                s2=tf.expand_dims(s2,1)

                Stemp1=tf.linalg.matmul(s1,tf.ones([1,n_dat],dtype=tf.float32))
                Stemp2=tf.linalg.matmul(s2,tf.ones([1,n_dat],dtype=tf.float32))

                wS_extend1=self.extend_2d_to_3d(Stemp1)
                wS_extend2=self.extend_2d_to_3d(Stemp2)

                F_rhs_rho=wS_extend1*self.advection(F_rhs_rho,vextend)+wS_extend2*self.projection(F_rhs_rho,wvextend)
                out=F_rhs_rho+out
                
            s=out/eps**K+s


        """
        s=rho_extend*0.0
        for K in range(0,Max_orders):
            F_rhs_r=rho_extend
            out=rho_extend*0.0
            for I in range(0,N_layers):
                W=w[I]
                s1=tf.reduce_sum(W[0:int((2**(I+2))/2),K])
                s2=tf.reduce_sum(W[int((2**(I+2))/2):int(2**(I+2)),K])
                #Projection does not make sense when projection applied to rho.
                F_rhs_r=s1*self.advection(F_rhs_r,vextend)#+s2*self.projection(F_rhs_r,wvextend)
                out=F_rhs_r+out
            s=out/eps**K+s
        """
        """
        F_rhs_r=rho_extend
        out=rho_extend*0.0
        for I in range(0,N_layers):
            W=w[I]
            s1=tf.reduce_sum(W[0:int((2**(I+2))/2),0])
            s2=tf.reduce_sum(W[int((2**(I+2))/2):int(2**(I+2)),0])
            #Projection does not make sense when projection applied to rho.
            F_rhs_r=s1*self.advection(F_rhs_r,vextend)#+s2*self.projection(F_rhs_r,wvextend)
            out=F_rhs_r+out
        """
        return s,w,wS,cont0,cont1,cont2,cont3,cont4,diff1,diff2,diff3,diff4

    def operators(self,g,vextend,K):
        if K==0:
            advec=g
            advec=vextend*(tf.roll(advec,1,axis=1)-tf.roll(advec,-1,axis=1))/(2*hx)
            return advec
        elif K==1:
            prj=g
            s=wvextend*prj
            reduced=tf.math.reduce_sum(s/2.0,0)
            vv=tf.concat([reduced,reduced],0)
            for I in range(0,Nv-2):
                vv=tf.concat([vv,reduced],0)
            prj=tf.reshape(vv,[Nv,Nx,n_dat])
            return prj

    def advection(self,g,vextend):
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

    def extend_2d_to_3d(self,g):
        reduced=g
        vv=tf.concat([reduced,reduced],0)
        for I in range(0,Nv-2):
            vv=tf.concat([vv,reduced],0)
        prj=tf.reshape(vv,[Nv,Nx,n_dat])
        return prj


    def net_uv(self,rho_extend,g,vextend,wvextend):

        w_eps=self.w_eps
        eps=(tf.math.tanh(w_eps) + 1)*(max_epsilon-min_epsilon )/2+min_epsilon

        Fg, weights_g, wS,cont0,cont1,cont2,cont3,cont4,diff1,diff2,diff3,diff4 = self.RNNg(g,self.weights_g,self.Max_orders,self.N_layers,eps)
        Frho, weights_rho, wSr,cont0r,cont1r,cont2r,cont3r,cont4r,diff1r,diff2r,diff3r,diff4r = self.RNNrho(rho_extend,self.weights_rho,self.Max_orders,self.N_layers,eps)
        temp=(Fg+Frho)
        
        print(diff1)
        print(diff2)
        print(diff3)
        print(diff4)
        
        sigmaS=self.sigmaS
        sigmaA=self.sigmaA
        
#        K_g=g[:,:,1:n_dat-1]-g[:,:,0:n_dat-2]-dt*temp[:,:,0:n_dat-2]-sigmaS*g[:,:,0:n_dat-2]*dt-sigmaA*g[:,:,0:n_dat-2]*dt


        K_g=g[:,:,1:n_dat-1]-g[:,:,0:n_dat-2]-dt*temp[:,:,0:n_dat-2]-sigmaS*g[:,:,0:n_dat-2]*dt-sigmaA*g[:,:,0:n_dat-2]*dt
        pvg=self.projection(self.advection(g,vextend),wvextend)
        K_g=K_g+dt*pvg[:,:,0:n_dat-2]*1/eps

#        K_g=g[:,:,1:n_dat-1]-g[:,:,0:n_dat-2]-dt*temp[:,:,0:n_dat-2]-sigmaS/eps**2*g[:,:,1:n_dat-1]*dt-sigmaA*g[:,:,0:n_dat-2]*dt

        return K_g,weights_g,weights_rho,w_eps,sigmaS,sigmaA,wS,cont0,cont1,cont2,cont3,cont4,diff1,diff2,diff3,diff4,wSr,cont0r,cont1r,cont2r,cont3r,cont4r,diff1r,diff2r,diff3r,diff4r


    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        tf_dict = {self.g_tf: self.g,
                   self.rho_extend_tf: self.rho_extend,
                   self.vextend_tf: self.vextend,
                   self.wvextend_tf: self.wvextend}
        
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
        tf_dict = {self.g_tf: g,
                   self.rho_extend_tf: self.rho_extend,
                   self.vextend_tf: self.vextend,
                   self.wvextend_tf: self.wvextend}
        weights_g=self.sess.run(self.weights_g_tf)
        weights_rho=self.sess.run(self.weights_rho_tf)
    
        w_eps=self.sess.run(self.w_eps_tf)
        sigmaS=self.sess.run(self.sigmaS_tf)
        sigmaA=self.sess.run(self.sigmaA_tf)
        wSg=self.sess.run(self.wS_tf)
        wSrho=self.sess.run(self.wSr_tf)

        return weights_g,weights_rho,w_eps,sigmaS,sigmaA,wSg,wSrho
    
if __name__ == "__main__":
    
    TTdata = scipy.io.loadmat('data_mm.mat')
    
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
    n_start=round(0*n/3.0)
    n_end=round(3.0*n/3.0)
    n_dat_temp=n_end-n_start
    n_dat=round(n_dat_temp/nskip)
    T_final=n*dt
    dt=nskip*dt
    dx=hx
    
    


    #N_layers should start at 1    
    N_layers=1
    #This is the multiscale setting
    #Max_orders=0 means no epsilon dependence, 
    #Max_orders=2 means we have slow, medium, and fast scales.
    
#    Max_orders should start at 1
    Max_orders=1
         
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

    vplus=np.zeros([Nv,1],dtype=float)
    vminus=np.zeros([Nv,1],dtype=float)
    for j in range(0,Nv):
        if v[j]>0:
            vplus[j,0]=v[j,0]
        else:
            vminus[j,0]=v[j,0]
    vplus=np.float32(vplus)
    vminus=np.float32(vminus)
    vplusextend=np.zeros([Nv,Nx,n_dat],dtype=float)
    vminusextend=np.zeros([Nv,Nx,n_dat],dtype=float)
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,n_dat):
                vplusextend[I,J,K]=vplus[I,0]
                vminusextend[I,J,K]=vminus[I,0]

    vplusextend=np.float32(vplusextend)
    vminusextend=np.float32(vminusextend)
                
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

#    x1=np.linspace(0,0.2-hx[0,0],int(Nx/5))
#    x2=np.linspace(0.2,0.4-hx[0,0],int(Nx/5))
#    x3=np.linspace(0.4,0.6-hx[0,0],int(Nx/5))
#    x4=np.linspace(0.6,0.8-hx[0,0],int(Nx/5))
#    x5=np.linspace(0.8,1-hx[0,0],int(Nx/5))

    x1=np.linspace(0,0.2-hx[0],int(Nx/5))
    x2=np.linspace(0.2,0.4-hx[0],int(Nx/5))
    x3=np.linspace(0.4,0.6-hx[0],int(Nx/5))
    x4=np.linspace(0.6,0.8-hx[0],int(Nx/5))
    x5=np.linspace(0.8,1-hx[0],int(Nx/5))


    x1=np.float32(x1)
    x2=np.float32(x2)
    x3=np.float32(x3)
    x4=np.float32(x4)
    x5=np.float32(x5)
        
    v=np.float32(v)

    max_epsilon=1
    min_epsilon=0.01

    Nsteps=20
    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    modelrho = PDE_CLASSIFY(rho_extend,g,n_dat,N_layers,Nv,Nx,dt,hx,vextend,wvextend,max_epsilon,min_epsilon,Max_orders,Losss,x1,x2,x3,x4,x5)

    start_time = time.time()
    modelrho.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    weights_g,weights_rho,w_eps,sigmaS,sigmaA,wSg,wSrho= modelrho.predictrho(g)

    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2, label='Log of Loss vs Number of Iterations')
    plt.show()
    plt.legend()
    fig.savefig('LossBDF2_example1.png')

    savemat('Loss1.mat',{'L_domain':Losss_domain, 'Loss_y':LLosss})

    u_pred_ndat=np.zeros([Nx,1],dtype=float)

    eps=(np.tanh(w_eps) + 1)*(max_epsilon-min_epsilon)/2+min_epsilon


    c_g,c_v_g_x,pvgx,c_v_2_g_xx=coeffsg(wSg,N_layers,eps)
    c_rho,c_v_rho_x,c_pvrhox=coeffsrho(wSrho,N_layers,eps)

    cg2=c_g
    c_g=c_g-sigmaS/eps**2-sigmaA
#    c_g=c_g-sigmaS-sigmaA
##    c_g=c_g-1/eps**2

    print('The learned IMEX1 G-equation is:')
    print(c_g,'g +',c_v_g_x,'v*g_x+',pvgx,'<vg_x>')
    print(c_rho,'rho +',c_v_rho_x,'v*rho_x+')

    ep=np.int(1)
    ep2=ep**2

    c1=abs(c_g)-ep2
    c2=0#abs(c_v_g_x)-ep
    c3=abs(pvgx)-ep
    c4=0#abs(c_v_rho_x)-ep2
    c5=abs(c_v_2_g_xx)
    c6=abs(c_rho)
    c7=abs(c_pvrhox)
    error=(abs(c1)+abs(c2)+abs(c3)+abs(c4)+abs(c5)+abs(c6)+abs(c7))/(2*ep2+2*ep)*100
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



    fig = plt.figure()
    plt.scatter(x,c_v_g_x, color='black', linewidth=2, label='Plot of $v\partial_{x}g$')
    plt.show()
    fig.savefig('c_v_g_x.png')




    fig = plt.figure()
    plt.scatter(x,c_v_rho_x, color='black', linewidth=2, label='Plot of $v\partial_{x}\rho$')
    plt.show()
    fig.savefig('c_v_rho_x.png')





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
#    print('1/',ep,' & No & $\partial_{t}g=-(',ep,'^2-{\color{red}',c1/10**J1,'\cdot 10^{',J1,'}}) g -(',ep,'-{\color{red}',c2/10**J2,'\cdot 10 ^{',J2,'}}) v\cdot \partial_{x}g$  & \\\\')
    #line 2
#    print('& & $+ (',ep,'-{\color{red}',c3/10**J3,'\cdot 10^{',J3,'}})\langle v\partial_{x}g\\rangle -(',ep2,'-{\color{red}',c4/10**J4,'\cdot 10^{',J4,'}})v\cdot\partial_{x}\\rho+\cdots$ & {',error,'\%}\\\\')
    
    """    
    \bottomrule
    \end{tabular}
    \end{sc}
    \end{scriptsize}
    \end{center}\caption{Learned $g$-equation using the DC-RNN algorithm based on Forward-Euler (FE) and IMEX schemes.}
    \vskip -0.1in
    \end{table}
    """
    
#    

#    savemat('W_bdf0',{'weightsg0':weightsg0})
#    savemat('W_bdf1',{'weightsg1':weightsg1})
#    savemat('W_stiff_bdf',{'wstiffg':wstiffg})
#    savemat('W_stiff_bdf2',{'wstiffg2':wstiffg2})



    
    
    