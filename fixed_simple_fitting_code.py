# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:01:24 2021

@author: Ricardo
"""

#Getting rid of the identity operator
#Foward Euler and IMEX


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
    c_v_g_x=0
    pvgx=0
    c_v_2_g_xx=0
    if N_layers==1:
        W1=ww[0]
        for I in range(0,Max_orders+1):
            c_g=c_g+0.0
            c_v_g_x=c_v_g_x+(W1[0,I]+W1[1,I])/eps**I
            pvgx=pvgx
            c_v_2_g_xx=0
    if N_layers==2:
        W1=ww[0]
        W2=ww[1]
        for I in range(0,Max_orders+1):
            c_g=c_g+0.0
            c_v_g_x=c_v_g_x+(W1[0,I]+W1[1,I])/eps**I
            pvgx=pvgx+(W2[4,I]+W2[5,I]+W2[6,I]+W2[7,I])*(W1[0,I]+W1[1,I])/eps**I
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
        W1=ww[0]
        for I in range(0,Max_orders+1):
            c_rho=c_rho+0.0
            c_v_rho_x=c_v_rho_x+(W1[0,I]+W1[1,I])/eps**I
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
    def __init__(self,rho_extend,g,n_dat,N_layers,Nv,Nx,dt,hx,vextend,wvextend,max_epsilon,min_epsilon,Max_orders,Losss):
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

        self.weights_g,self.weights_rho,self.w_eps, self.sigmaS, self.sigmaA= self.initialize_NNg(N_layers,Max_orders)

        self.rho_extend_tf = tf.placeholder(tf.float32, shape=[None, None, self.rho_extend.shape[2]])

        self.g_tf = tf.placeholder(tf.float32, shape=[None, None, self.g.shape[2]])

        self.vextend_tf = tf.placeholder(tf.float32, shape=[None, None, self.vextend.shape[2]])
        self.wvextend_tf = tf.placeholder(tf.float32, shape=[None, None, self.wvextend.shape[2]])

        # tf GraphsK_g,weights_g,weights_rho,w_eps,sigmaS,sigmaA

        self.g_pred,self.weights_g_tf, self.weights_rho_tf,self.w_eps_tf,self.sigmaS_tf,self.sigmaA_tf = self.net_uv(self.rho_extend_tf,self.g_tf,self.vextend_tf,self.wvextend_tf)
        # Loss

        gamma=0.000001

        self.loss = tf.reduce_mean(tf.abs(self.g_pred))#+gamma*(tf.reduce_mean(tf.abs(self.weights_g_tf[0]))+tf.reduce_mean(tf.abs(self.weights_rho_tf[0])))+gamma*(tf.reduce_mean(tf.abs(self.weights_g_tf[1]))+tf.reduce_mean(tf.abs(self.weights_rho_tf[1])))+gamma*(tf.reduce_mean(tf.abs(self.weights_g_tf[2]))+tf.reduce_mean(tf.abs(self.weights_rho_tf[2])))

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
        initializeit=0.25
        init=np.zeros([N_layers],dtype=float)
        for I in range(0,N_layers):
            init[I]=abs(np.random.normal())
        init=np.flip(np.sort(init))    
        for I in range(0,N_layers):
#            W=initializeit*tf.Variable(tf.ones([2**(N_layers-I),Max_orders+1],dtype=tf.float32),dtype=tf.float32)
#            weights_g.append(W)
#            W=initializeit*tf.Variable(tf.ones([2**(N_layers-I),Max_orders+1],dtype=tf.float32),dtype=tf.float32)
#            weights_rho.append(W)
            W=init[I]*tf.Variable(tf.ones([2**(I+2),Max_orders+1],dtype=tf.float32),dtype=tf.float32)
            weights_g.append(W)
            W=init[I]*tf.Variable(tf.ones([2**(I+2),Max_orders+1],dtype=tf.float32),dtype=tf.float32)
            weights_rho.append(W)
        w_eps=0.05*tf.Variable(1.0,dtype=tf.float32)
        sigmaS=initializeit*tf.Variable(1.0,dtype=tf.float32)
        sigmaA=initializeit*tf.Variable(1.0,dtype=tf.float32)

        return weights_g,weights_rho,w_eps,sigmaS,sigmaA



    def RNNg(self,g,w,Max_orders,N_layers,eps):
######################
### G-RNN
######################
        s=g*0.0
        for K in range(0,Max_orders+1):
            out=g*0.0
            F_rhs_g=g
            for I in range(0,N_layers):
                W=w[I]
                s1=tf.reduce_sum(W[0:int((2**(I+2))/2),K])
                s2=tf.reduce_sum(W[int((2**(I+2))/2):int(2**(I+2)),K])
                F_rhs_g=s1*self.advection(F_rhs_g,vextend)+s2*self.projection(F_rhs_g,wvextend)
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
        return s,w

    def RNNrho(self,rho_extend,w,Max_orders,N_layers,eps):
###################   
### RHO RNN
###################
        s=rho_extend*0.0
        for K in range(0,Max_orders+1):
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
        return s,w

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

    def net_uv(self,rho_extend,g,vextend,wvextend):

        w_eps=self.w_eps
        eps=(tf.math.tanh(w_eps) + 1)*(max_epsilon-min_epsilon )/2+min_epsilon

        Fg, weights_g = self.RNNg(g,self.weights_g,self.Max_orders,self.N_layers,eps)
        Frho, weights_rho = self.RNNrho(rho_extend,self.weights_rho,self.Max_orders,self.N_layers,eps)
        temp=(Fg+Frho)

        sigmaS=self.sigmaS
        sigmaA=self.sigmaA



        K_g=g[:,:,1:n_dat-1]-g[:,:,0:n_dat-2]-dt*temp[:,:,0:n_dat-2]-sigmaS*g[:,:,0:n_dat-2]*dt-sigmaA*g[:,:,0:n_dat-2]*dt
        pvg=self.projection(self.advection(g,vextend),wvextend)
        K_g=K_g+dt*pvg[:,:,0:n_dat-2]*1/eps

#        K_g=g[:,:,1:n_dat-1]-g[:,:,0:n_dat-2]-dt*temp[:,:,0:n_dat-2]-sigmaS/eps**2*g[:,:,0:n_dat-2]*dt-sigmaA*g[:,:,0:n_dat-2]*dt

        return K_g,weights_g,weights_rho,w_eps,sigmaS,sigmaA


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

        return weights_g,weights_rho,w_eps,sigmaS,sigmaA
    
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
        
    v=np.float32(v)

    max_epsilon=1
    min_epsilon=0.01

    Nsteps=50000
    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    modelrho = PDE_CLASSIFY(rho_extend,g,n_dat,N_layers,Nv,Nx,dt,hx,vextend,wvextend,max_epsilon,min_epsilon,Max_orders,Losss)

    start_time = time.time()
    modelrho.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    weights_g,weights_rho,w_eps,sigmaS,sigmaA= modelrho.predictrho(g)

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


    c_g,c_v_g_x,pvgx,c_v_2_g_xx=coeffsg(weights_g,N_layers,eps)
    c_rho,c_v_rho_x,c_pvrhox=coeffsrho(weights_rho,N_layers,eps)

    cg2=c_g
    c_g=c_g-sigmaS/eps**2-sigmaA
#    c_g=c_g-sigmaS-sigmaA
##    c_g=c_g-1/eps**2

    print('The learned IMEX1 G-equation is:')
    print(c_g,'g +',c_v_g_x,'v*g_x+',pvgx,'<vg_x>')
    print(c_rho,'rho +',c_v_rho_x,'v*rho_x+')

    ep=np.int(3)
    ep2=ep**2

    c1=abs(c_g)-ep2
    c2=abs(c_v_g_x)-ep
    c3=abs(pvgx)-ep
    c4=abs(c_v_rho_x)-ep2
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
    
    

#    savemat('W_bdf0',{'weightsg0':weightsg0})
#    savemat('W_bdf1',{'weightsg1':weightsg1})
#    savemat('W_stiff_bdf',{'wstiffg':wstiffg})
#    savemat('W_stiff_bdf2',{'wstiffg2':wstiffg2})