
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
from scipy.stats import norm
from statistics import NormalDist
import statsmodels.api as sm
import pylab as py
import math



class NeuralNetworkCode1:
    def __init__(self,x_flat,y_flat,V_flat,layers,Losss):

        self.layers = layers
        self.Losss=Losss
        self.x_flat=x_flat
        self.y_flat=y_flat
        self.V_flat = V_flat

        self.V_tf = tf.placeholder(tf.float32, shape=[Nx*Ny,1])

        # Initialize NNs
        self.weights, self.biases, self.D1, self.D2 = self.initialize_NN(layers)

#        self.V_pred = self.net_uv(self.x_flat)
        self.V_pred, self.Vx_pred, self.Vy_pred = self.net_uv(self.x_flat)

        #Define the Loss
#        self.loss = tf.reduce_mean(tf.abs(self.V_pred[pad:Nx-pad,pad:Nx-pad]-self.V[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(self.Vx_pred[pad:Nx-pad,pad:Nx-pad]-self.Vx[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))

        self.loss =tf.reduce_mean(tf.square(self.V_pred-self.V_tf))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))
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
        D1=0.0*tf.Variable(tf.ones([Nx*Ny,1],dtype=tf.float32), dtype=tf.float32)
        D2=0.0*tf.Variable(tf.ones([Nx*Ny,1],dtype=tf.float32), dtype=tf.float32)
        return weights, biases, D1, D2
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


    def neural_net(self, weights, biases, D1, D2):
        X=D1+self.x_flat
        Y=D2+self.y_flat
        H=tf.concat([X,Y], 1)
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
        V = self.neural_net(self.weights, self.biases, self.D1, self.D2)
        V_x=tf.gradients(V,self.D1)[0]
        V_y=tf.gradients(V,self.D2)[0]
        
        return V,V_x,V_y


    def callback(self, loss):
        print('Loss:', loss)


    def train(self, nIter):
        tf_dict = {self.V_tf: V_flat}
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
        Vy_pred=self.sess.run(self.Vy_pred)
        return V_pred, Vx_pred, Vy_pred
    





   
    
    
    
if __name__ == "__main__":
    
    TTdata = scipy.io.loadmat('data_SDE.mat')
    Tdata=scipy.io.loadmat('data_SDE_density.mat')



#    B = TTdata['B']
    t = TTdata['t']
    y = TTdata['y']
    y_mean = TTdata['y_mean']
    y_var = TTdata['y_var']
    dt = TTdata['dt']
    Nt = TTdata['Nt']
    Num_sim = TTdata['Num_sim']
    a = TTdata['a']
    b=TTdata['b']


    dt=dt[0,0]
    Nt=Nt[0,0]
    Num_sim=Num_sim[0,0]
    a=a[0,0]
    b=b[0,0]


    
#    savemat('data_SDE_density.mat',{'rho':rho,'t':t,'dt':dt ,'Nt':Nt, 'Nx':Nx,'aa':aa, 'bb':bb,'t1':t1})
    rho=Tdata['rho']
    Nx=Tdata['Nx']

    [Nx,Ny]=rho.shape


    aa=Tdata['aa']
    bb=Tdata['bb']
    t1=Tdata['t1']
    dx=Tdata['dx']


#    layers = [2, 5, 5, 5, 5, 5, 5, 1]
    layers = [2, 5, 5, 5, 5, 5, 1]
    ax=aa
    bx=bb
    
#    Nx=Nx[0,0]
    dx=dx[0,0]
    ay=t1[0,0]
    by=1

    #Defining grid points (x,y)
    x=np.zeros([Nx,1], dtype=float)
    for I in range(0,Nx):
        x[I,0] = ax+(bx-ax)/(Nx-1)*I

    y=np.zeros([Ny,1], dtype=float)
    for I in range(0,Ny):
        y[I,0] = ay+(by-ay)/(Ny-1)*I        

    V=np.zeros([Nx,Ny],dtype=float)
    Vx=np.zeros([Nx,Ny],dtype=float)
    Vy=np.zeros([Nx,Ny],dtype=float)
    for I in range(0,Nx):
        for J in range(0,Ny):
            V[I,J]=rho[I,J]
#            Vx[I,J]=np.roll(rho,1,0)-np.roll(rho,1,0)
#            Vy[I,J]=np.pi*np.sin(np.pi*x[I,0])*np.cos(np.pi*y[J,0])
            
    x_flat=np.zeros([Nx*Ny,1],dtype=float)
    y_flat=np.zeros([Nx*Ny,1],dtype=float)
    V_flat=np.zeros([Nx*Ny,1],dtype=float)
    L=0
    for I in range(0,Nx):
        for J in range(0,Ny):
            x_flat[L,0]=x[I,0]
            y_flat[L,0]=y[J,0]
            V_flat[L,0]=V[I,J]
            L=L+1

    #This is what Nsteps does
    Nsteps=50000

    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model = NeuralNetworkCode1(x_flat,y_flat,V_flat,layers,Losss)
        
    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    V_pred, Vx_pred, Vy_pred = model.predict()


    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2)
    

    V_pred_3d = np.zeros([Nx,Ny],dtype=float)
    Vx_pred_3d = np.zeros([Nx,Ny],dtype=float)
    Vy_pred_3d = np.zeros([Nx,Ny],dtype=float)

    L=0
    for I in range(0,Nx):
        for J in range(0,Ny):
            V_pred_3d[I,J]=V_pred[L,0]
            Vx_pred_3d[I,J]=Vx_pred[L,0]
            Vy_pred_3d[I,J]=Vy_pred[L,0]
            L=L+1

    x_mesh,y_mesh=np.meshgrid(y,x)
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, V_pred_3d, color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, V, color='r')
    plt.title('Probability Density Simulation',fontsize=14,fontweight='bold')


    savemat('Prob_distribution.mat',{'V':V, 'Nx':Nx, 'Ny':Ny})














    partial_x=(np.roll(V_pred_3d,1,0)-np.roll(V_pred_3d,-1,0))/(2*dx)
    partial_t=(np.roll(V_pred_3d,1,1)-np.roll(V_pred_3d,-1,1))/(2*dt)
    
    
    [NNc,NNNc]=partial_t.shape
    partial_t[:,0]=0
    partial_t[:,NNNc-1]=0

    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, Vy_pred_3d, color='r')
    plt.title('partial in t using NN',fontsize=14,fontweight='bold')

    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, partial_t, color='r')
    plt.title('partial in t using FD',fontsize=14,fontweight='bold')

    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, partial_t-Vy_pred_3d, color='r')
    plt.title('partial in t using FD',fontsize=14,fontweight='bold')
   
    

    

#    Nsteps=15000
#    Losss=np.zeros([Nsteps,1],dtype=float)
#    Losss_domain=np.zeros([Nsteps,1],dtype=float)
#    for I in range(0,Nsteps):
#        Losss_domain[I,0]=I
               

    #Fitting using Legendre polynomials.



    x=x/(bb-aa)

    L=np.zeros([Nx,10],dtype=float)
    L[:,0]=1+0*x[:,0]
    L[:,1]=x[:,0]
    L[:,2]=0.5*(3*x[:,0]*x[:,0]-1)
    L[:,3]=1.0/2.0*(5*x[:,0]**3-3*x[:,0])
    L[:,4]=1.0/8.0*(35*x[:,0]**4-30*x[:,0]**2+3)
    L[:,5]=1.0/8.0*(63*x[:,0]**5-70*x[:,0]**3+15*x[:,0])
    L[:,6]=1.0/16.0*(231*x[:,0]**6-315*x[:,0]**4+105*x[:,0]**2-5)
    L[:,7]=1.0/16.0*(429*x[:,0]**7-693*x[:,0]**5+315*x[:,0]**3-35*x[:,0])
    L[:,8]=1.0/128.0*(6435*x[:,0]**8-12012*x[:,0]**6+6930*x[:,0]**4-1260*x[:,0]**2+35)
    L[:,9]=1.0/128.0*(12155*x[:,0]**9-25740*x[:,0]**7+18018*x[:,0]**5-4620*x[:,0]**3+315*x[:,0])
    
    V_L = np.zeros([Nx,Ny],dtype=float)    
    N_size=10
    A=np.zeros([Nx,N_size],dtype=float)
    for I in range(0,N_size):
        A[:,I]=L[:,I]
    
    #solving the normal equations:
    for I in range(0,Ny):
        alpha = np.linalg.lstsq(A, V[:,I], rcond=None)[0]
    
        y_fit=np.zeros([Nx,1],dtype=float)
        for K in range(0,N_size):
            y_fit[:,0]=y_fit[:,0]+alpha[K]*L[:,K]
        V_L[:,I]=y_fit[:,0]
 
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, V_L, color='r')
    plt.title('V using legendre polynomials',fontsize=14,fontweight='bold')
    
    
        #Fitting using Legendre polynomials.
    
    
    y=y-y[0]
    y=y/(y[Ny-1]-y[0])

    
    L=np.zeros([Ny,10],dtype=float)
    L[:,0]=1+0*y[:,0]
    L[:,1]=y[:,0]
    L[:,2]=0.5*(3*y[:,0]*y[:,0]-1)
    L[:,3]=1.0/2.0*(5*y[:,0]**3-3*y[:,0])
    L[:,4]=1.0/8.0*(35*y[:,0]**4-30*y[:,0]**2+3)
    L[:,5]=1.0/8.0*(63*y[:,0]**5-70*y[:,0]**3+15*y[:,0])
    L[:,6]=1.0/16.0*(231*y[:,0]**6-315*y[:,0]**4+105*y[:,0]**2-5)
    L[:,7]=1.0/16.0*(429*y[:,0]**7-693*y[:,0]**5+315*y[:,0]**3-35*y[:,0])
    L[:,8]=1.0/128.0*(6435*y[:,0]**8-12012*y[:,0]**6+6930*y[:,0]**4-1260*y[:,0]**2+35)
    L[:,9]=1.0/128.0*(12155*y[:,0]**9-25740*y[:,0]**7+18018*y[:,0]**5-4620*y[:,0]**3+315*y[:,0])
    
    V_L = np.zeros([Nx,Ny],dtype=float)    
    N_size=10
    A=np.zeros([Ny,N_size],dtype=float)
    for I in range(0,N_size):
        A[:,I]=L[:,I]
    
    #solving the normal equations:
    for I in range(0,Nx):
        alpha = np.linalg.lstsq(A, V[I,:], rcond=None)[0]
    
        y_fit=np.zeros([Ny,1],dtype=float)
        for K in range(0,N_size):
            y_fit[:,0]=y_fit[:,0]+alpha[K]*L[:,K]
        V_L[I,:]=y_fit[:,0]
 
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, V_L, color='r')
    plt.title('V using legendre polynomials',fontsize=14,fontweight='bold')
    
    




        #Fitting using Gaussians.
    
    
    
    
    
    
    
    
    

 
