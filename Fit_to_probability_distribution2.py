
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
    def __init__(self,Nx,Ny,x_mesh,y_mesh,V,layers,Losss):

        self.Nx=Nx
        self.Ny=Ny
        self.layers = layers
        self.Losss=Losss
        self.x_mesh=x_mesh
        self.y_mesh=y_mesh
        self.V = V

        self.V_tf = tf.placeholder(tf.float32, shape=[Nx,Ny])

        # Initialize NNs
        self.weights, self.biases, self.D1, self.D2 = self.initialize_NN(layers,Ny)

#        self.V_pred = self.net_uv(self.x_flat)
        self.V_pred,self.Vx_pred,self.Vxx_pred,self.Vt_pred = self.net_uv(self.x_mesh)

        #Define the Loss
#        self.loss = tf.reduce_mean(tf.abs(self.V_pred[pad:Nx-pad,pad:Nx-pad]-self.V[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(self.Vx_pred[pad:Nx-pad,pad:Nx-pad]-self.Vx[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))

        self.loss =tf.reduce_mean(tf.square(self.V_pred-self.V_tf))
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

    def initialize_NN(self, layers, Ny):
        weights = []
        biases = []
#        num_layers = len(layers) 
#        for l in range(0,num_layers-1):
#            W = self.xavier_init(size=[layers[l], layers[l+1]])
#            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
#            weights.append(W)
#            biases.append(b)
        D1=0.0*tf.Variable(tf.ones([Nx*Ny,1],dtype=tf.float32), dtype=tf.float32)
        D2=0.0*tf.Variable(tf.ones([Nx*Ny,1],dtype=tf.float32), dtype=tf.float32)

        W=tf.Variable(tf.random.uniform([1,Ny],dtype=tf.float32),dtype=tf.float32)
        weights.append(W)
        W=tf.Variable(tf.random.uniform([1,Ny],dtype=tf.float32),dtype=tf.float32)
        weights.append(W)
        W=tf.Variable(tf.random.uniform([1,Ny],dtype=tf.float32),dtype=tf.float32)
        weights.append(W)

        W=tf.Variable(1.0,dtype=tf.float32)
        weights.append(W)

        W=tf.Variable(1.0,dtype=tf.float32)
        weights.append(W)

#        W=tf.Variable(tf.ones([1,Ny],dtype=tf.float32),dtype=tf.float32)
#        weights.append(W)
#        W=tf.Variable(tf.ones([1,Ny],dtype=tf.float32),dtype=tf.float32)
#        weights.append(W)
#        W=tf.Variable(tf.random.uniform([1,Ny],dtype=tf.float32),dtype=tf.float32)
#        weights.append(W)





        return weights, biases, D1, D2
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


    def neural_net(self, weights, biases, D1, D2):
        A=weights[0]
        alpha=weights[1]
        beta=weights[2]
        c=weights[3]
        d=weights[4]
        
        X=self.x_mesh
        Y=self.y_mesh

        AA=tf.linalg.matmul(tf.ones([Nx,1],dtype=tf.float32),A)
        Alpha=tf.linalg.matmul(tf.ones([Nx,1],dtype=tf.float32),alpha)
        Beta=tf.linalg.matmul(tf.ones([Nx,1],dtype=tf.float32),beta)
        
        

#        model=AA*tf.math.exp(-abs(Alpha)*(X-Beta)**2)
        
        model=AA*tf.math.exp(-Alpha*(X-c*Y-d)**2)
        
        Vx=-2*Alpha*AA*(X-c*Y-d)*tf.math.exp(-Alpha*(X-c*Y-d)**2)
        Vxx=(-2*Alpha*AA+4*Alpha**2*AA*(X-c*Y-d)**2)*tf.math.exp(-Alpha*(X-c*Y-d)**2)

        Vt=2*c*Alpha*(X-c*Y-d)*AA*tf.math.exp(-(Alpha)*(X-c*Y-d)**2)
        
        return model,Vx,Vxx,Vt


    def net_uv(self, x_flat ):
        V,Vx,Vxx,Vt = self.neural_net(self.weights, self.biases, self.D1, self.D2)
        
        
        
        
        return V,Vx,Vxx,Vt


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
        Vxx_pred=self.sess.run(self.Vxx_pred)
        Vt_pred=self.sess.run(self.Vt_pred)
        return V_pred,Vx_pred,Vxx_pred,Vt_pred
    





   
    
    
    
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
    for I in range(0,Nx):
        for J in range(0,Ny):
            V[I,J]=rho[I,J]

            
    x_mesh=np.zeros([Nx,Ny],dtype=float)
    y_mesh=np.zeros([Nx,Ny],dtype=float)
    for I in range(0,Nx):
        for J in range(0,Ny):
            x_mesh[I,J]=x[I,0]
            y_mesh[I,J]=y[J,0]

    #This is what Nsteps does
    Nsteps=70000

    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model = NeuralNetworkCode1(Nx,Ny,x_mesh,y_mesh,V,layers,Losss)
        
    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    V_pred, Vx_pred,Vxx_pred,Vt_pred= model.predict()


    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2)
    

    V_pred_3d = np.zeros([Nx,Ny],dtype=float)

    x_mesh,y_mesh=np.meshgrid(y,x)
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, V, color='r')
    plt.title('Probability Density Simulation',fontsize=14,fontweight='bold')

    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, V_pred, color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')


    savemat('Prob_distribution.mat',{'V':V_pred, 'Vx':Vx_pred, 'Vxx':Vxx_pred, 'Vt':Vt_pred ,'Nx':Nx, 'Ny':Ny})



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

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    

 
