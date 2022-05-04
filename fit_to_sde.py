


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



global_seed = 9
N_chains = 5
np.random.seed(global_seed)
seeds = np.random.randint(0, 9, size=N_chains)


def histogram_create(Normal_distribution,N_space,n_c,Nt):
    domain = np.linspace(-n_c,n_c,N_space)    
    histogram_domain=np.zeros([N_space,1],dtype=float)
    for I in range(0,N_space):
        histogram_domain[I,0]=domain[I]
    histogram=np.zeros([N_space,1],dtype=float)
    for I in range(0,N_space):
        for J in range(0,Nt):
            if ((-n_c+n_c/(N_space/2)*I<=Normal_distribution[J]) & (Normal_distribution[J]<=-n_c+n_c/(N_space/2)*(I+1))):
                histogram[I,0]=histogram[I,0]+1
    return histogram_domain, histogram


#aand fit to normal: to do

class randomness_fit:
    def __init__(self,H_domain,H,N_space,layers,Losss):
        self.layers = layers
        self.Losss=Losss
        self.H_domain=H_domain
        self.H=H
        self.N_space=N_space

        self.H_tf = tf.placeholder(tf.float32, shape=[N_space,1])

        # Initialize NNs
        self.weights, self.biases, self.D1, self.D2 = self.initialize_NN(layers)

        self.H_pred, self.Hx_pred = self.net_uv(self.H_domain)

        #Define the Loss
        self.loss = tf.reduce_mean(tf.square(self.H_pred-self.H))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))
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
        D1=0.0*tf.Variable(tf.ones([N_space,1],dtype=tf.float32), dtype=tf.float32)
        D2=0.0*tf.Variable(tf.ones([N_space,1],dtype=tf.float32), dtype=tf.float32)
        return weights, biases, D1, D2
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
        
    def neural_net(self, weights, biases, D1, D2):
        X=D1+histogram_domain
        V=X
        num_layers = len(weights) + 1
        print('1')
        print(V.shape)
        print('2')
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
#            V = tf.tanh(tf.add(tf.matmul(V, W), b))
#            V = tf.nn.relu(tf.add(tf.matmul(V, W), b))
            V = tf.nn.sigmoid(tf.add(tf.matmul(V, W), b))
            print(V.shape)
        W = weights[-1]
        b = biases[-1]
        print('3')
        V = tf.add(tf.matmul(V, W), b)
        print('4')
        return V

    def net_uv(self, histogram_domain):
        V = self.neural_net(self.weights, self.biases, self.D1, self.D2)
        print('5')
        V_s=tf.gradients(V,self.D1)[0]
        print('6')
        return V,V_s


    def callback(self, loss):
        print('Loss:', loss)


    def train(self, nIter):
        tf_dict = {self.H_tf: self.H}
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


    def predict_randomness(self,H):
        sigma_pred=self.sess.run(self.H_pred)
        sigmax_pred=self.sess.run(self.Hx_pred)
        return sigma_pred,sigmax_pred




class SDE_CLASSIFY:
    def __init__(self,S,N_terms,alpha1,alpha2,alpha3,alpha4,alpha5,beta1,beta2,beta3,beta4,beta5):
        self.S=S
        self.N_terms=N_terms
        self.alpha1=alpha1
        self.alpha2=alpha2
        self.alpha3=alpha3
        self.alpha4=alpha4 
        self.alpha5=alpha5        
        self.beta1=beta1
        self.beta2=beta2
        self.beta3=beta3
        self.beta4=beta4 
        self.beta5=beta5        
        
        self.S_tf = tf.placeholder(tf.float32, shape=[Nt])
        """
        self.alpha1_tf = tf.placeholder(tf.float32, shape=[Nt])
        self.alpha2_tf = tf.placeholder(tf.float32, shape=[Nt])
        self.alpha3_tf = tf.placeholder(tf.float32, shape=[Nt])
        self.alpha4_tf = tf.placeholder(tf.float32, shape=[Nt])
        self.alpha5_tf = tf.placeholder(tf.float32, shape=[Nt])
        self.beta1_tf = tf.placeholder(tf.float32, shape=[Nt])
        self.beta2_tf = tf.placeholder(tf.float32, shape=[Nt])
        self.beta3_tf = tf.placeholder(tf.float32, shape=[Nt])
        self.beta4_tf = tf.placeholder(tf.float32, shape=[Nt])
        self.beta5_tf = tf.placeholder(tf.float32, shape=[Nt])
        """                
        self.c,self.d= self.initialize_NNg(N_terms)
        
        # tf Graphs
        self.S_pred,self.mu_pred,self.sigma_pred,self.c_pred,self.d_pred = self.net_uv(self.S_tf)
        # Loss

        gamma=0.000001
        self.loss = tf.reduce_mean(tf.square(self.S_pred-self.S[1:Nt-1]))

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


    def initialize_NNg(self, N_terms):

        initializeit=1.0
        c=initializeit*tf.Variable(tf.ones([N_terms,1],dtype=tf.float32),dtype=tf.float32)
        d=initializeit*tf.Variable(tf.ones([N_terms,1],dtype=tf.float32),dtype=tf.float32)

        return c,d


    def Right_hand_side(self,S,N_terms,c,d):

        Deterministic=c[0]*alpha1#+c[4]*alpha5
#        Deterministic=c[0]*alpha1+c[1]*alpha2+c[2]*alpha3+c[3]*alpha4+c[4]*alpha5
        Stochastic=0.0
        
        return Deterministic,Stochastic


    def net_uv(self,S):
        mu,sigma=self.Right_hand_side(S,N_terms,self.c,self.d)
        S_pred=S[0:Nt-2]*(1+mu[0:Nt-2]*dt)
        return S_pred,mu,sigma,self.c,self.d


    def callback(self, loss):
        print('Loss:', loss)

        
    def train(self, nIter):
        tf_dict = {self.S_tf: self.S}        
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


    def predictsde(self,S):
#        tf_dict = {self.S_tf: self.S}

        mu_pred=self.sess.run(self.mu_pred)
        c=self.sess.run(self.c_pred)
        d=self.sess.run(self.d_pred)
        return mu_pred,c,d
    """            
    def predictsde(self,S):
        tf_dict = {self.S_tf: self.S}
        c=self.sess.run(self.c_pred)
        d=self.sess.run(self.d_pred)
        S_pred=self.sess.run(self.S_pred)
        mu=self.sess.run(self.mu_pred)
        sigma=self.sess.run(self.sigma_pred)
        return c,d,S_pred,mu,sigma

    """
if __name__ == "__main__":

    

    ##################################
    ##Generate Synthetic Data:      ##
    ##################################
        
    Ts=0.0#Start time is zero
    Tf=1.0 #final time is Tf    

    #Creating t_i, from Ts to Tf
    Nt=200
    t = np.linspace ( Ts, Tf, Nt)#time axis
    dt=t[1]-t[0]

    #selecting Nt random numbers from the normal distribution 
    Normal_distribution=np.zeros([Nt],dtype=float)
    sigma=1

    for I in range ( 0, Nt ):
        Normal_distribution[I] = np.random.normal(loc=0.0, scale=sigma, size=None)
    
    n_a=np.min(Normal_distribution)
    n_b=np.max(Normal_distribution)
    n_c=max(abs(n_a),abs(n_b))
    n_c=n_c+n_c/2
    
    #Use this code to plot a histogram: the distribution of the random normal numbers
    N_space=20 #(Should be less than Nt)
    histogram_domain, histogram=histogram_create(Normal_distribution,N_space,n_c,Nt)

    #Plot of the distribution
    fig = plt.figure()
    plt.bar(range(len(histogram_domain)), histogram[:,0])
    plt.title('Distribution of Returns')
    plt.xlabel('2-year time period')
    plt.ylabel('Number of Returns within chosen interval size')


    #Plot of the distribution of returns
    fig = plt.figure()
    plt.plot(range(len(histogram_domain)), histogram/Nt)
    plt.title('Distribution of Returns')
    plt.xlabel('2-year time period')
    plt.ylabel('returns')




    #Defining a Brownian motion with drift 
    #and starting point at x0
    x0=5
    B=np.zeros([Nt],dtype=float)
    B=B+x0
    for I in range(0,Nt-1):
        B[I+1]=B[I]*(1+0.005+sigma/np.sqrt(Nt)*Normal_distribution[I])


    fig = plt.figure()
    plt.scatter(t,B, color='black', linewidth=2)
    plt.title('Brownian Motion')
        
    S=B #This is the data we are going to model
        
    #Generating Basis functions:
    N_terms=5
    """
    alpha=np.zeros([Nt,N_terms],dtype=float)
    beta=np.zeros([Nt,N_terms],dtype=float)
    for I in range(0,N_terms):
        for J in range(0,Nt):
            alpha[J,I]=t[J]**(2.0*I/N_terms)
            beta[J,I]=t[J]**(2.0*I/N_terms)
    """
    alpha1=t**(2.0*0.0/(N_terms-1))
    alpha2=t**(2.0*0.5/(N_terms-1))
    alpha3=t**(2.0*1.0/(N_terms-1))
    alpha4=t**(2.0*1.5/(N_terms-1))
    alpha5=t**(2.0*2.0/(N_terms-1))

    beta1=t**(2.0*0.0/N_terms)
    beta2=t**(2.0*0.5/N_terms)
    beta3=t**(2.0*1.0/N_terms)
    beta4=t**(2.0*1.5/N_terms)
    beta5=t**(2.0*2.0/N_terms)




    layers = [1, 10, 10, 10, 10, 1]



    Nsteps=10000
    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    randomness_model = randomness_fit(histogram_domain,histogram/Nt,N_space,layers,Losss)


    start_time = time.time()
    randomness_model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    sigma_pred, sigmax_pred = randomness_model.predict_randomness(histogram_domain)


    fig = plt.figure()
    plt.plot(histogram_domain, sigma_pred)
    plt.title('Distribution of Returns')
    plt.xlabel('2-year time period')
    plt.ylabel('Number of Returns within chosen interval size')

    fig = plt.figure()
    plt.plot(histogram_domain, sigmax_pred)
    plt.title('Distribution of Returns')
    plt.xlabel('2-year time period')
    plt.ylabel('Number of Returns within chosen interval size')

    
    """
    Nsteps=20
    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I
    
    model = SDE_CLASSIFY(S,N_terms,alpha1,alpha2,alpha3,alpha4,alpha5,beta1,beta2,beta3,beta4,beta5)

    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    #c,d,S_pred,mu,sigma = model.predictsde(S)
    mu_pred,c,d = model.predictsde(S)

    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2, label='Log of Loss vs Number of Iterations')
    plt.show()



    fig = plt.figure()
    plt.scatter(t[0:Nt-2],S[0:Nt-2]*(1+dt*mu_pred[0:Nt-2]), color='red', linewidth=1, label='$S$ Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predict vs Actual')
    plt.legend()
    plt.show()
    

    fig = plt.figure()
    plt.plot(t,S, color='black',linewidth=4, label='$S$ Exact')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predict vs Actual')
    plt.legend()
    plt.show()
    

    fig = plt.figure()
    plt.plot(t,mu_pred, color='black',linewidth=4, label='$S$ Exact')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predict vs Actual')
    plt.legend()
    plt.show()

    """