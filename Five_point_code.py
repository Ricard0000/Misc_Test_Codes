
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
    def __init__(self,v_flat,wv_flat,x_flat,g1_flat,gt1_flat,vrhox1_flat,g2_flat,gt2_flat,vrhox2_flat,g3_flat,gt3_flat,vrhox3_flat,g4_flat,gt4_flat,vrhox4_flat,g5_flat,gt5_flat,vrhox5_flat,layers,Losss):

        self.layers = layers
        self.Losss=Losss
        self.v_flat=v_flat
        self.wv_flat=wv_flat
        self.x_flat=x_flat
        self.g1_flat = g1_flat
        self.gt1_flat = gt1_flat
        self.vrhox1_flat=vrhox1_flat
        self.g2_flat = g2_flat
        self.gt2_flat = gt2_flat
        self.vrhox2_flat=vrhox2_flat
        self.g3_flat = g3_flat
        self.gt3_flat = gt3_flat
        self.vrhox3_flat=vrhox3_flat
        self.g4_flat = g4_flat
        self.gt4_flat = gt4_flat
        self.vrhox4_flat=vrhox4_flat
        self.g5_flat = g5_flat
        self.gt5_flat = gt5_flat
        self.vrhox5_flat=vrhox5_flat
        self.g1_tf = tf.placeholder(tf.float32, shape=[Nv*(Nx-2*pad),1])
        self.g2_tf = tf.placeholder(tf.float32, shape=[Nv*(Nx-2*pad),1])
        self.g3_tf = tf.placeholder(tf.float32, shape=[Nv*(Nx-2*pad),1])
        self.g4_tf = tf.placeholder(tf.float32, shape=[Nv*(Nx-2*pad),1])
        self.g5_tf = tf.placeholder(tf.float32, shape=[Nv*(Nx-2*pad),1])



        # Initialize NNs
        self.weights1, self.biases1,self.weights2, self.biases2,self.weights3, self.biases3,self.weights4, self.biases4,self.weights5, self.biases5, self.D1, self.D2, self.c1,self.c2,self.c3,self.c4 = self.initialize_NN(layers)

        self.g1_pred, self.g2_pred, self.g3_pred,self.g4_pred,self.g5_pred, self.pinn1, self.pinn2, self.pinn3, self.pinn4, self.pinn5 = self.net_uv(self.x_flat)

        #Define the Loss
#        self.loss = tf.reduce_mean(tf.abs(self.V_pred[pad:Nx-pad,pad:Nx-pad]-self.V[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(self.Vx_pred[pad:Nx-pad,pad:Nx-pad]-self.Vx[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))

        self.loss =0.000001*tf.reduce_mean(tf.square(self.pinn1))+0.000001*tf.reduce_mean(tf.square(self.pinn2))+0.000001*tf.reduce_mean(tf.square(self.pinn3))+0.000001*tf.reduce_mean(tf.square(self.pinn4))+0.000001*tf.reduce_mean(tf.square(self.pinn5))+tf.reduce_mean(tf.square(self.g1_pred-self.g1_tf))+tf.reduce_mean(tf.square(self.g2_pred-self.g2_tf))+tf.reduce_mean(tf.square(self.g3_pred-self.g3_tf))+tf.reduce_mean(tf.square(self.g4_pred-self.g4_tf))+tf.reduce_mean(tf.square(self.g5_pred-self.g5_tf))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))
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
        weights1 = []
        biases1 = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights1.append(W)
            biases1.append(b)
        weights2 = []
        biases2 = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights2.append(W)
            biases2.append(b)
        weights3 = []
        biases3 = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights3.append(W)
            biases3.append(b)
        weights4 = []
        biases4 = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights4.append(W)
            biases4.append(b)
        weights5 = []
        biases5 = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights5.append(W)
            biases5.append(b)            
            
            
        D1=0.0*tf.Variable(tf.ones([Nv*(Nx-2*pad),1],dtype=tf.float32), dtype=tf.float32)
        D2=0.0*tf.Variable(tf.ones([Nv*(Nx-2*pad),1],dtype=tf.float32), dtype=tf.float32)
        init=0.5
        c1=init*tf.Variable(1.0, dtype=tf.float32)
        c2=init*tf.Variable(1.0, dtype=tf.float32)
        c3=init*tf.Variable(1.0, dtype=tf.float32)
        c4=init*tf.Variable(1.0, dtype=tf.float32)
        return weights1,biases1,weights2,biases2,weights3,biases3,weights4,biases4,weights5,biases5,D1,D2,c1,c2,c3,c4
        
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
        gout1 = self.neural_net(self.weights1, self.biases1, self.D1, self.D2)
        g_x1=tf.gradients(gout1,self.D2)[0]
        vgx=v_flat*g_x1
        Temp = (self.wv_flat[0:Nx-2*pad]*vgx[0:Nx-2*pad]+self.wv_flat[Nx-2*pad:2*Nx-4*pad]*vgx[Nx-2*pad:2*Nx-4*pad]+self.wv_flat[2*Nx-4*pad:3*Nx-6*pad]*vgx[2*Nx-4*pad:3*Nx-6*pad]+self.wv_flat[3*Nx-6*pad:4*Nx-8*pad]*vgx[3*Nx-6*pad:4*Nx-8*pad]+self.wv_flat[4*Nx-8*pad:5*Nx-10*pad]*vgx[4*Nx-8*pad:5*Nx-10*pad]+self.wv_flat[5*Nx-10*pad:6*Nx-12*pad]*vgx[5*Nx-10*pad:6*Nx-12*pad]+self.wv_flat[6*Nx-12*pad:7*Nx-14*pad]*vgx[6*Nx-12*pad:7*Nx-14*pad]+self.wv_flat[7*Nx-14*pad:8*Nx-16*pad]*vgx[7*Nx-14*pad:8*Nx-16*pad]+self.wv_flat[8*Nx-16*pad:9*Nx-18*pad]*vgx[8*Nx-16*pad:9*Nx-18*pad]+self.wv_flat[9*Nx-18*pad:10*Nx-20*pad]*vgx[9*Nx-18*pad:10*Nx-20*pad]+self.wv_flat[10*Nx-20*pad:11*Nx-22*pad]*vgx[10*Nx-20*pad:11*Nx-22*pad]+self.wv_flat[11*Nx-22*pad:12*Nx-24*pad]*vgx[11*Nx-22*pad:12*Nx-24*pad]+self.wv_flat[12*Nx-24*pad:13*Nx-26*pad]*vgx[12*Nx-24*pad:13*Nx-26*pad]+self.wv_flat[13*Nx-26*pad:14*Nx-28*pad]*vgx[13*Nx-26*pad:14*Nx-28*pad]+self.wv_flat[14*Nx-28*pad:15*Nx-30*pad]*vgx[14*Nx-28*pad:15*Nx-30*pad]+self.wv_flat[15*Nx-30*pad:16*Nx-32*pad]*vgx[15*Nx-30*pad:16*Nx-32*pad])/2
        pvgx=tf.concat([Temp,Temp],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx1=tf.concat([pvgx,pvgx],0)
        pinn1=gt1_flat+self.c1*v_flat*g_x1+self.c2*pvgx1+self.c3*vrhox1_flat+self.c4*gout1

        gout2 = self.neural_net(self.weights2, self.biases2, self.D1, self.D2)
        g_x2=tf.gradients(gout2,self.D2)[0]
        vgx=v_flat*g_x2
        Temp = (self.wv_flat[0:Nx-2*pad]*vgx[0:Nx-2*pad]+self.wv_flat[Nx-2*pad:2*Nx-4*pad]*vgx[Nx-2*pad:2*Nx-4*pad]+self.wv_flat[2*Nx-4*pad:3*Nx-6*pad]*vgx[2*Nx-4*pad:3*Nx-6*pad]+self.wv_flat[3*Nx-6*pad:4*Nx-8*pad]*vgx[3*Nx-6*pad:4*Nx-8*pad]+self.wv_flat[4*Nx-8*pad:5*Nx-10*pad]*vgx[4*Nx-8*pad:5*Nx-10*pad]+self.wv_flat[5*Nx-10*pad:6*Nx-12*pad]*vgx[5*Nx-10*pad:6*Nx-12*pad]+self.wv_flat[6*Nx-12*pad:7*Nx-14*pad]*vgx[6*Nx-12*pad:7*Nx-14*pad]+self.wv_flat[7*Nx-14*pad:8*Nx-16*pad]*vgx[7*Nx-14*pad:8*Nx-16*pad]+self.wv_flat[8*Nx-16*pad:9*Nx-18*pad]*vgx[8*Nx-16*pad:9*Nx-18*pad]+self.wv_flat[9*Nx-18*pad:10*Nx-20*pad]*vgx[9*Nx-18*pad:10*Nx-20*pad]+self.wv_flat[10*Nx-20*pad:11*Nx-22*pad]*vgx[10*Nx-20*pad:11*Nx-22*pad]+self.wv_flat[11*Nx-22*pad:12*Nx-24*pad]*vgx[11*Nx-22*pad:12*Nx-24*pad]+self.wv_flat[12*Nx-24*pad:13*Nx-26*pad]*vgx[12*Nx-24*pad:13*Nx-26*pad]+self.wv_flat[13*Nx-26*pad:14*Nx-28*pad]*vgx[13*Nx-26*pad:14*Nx-28*pad]+self.wv_flat[14*Nx-28*pad:15*Nx-30*pad]*vgx[14*Nx-28*pad:15*Nx-30*pad]+self.wv_flat[15*Nx-30*pad:16*Nx-32*pad]*vgx[15*Nx-30*pad:16*Nx-32*pad])/2
        pvgx=tf.concat([Temp,Temp],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx2=tf.concat([pvgx,pvgx],0)
        pinn2=gt2_flat+self.c1*v_flat*g_x2+self.c2*pvgx2+self.c3*vrhox2_flat+self.c4*gout2

        gout3 = self.neural_net(self.weights3, self.biases3, self.D1, self.D2)
        g_x3=tf.gradients(gout3,self.D2)[0]
        vgx=v_flat*g_x3
        Temp = (self.wv_flat[0:Nx-2*pad]*vgx[0:Nx-2*pad]+self.wv_flat[Nx-2*pad:2*Nx-4*pad]*vgx[Nx-2*pad:2*Nx-4*pad]+self.wv_flat[2*Nx-4*pad:3*Nx-6*pad]*vgx[2*Nx-4*pad:3*Nx-6*pad]+self.wv_flat[3*Nx-6*pad:4*Nx-8*pad]*vgx[3*Nx-6*pad:4*Nx-8*pad]+self.wv_flat[4*Nx-8*pad:5*Nx-10*pad]*vgx[4*Nx-8*pad:5*Nx-10*pad]+self.wv_flat[5*Nx-10*pad:6*Nx-12*pad]*vgx[5*Nx-10*pad:6*Nx-12*pad]+self.wv_flat[6*Nx-12*pad:7*Nx-14*pad]*vgx[6*Nx-12*pad:7*Nx-14*pad]+self.wv_flat[7*Nx-14*pad:8*Nx-16*pad]*vgx[7*Nx-14*pad:8*Nx-16*pad]+self.wv_flat[8*Nx-16*pad:9*Nx-18*pad]*vgx[8*Nx-16*pad:9*Nx-18*pad]+self.wv_flat[9*Nx-18*pad:10*Nx-20*pad]*vgx[9*Nx-18*pad:10*Nx-20*pad]+self.wv_flat[10*Nx-20*pad:11*Nx-22*pad]*vgx[10*Nx-20*pad:11*Nx-22*pad]+self.wv_flat[11*Nx-22*pad:12*Nx-24*pad]*vgx[11*Nx-22*pad:12*Nx-24*pad]+self.wv_flat[12*Nx-24*pad:13*Nx-26*pad]*vgx[12*Nx-24*pad:13*Nx-26*pad]+self.wv_flat[13*Nx-26*pad:14*Nx-28*pad]*vgx[13*Nx-26*pad:14*Nx-28*pad]+self.wv_flat[14*Nx-28*pad:15*Nx-30*pad]*vgx[14*Nx-28*pad:15*Nx-30*pad]+self.wv_flat[15*Nx-30*pad:16*Nx-32*pad]*vgx[15*Nx-30*pad:16*Nx-32*pad])/2
        pvgx=tf.concat([Temp,Temp],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx3=tf.concat([pvgx,pvgx],0)
        pinn3=gt3_flat+self.c1*v_flat*g_x3+self.c2*pvgx3+self.c3*vrhox3_flat+self.c4*gout3
        
        gout4 = self.neural_net(self.weights4, self.biases4, self.D1, self.D2)
        g_x4=tf.gradients(gout4,self.D2)[0]
        vgx=v_flat*g_x4
        Temp = (self.wv_flat[0:Nx-2*pad]*vgx[0:Nx-2*pad]+self.wv_flat[Nx-2*pad:2*Nx-4*pad]*vgx[Nx-2*pad:2*Nx-4*pad]+self.wv_flat[2*Nx-4*pad:3*Nx-6*pad]*vgx[2*Nx-4*pad:3*Nx-6*pad]+self.wv_flat[3*Nx-6*pad:4*Nx-8*pad]*vgx[3*Nx-6*pad:4*Nx-8*pad]+self.wv_flat[4*Nx-8*pad:5*Nx-10*pad]*vgx[4*Nx-8*pad:5*Nx-10*pad]+self.wv_flat[5*Nx-10*pad:6*Nx-12*pad]*vgx[5*Nx-10*pad:6*Nx-12*pad]+self.wv_flat[6*Nx-12*pad:7*Nx-14*pad]*vgx[6*Nx-12*pad:7*Nx-14*pad]+self.wv_flat[7*Nx-14*pad:8*Nx-16*pad]*vgx[7*Nx-14*pad:8*Nx-16*pad]+self.wv_flat[8*Nx-16*pad:9*Nx-18*pad]*vgx[8*Nx-16*pad:9*Nx-18*pad]+self.wv_flat[9*Nx-18*pad:10*Nx-20*pad]*vgx[9*Nx-18*pad:10*Nx-20*pad]+self.wv_flat[10*Nx-20*pad:11*Nx-22*pad]*vgx[10*Nx-20*pad:11*Nx-22*pad]+self.wv_flat[11*Nx-22*pad:12*Nx-24*pad]*vgx[11*Nx-22*pad:12*Nx-24*pad]+self.wv_flat[12*Nx-24*pad:13*Nx-26*pad]*vgx[12*Nx-24*pad:13*Nx-26*pad]+self.wv_flat[13*Nx-26*pad:14*Nx-28*pad]*vgx[13*Nx-26*pad:14*Nx-28*pad]+self.wv_flat[14*Nx-28*pad:15*Nx-30*pad]*vgx[14*Nx-28*pad:15*Nx-30*pad]+self.wv_flat[15*Nx-30*pad:16*Nx-32*pad]*vgx[15*Nx-30*pad:16*Nx-32*pad])/2
        pvgx=tf.concat([Temp,Temp],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx4=tf.concat([pvgx,pvgx],0)
        pinn4=gt4_flat+self.c1*v_flat*g_x4+self.c2*pvgx4+self.c3*vrhox4_flat+self.c4*gout4        
        
        gout5 = self.neural_net(self.weights5, self.biases5, self.D1, self.D2)
        g_x5=tf.gradients(gout5,self.D2)[0]
        vgx=v_flat*g_x5
        Temp = (self.wv_flat[0:Nx-2*pad]*vgx[0:Nx-2*pad]+self.wv_flat[Nx-2*pad:2*Nx-4*pad]*vgx[Nx-2*pad:2*Nx-4*pad]+self.wv_flat[2*Nx-4*pad:3*Nx-6*pad]*vgx[2*Nx-4*pad:3*Nx-6*pad]+self.wv_flat[3*Nx-6*pad:4*Nx-8*pad]*vgx[3*Nx-6*pad:4*Nx-8*pad]+self.wv_flat[4*Nx-8*pad:5*Nx-10*pad]*vgx[4*Nx-8*pad:5*Nx-10*pad]+self.wv_flat[5*Nx-10*pad:6*Nx-12*pad]*vgx[5*Nx-10*pad:6*Nx-12*pad]+self.wv_flat[6*Nx-12*pad:7*Nx-14*pad]*vgx[6*Nx-12*pad:7*Nx-14*pad]+self.wv_flat[7*Nx-14*pad:8*Nx-16*pad]*vgx[7*Nx-14*pad:8*Nx-16*pad]+self.wv_flat[8*Nx-16*pad:9*Nx-18*pad]*vgx[8*Nx-16*pad:9*Nx-18*pad]+self.wv_flat[9*Nx-18*pad:10*Nx-20*pad]*vgx[9*Nx-18*pad:10*Nx-20*pad]+self.wv_flat[10*Nx-20*pad:11*Nx-22*pad]*vgx[10*Nx-20*pad:11*Nx-22*pad]+self.wv_flat[11*Nx-22*pad:12*Nx-24*pad]*vgx[11*Nx-22*pad:12*Nx-24*pad]+self.wv_flat[12*Nx-24*pad:13*Nx-26*pad]*vgx[12*Nx-24*pad:13*Nx-26*pad]+self.wv_flat[13*Nx-26*pad:14*Nx-28*pad]*vgx[13*Nx-26*pad:14*Nx-28*pad]+self.wv_flat[14*Nx-28*pad:15*Nx-30*pad]*vgx[14*Nx-28*pad:15*Nx-30*pad]+self.wv_flat[15*Nx-30*pad:16*Nx-32*pad]*vgx[15*Nx-30*pad:16*Nx-32*pad])/2
        pvgx=tf.concat([Temp,Temp],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx5=tf.concat([pvgx,pvgx],0)
        pinn5=gt5_flat+self.c1*v_flat*g_x5+self.c2*pvgx5+self.c3*vrhox5_flat+self.c4*gout5        
        
        
        

        return gout1,gout2,gout3,gout4,gout5,pinn1,pinn2,pinn3,pinn4,pinn5


    def callback(self, loss):
        print('Loss:', loss)


    def train(self, nIter):
        tf_dict = {self.g1_tf: g1_flat,
                   self.g2_tf: g2_flat,
                   self.g3_tf: g3_flat,
                   self.g4_tf: g4_flat,
                   self.g5_tf: g5_flat}
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
        g1_pred=self.sess.run(self.g1_pred)
        g2_pred=self.sess.run(self.g2_pred)
        g3_pred=self.sess.run(self.g3_pred)
        c1=self.sess.run(self.c1)
        c2=self.sess.run(self.c2)
        c3=self.sess.run(self.c3)
        c4=self.sess.run(self.c4)
        return g1_pred, g2_pred, g3_pred,c1,c2,c3,c4
    
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

    Time1=23
    Time2=24
    Time3=25
    Time4=26
    Time5=27 
    
    #fitting at a random point in time
    #Since the neural network has trouble fitting
    #to all points in time
    
    pad=122
    rho1=np.zeros([Nv,Nx],dtype=float)
    rho2=np.zeros([Nv,Nx],dtype=float)
    rho3=np.zeros([Nv,Nx],dtype=float)
    rho4=np.zeros([Nv,Nx],dtype=float)
    rho5=np.zeros([Nv,Nx],dtype=float)
    vrhox1=np.zeros([Nv,Nx],dtype=float)
    vrhox2=np.zeros([Nv,Nx],dtype=float)
    vrhox3=np.zeros([Nv,Nx],dtype=float) 
    vrhox4=np.zeros([Nv,Nx],dtype=float) 
    vrhox5=np.zeros([Nv,Nx],dtype=float)    
    for I in range(0,Nv):
        for J in range(0,Nx):
            rho1[I,J]=rho_Data[J,Time1]
            rho2[I,J]=rho_Data[J,Time2]
            rho3[I,J]=rho_Data[J,Time3]
            rho4[I,J]=rho_Data[J,Time4]
            rho5[I,J]=rho_Data[J,Time5]

    rhox31=(np.roll(rho1,1,1)-np.roll(rho1,-1,1))/(2*hx)
    rhox32=(np.roll(rho2,1,1)-np.roll(rho2,-1,1))/(2*hx)
    rhox33=(np.roll(rho3,1,1)-np.roll(rho3,-1,1))/(2*hx)
    rhox34=(np.roll(rho4,1,1)-np.roll(rho4,-1,1))/(2*hx)
    rhox35=(np.roll(rho5,1,1)-np.roll(rho5,-1,1))/(2*hx)
    for I in range(0,Nv):
        for J in range(0,Nx):
            vrhox1[I,J]=v[I]*rhox31[I,J]
            vrhox2[I,J]=v[I]*rhox32[I,J]
            vrhox3[I,J]=v[I]*rhox33[I,J]
            vrhox4[I,J]=v[I]*rhox34[I,J]
            vrhox5[I,J]=v[I]*rhox35[I,J]

    gv3=(np.roll(gData,1,0)-np.roll(gData,-1,0))/(2*hv)
    gx3=(np.roll(gData,1,1)-np.roll(gData,-1,1))/(2*hx)
    gt33=(np.roll(gData,1,2)-np.roll(gData,-1,2))/(2*dt)

    
    g1=np.zeros([Nv,Nx],dtype=float)
    g2=np.zeros([Nv,Nx],dtype=float)
    g3=np.zeros([Nv,Nx],dtype=float)
    g4=np.zeros([Nv,Nx],dtype=float)
    g5=np.zeros([Nv,Nx],dtype=float)
    
    gt1=np.zeros([Nv,Nx],dtype=float)
    gt2=np.zeros([Nv,Nx],dtype=float)
    gt3=np.zeros([Nv,Nx],dtype=float)
    gt4=np.zeros([Nv,Nx],dtype=float)
    gt5=np.zeros([Nv,Nx],dtype=float)

    for I in range(0,Nv):
        for J in range(0,Nx):
            g1[I,J]=gData[I,J,Time1]
            g2[I,J]=gData[I,J,Time2]
            g3[I,J]=gData[I,J,Time3]
            g4[I,J]=gData[I,J,Time4]
            g5[I,J]=gData[I,J,Time5]
            gt1[I,J]=gt33[I,J,Time1]
            gt2[I,J]=gt33[I,J,Time2]
            gt3[I,J]=gt33[I,J,Time3]
            gt4[I,J]=gt33[I,J,Time4]
            gt5[I,J]=gt33[I,J,Time5]


            
    v_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    wv_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    x_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    g1_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    gt1_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    g2_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    gt2_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    g3_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    gt3_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)    
    g4_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    gt4_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    g5_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)
    gt5_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)

    vrhox1_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)    
    vrhox2_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)    
    vrhox3_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)    
    vrhox4_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)    
    vrhox5_flat=np.zeros([Nv*(Nx-2*pad),1],dtype=float)        
    
    L=0
    for I in range(0,Nv):
        for J in range(pad,Nx-pad):
            v_flat[L,0]=v[I,0]
            wv_flat[L,0]=wv[I,0]
            x_flat[L,0]=x[J,0]
            g1_flat[L,0]=g1[I,J]
            gt1_flat[L,0]=gt1[I,J]
            vrhox1_flat[L,0]=vrhox1[I,J]
            g2_flat[L,0]=g2[I,J]
            gt2_flat[L,0]=gt2[I,J]
            vrhox2_flat[L,0]=vrhox2[I,J]
            g3_flat[L,0]=g3[I,J]
            gt3_flat[L,0]=gt3[I,J]
            vrhox3_flat[L,0]=vrhox3[I,J]
            g4_flat[L,0]=g4[I,J]
            gt4_flat[L,0]=gt4[I,J]
            vrhox4_flat[L,0]=vrhox4[I,J]
            g5_flat[L,0]=g5[I,J]
            gt5_flat[L,0]=gt5[I,J]
            vrhox5_flat[L,0]=vrhox5[I,J]
            L=L+1



    #This is what Nsteps does
    Nsteps=200000

    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model = NeuralNetworkCode(v_flat,wv_flat,x_flat,g1_flat,gt1_flat,vrhox1_flat,g2_flat,gt2_flat,vrhox2_flat,g3_flat,gt3_flat,vrhox3_flat,g4_flat,gt4_flat,vrhox4_flat,g5_flat,gt5_flat,vrhox5_flat,layers,Losss)
        
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
    print('The learned IMEX1 G-equation is:')
    print(c4,'g +',c1,'v*g_x+',c2,'<vg_x>',c3,'vrho_x')

    ep=np.int(4)
    ep2=ep**2

    cc1=abs(c4)-ep2
    cc2=abs(c1)-ep
    cc3=abs(c2)-ep
    cc4=abs(c3)-ep2
    
    error=(abs(cc1)+abs(cc2)+abs(cc3)+abs(cc4))/(2*ep2+2*ep)*100
    
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













