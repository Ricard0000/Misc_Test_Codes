
import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import scipy.io


def Normal_CDF(inpt):
    out=0.5*(1+math.erf(inpt/np.sqrt(2)))
    return out


class NeuralNetworkCode:
    def __init__(self,v_flat,x_flat,t_flat,g_flat,rho_flat,layers,Losss,wv_flat,vrhox_flat,gt_flat,mask_flat):

        self.layers = layers
        self.Losss=Losss
        self.v_flat=v_flat
        self.x_flat=x_flat
        self.t_flat=t_flat
        self.g_flat = g_flat
        self.rho_flat = rho_flat
        self.wv_flat=wv_flat
        self.vrhox_flat=vrhox_flat
        self.gt_flat=gt_flat
        self.mask_flat=mask_flat

        self.g_tf = tf.placeholder(tf.float32, shape=[Nv*Nx*(n-2*pad),1])
        self.rho_tf = tf.placeholder(tf.float32, shape=[Nv*Nx*(n-2*pad),1])

        # Initialize NNs
        self.weights, self.biases, self.D1, self.D2, self.D3, self.c1, self.c2, self.c3, self.c4 = self.initialize_NN(layers)

        self.g_pred, self.gv_pred, self.gx_pred, self.gt_pred, self.pinn = self.net_uv(self.x_flat)

        #Define the Loss
#        self.loss = tf.reduce_mean(tf.abs(self.V_pred[pad:Nx-pad,pad:Nx-pad]-self.V[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(self.Vx_pred[pad:Nx-pad,pad:Nx-pad]-self.Vx[pad:Nx-pad,pad:Nx-pad]))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))

#        self.loss =tf.reduce_mean(tf.abs(self.g_pred-self.g_tf))+0.1*tf.reduce_mean(tf.abs(self.pinn))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))+tf.reduce_mean(tf.abs(0.0 - self.D3))
#        self.loss =tf.reduce_mean(tf.abs(self.g_pred-self.g_tf))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))
#        self.loss =tf.reduce_mean(tf.square(self.g_pred-self.g_tf))+0.1*tf.reduce_mean(tf.square(self.pinn))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))
        self.loss =tf.reduce_mean(tf.square(self.g_pred-self.g_tf))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))
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
        init=0.5
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        D1=0.0*tf.Variable(tf.ones([Nv*Nx*(n-2*pad),1],dtype=tf.float32), dtype=tf.float32)
        D2=0.0*tf.Variable(tf.ones([Nv*Nx*(n-2*pad),1],dtype=tf.float32), dtype=tf.float32)
        D3=0.0*tf.Variable(tf.ones([Nv*Nx*(n-2*pad),1],dtype=tf.float32), dtype=tf.float32)
        c1=init*tf.Variable(1.0, dtype=tf.float32)
        c2=init*tf.Variable(1.0, dtype=tf.float32)
        c3=init*tf.Variable(1.0, dtype=tf.float32)
        c4=init*tf.Variable(1.0, dtype=tf.float32)
        
        return weights, biases, D1, D2, D3, c1,c2,c3,c4

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


    def neural_net(self, weights, biases, D1, D2,D3):
        V=D1+self.v_flat
        X=D2+self.x_flat
        T=D3+self.t_flat
#        H=tf.concat([V,X,T], 1)
        H=tf.concat([V,X], 1)
        num_layers = len(weights) + 1
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
#            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
#            H = tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)    
        return H


    def net_uv(self, x_flat ):
        g = self.neural_net(self.weights, self.biases, self.D1, self.D2, self.D3)
        g_v=tf.gradients(g,self.D1)[0]
        g_x=tf.gradients(g,self.D2)[0]
#        g_t=tf.gradients(g,self.D3)[0]
            
        vgx=v_flat*g_x
        print(vgx.shape)
        print(vgx.shape)
        print(vgx.shape)
#        Temp = (self.wv_flat[0:Nx*n]*vgx[0:Nx*n]+self.wv_flat[Nx*n:2*Nx*n]*vgx[Nx*n:2*Nx*n]+self.wv_flat[2*Nx*n:3*Nx*n]*vgx[2*Nx*n:3*Nx*n]+self.wv_flat[3*Nx*n:4*Nx*n]*vgx[3*Nx*n:4*Nx*n]+self.wv_flat[4*Nx*n:5*Nx*n]*vgx[4*Nx*n:5*Nx*n]+self.wv_flat[5*Nx*n:6*Nx*n]*vgx[5*Nx*n:6*Nx*n]+self.wv_flat[6*Nx*n:7*Nx*n]*vgx[6*Nx*n:7*Nx*n]+self.wv_flat[7*Nx*n:8*Nx*n]*vgx[7*Nx*n:8*Nx*n]+self.wv_flat[8*Nx*n:9*Nx*n]*vgx[8*Nx*n:9*Nx*n]+self.wv_flat[9*Nx*n:10*Nx*n]*vgx[9*Nx*n:10*Nx*n]+self.wv_flat[10*Nx*n:11*Nx*n]*vgx[10*Nx*n:11*Nx*n]+self.wv_flat[11*Nx*n:12*Nx*n]*vgx[11*Nx*n:12*Nx*n]+self.wv_flat[12*Nx*n:13*Nx*n]*vgx[12*Nx*n:13*Nx*n]+self.wv_flat[13*Nx*n:14*Nx*n]*vgx[13*Nx*n:14*Nx*n]+self.wv_flat[14*Nx*n:15*Nx*n]*vgx[14*Nx*n:15*Nx*n]+self.wv_flat[15*Nx*n:16*Nx*n]*vgx[15*Nx*n:16*Nx*n])/2

        Temp = (self.wv_flat[0:Nx*(n-2*pad)]*vgx[0:Nx*(n-2*pad)]+self.wv_flat[Nx*(n-2*pad):2*Nx*(n-2*pad)]*vgx[Nx*(n-2*pad):2*Nx*(n-2*pad)]+self.wv_flat[2*Nx*(n-2*pad):3*Nx*(n-2*pad)]*vgx[2*Nx*(n-2*pad):3*Nx*(n-2*pad)]+self.wv_flat[3*Nx*(n-2*pad):4*Nx*(n-2*pad)]*vgx[3*Nx*(n-2*pad):4*Nx*(n-2*pad)]+self.wv_flat[4*Nx*(n-2*pad):5*Nx*(n-2*pad)]*vgx[4*Nx*(n-2*pad):5*Nx*(n-2*pad)]+self.wv_flat[5*Nx*(n-2*pad):6*Nx*(n-2*pad)]*vgx[5*Nx*(n-2*pad):6*Nx*(n-2*pad)]+self.wv_flat[6*Nx*(n-2*pad):7*Nx*(n-2*pad)]*vgx[6*Nx*(n-2*pad):7*Nx*(n-2*pad)]+self.wv_flat[7*Nx*(n-2*pad):8*Nx*(n-2*pad)]*vgx[7*Nx*(n-2*pad):8*Nx*(n-2*pad)]+self.wv_flat[8*Nx*(n-2*pad):9*Nx*(n-2*pad)]*vgx[8*Nx*(n-2*pad):9*Nx*(n-2*pad)]+self.wv_flat[9*Nx*(n-2*pad):10*Nx*(n-2*pad)]*vgx[9*Nx*(n-2*pad):10*Nx*(n-2*pad)]+self.wv_flat[10*Nx*(n-2*pad):11*Nx*(n-2*pad)]*vgx[10*Nx*(n-2*pad):11*Nx*(n-2*pad)]+self.wv_flat[11*Nx*(n-2*pad):12*Nx*(n-2*pad)]*vgx[11*Nx*(n-2*pad):12*Nx*(n-2*pad)]+self.wv_flat[12*Nx*(n-2*pad):13*Nx*(n-2*pad)]*vgx[12*Nx*(n-2*pad):13*Nx*(n-2*pad)]+self.wv_flat[13*Nx*(n-2*pad):14*Nx*(n-2*pad)]*vgx[13*Nx*(n-2*pad):14*Nx*(n-2*pad)]+self.wv_flat[14*Nx*(n-2*pad):15*Nx*(n-2*pad)]*vgx[14*Nx*(n-2*pad):15*Nx*(n-2*pad)]+self.wv_flat[15*Nx*(n-2*pad):16*Nx*(n-2*pad)]*vgx[15*Nx*(n-2*pad):16*Nx*(n-2*pad)])/2

        pvgx=tf.concat([Temp,Temp],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx=tf.concat([pvgx,pvgx],0)
        pvgx=tf.concat([pvgx,pvgx],0)#This is proj of vgx
        print(pvgx.shape)
        print(pvgx.shape)
        print(pvgx.shape)
        
#        pinn=g_t+self.c1*v_flat*g_x+self.c2*pvgx+self.c3*vrhox_flat+self.c4*g
        pinn=self.gt_flat+v_flat*g_x-pvgx+vrhox_flat+g        
        bc=g*self.mask_flat
        return g,g_v,g_x,g_x,g_v


    def callback(self, loss):
        print('Loss:', loss)


    def train(self, nIter):
        tf_dict = {self.g_tf: g_flat,
                   self.rho_tf: rho_flat}
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
        gt_pred=self.sess.run(self.gt_pred)
        c1=self.sess.run(self.c1)
        c2=self.sess.run(self.c2)
        c3=self.sess.run(self.c3)
        c4=self.sess.run(self.c4)
        return g_pred, gv_pred,gx_pred, gt_pred,c1,c2,c3,c4
    
if __name__ == "__main__": 

    
    TTdata = scipy.io.loadmat('data_mm.mat')
    x = TTdata['x']
    rho_Data = TTdata['u']
    gData = TTdata['g']
    v = TTdata['v']
    wv=TTdata['wv']
    dt=TTdata['dt']
    hx=TTdata['hx']
    hv=v[1]-v[0]
    Nv,Nx,n=gData.shape
    t=np.zeros([n,1],dtype=float)
    for I in  range(0,n):
        t[I,0]=dt*I


    layers = [2, 18, 18, 18, 18, 18, 1]
        
    #This is the exact solution. We will use this to test the accuracy 
    #of the predicted neural network solution.
    g=np.zeros([Nv,Nx,n],dtype=float)
    rho=np.zeros([Nv,Nx,n],dtype=float)
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,n):
                g[I,J,K]=gData[I,J,K]
                rho[I,J,K]=rho_Data[J,K]
  
    #Compute partial of t for time-stepping scheme
    pad=1
    partial_tg=(np.roll(g,2,1)-np.roll(g,2,-1))/(2*dt)
    L=0
    gt_flat=np.zeros([Nv*Nx*(n-2*pad),1],dtype=float)
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(1,n-1):
                gt_flat[L,0]=partial_tg[I,J,K]
                L=L+1

    #Setting boundary condition to zero
    
    mask=np.zeros([Nv,Nx,n-2*pad],dtype=float)
    mask_flat=np.zeros([Nv*Nx*(n-2*pad),1],dtype=float)
    L=0
    for I in range(0,Nv):
        for K in range(0,n-2*pad):
            mask[I,0,K]=1
            mask[I,Nx-1,K]=1
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,n-2*pad):
                mask_flat[L,0]=mask[I,J,K]
                L=L+1


    v_flat=np.zeros([Nv*Nx*(n-2*pad),1],dtype=float)
    x_flat=np.zeros([Nv*Nx*(n-2*pad),1],dtype=float)
    t_flat=np.zeros([Nv*Nx*(n-2*pad),1],dtype=float)
    g_flat=np.zeros([Nv*Nx*(n-2*pad),1],dtype=float)
    rho_flat=np.zeros([Nv*Nx*(n-2*pad),1],dtype=float)
    wv_flat=np.zeros([Nv*Nx*(n-2*pad),1],dtype=float)
    L=0
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(pad,n-pad):
                v_flat[L,0]=v[I,0]
                x_flat[L,0]=x[J,0]
                t_flat[L,0]=t[K,0]
                g_flat[L,0]=g[I,J,K]
                rho_flat[L,0]=rho[I,J,K]
                wv_flat[L,0]=wv[I,0]
                L=L+1




##Rho does not get NN fitting for fairness:
    vrhox_flat=np.zeros([Nv*Nx*(n-2*pad),1],dtype=float)
    rhox=(np.roll(rho,1,1)-np.roll(rho,1,-1))/(2*hx)

    L=0
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(pad,n-pad):
                vrhox_flat[L,0]=v[I]*rhox[I,J,K]


    #This is what Nsteps does
    Nsteps=25000

    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model = NeuralNetworkCode(v_flat,x_flat,t_flat,g_flat,rho_flat,layers,Losss,wv_flat,vrhox_flat,gt_flat,mask_flat)

    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    g_pred, gv_pred, gx_pred, gt_pred, c1,c2,c3,c4 = model.predict()


    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2)
    

    #Restructuring the data from 1-d tensor to 2-d tensor

    g_pred_3d=np.zeros([Nv,Nx,(n-2*pad)],dtype=float)
    gv_pred_3d=np.zeros([Nv,Nx,(n-2*pad)],dtype=float)
    gx_pred_3d=np.zeros([Nv,Nx,(n-2*pad)],dtype=float)
    gt_pred_3d=np.zeros([Nv,Nx,(n-2*pad)],dtype=float)
    L=0
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,(n-2*pad)):
                g_pred_3d[I,J,K]=g_pred[L,0]
                gv_pred_3d[I,J,K]=gv_pred[L,0]
                gx_pred_3d[I,J,K]=gx_pred[L,0]
                gt_pred_3d[I,J,K]=gt_pred[L,0]
                L=L+1

    gv=(np.roll(g,0,1)-np.roll(g,0,-1))/(2*hv)
    gx=(np.roll(g,1,1)-np.roll(g,1,-1))/(2*hx)
    gt=(np.roll(g,2,1)-np.roll(g,2,-1))/(2*dt)

    v_mesh,x_mesh=np.meshgrid(x,v)
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, g_pred_3d[:,:,9], color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, g[:,:,10], color='r')
    plt.title('Exact Solution',fontsize=14,fontweight='bold')


    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, gv_pred_3d[:,:,9], color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, gv[:,:,10], color='r')
    plt.title('Exact Solution',fontsize=14,fontweight='bold')



    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, gx_pred_3d[:,:,9], color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, gx[:,:,10], color='r')
    plt.title('Exact Solution',fontsize=14,fontweight='bold')

    gx_fd=(np.roll(g,1,1)-np.roll(g,1,-1))/(2*hx)


    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(v_mesh,x_mesh, gx_fd[:,:,9], color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')