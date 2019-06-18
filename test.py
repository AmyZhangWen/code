import numpy as np

import matplotlib.pylab as plt

class Relu:
    def __init__(self):
        self.mask=None
        
    def forward(self,x):
        self.mask = (  x<=0)
        out =x.copy()
        out[self.mask]=0
        return out
        
    def backward(self,dout):
        dout[self.mask]=0
        dx=dout
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out =None
    
    def forward (self,x):
        out =1/(1+ np.exp(-x))
        self.out =out
        return out
    
    def backward(self,dout):
        return dout*(1-self.out)*self.out
    

class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None
        
    def forward(self,x):
        self.x=x
        out =np.dot(x,self.W)+self.b
        return out
    
    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)
        return dx
        
        
       
x=np.array([[0,0],[1,1]])
W=np.array([[0,0,0],[10,10,10]]) 
b=np.array([1, 2, 3])
affine_layer =Affine(W,b)
y=affine_layer.forward(x)
dy=np.array([[1, 2, 3,], [4, 5, 6]])
dx=affine_layer.backward(dy)
print(y)
print(dx)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        