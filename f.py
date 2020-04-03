import numpy as np
import math

class rf:#reversable function
    def __init__(self,function,reverse):
        self.function=function
        self.reverse=reverse
    def __call__(self,*argv,**argvs):
        return self.function(*argv,**argvs)
    def reverse(self,*argv,**argvs):
        return self.reverse(*argv,**argvs)

sigmoid=rf(
    function=lambda z: 1/(1+np.exp(-z)),
    reverse=lambda a,z: a*(1-a)
    )

tanh=rf(
    function=lambda z: np.tanh(z),
    reverse=lambda a,z: 1-a**2
    )

def _relu_func(z):
    relu=z.copy()
    relu[z<0]=0
    return relu
def _relu_reverse(a,z):
    relu_rev=np.zeros(a.shape)
    relu_rev[z>=0]=1
    return relu_rev
relu=rf(
    function=_relu_func,
    reverse=_relu_reverse
    )

def _leakyrelu_func(z):
    lr=z.copy()
    lr[z<0]*=0.01
    return lr
def _leakyrelu_reverse(a,z):
    lrr=np.zeros(a.shape)
    lrr[z>=0]=1
    lrr[z<0]=0.01
    return lrr
    
leaky_relu=rf(
    function=_leakyrelu_func,
    reverse=_leakyrelu_reverse
    )


#前面的：z动一点，a动多少？
def _loss(a,y):
    left=-(a*np.log(y))
    right=(1-a)*np.log(1-y)
    return left+right
def _loss_reverse(a,y):
    #a动一点点，L动多少？
    first=-(y/a)
    second=(1-y)/(1-a)
    return first+second
loss=rf(
    function=_loss,
    reverse=_loss_reverse
    )

if __name__=='__main__':
    np.random.seed(1)
    data=np.random.randn(10)
    tests={
        #I know this will be slow, but it's a test anyway
        #So beauty is more important than speed :D
        
        'data':             data,
        
        'sigmoid':          sigmoid(data),
        'sigmoid_rev':      sigmoid.reverse(sigmoid(data),data),
        
        'relu':             relu(data),
        'relu_rev':         relu.reverse(relu(data),data),
        
        'tanh':             tanh(data),
        'tanh_rev':         tanh.reverse(tanh(data),data),

        'leaky-relu':       leaky_relu(data),
        'leaky-relu_rev':   leaky_relu.reverse(leaky_relu(data),data),

        'loss':             loss(data,np.ones(data.shape)*0.5),
        'loss-rev':         loss.reverse(data,np.ones(data.shape)*0.5),
        }
    for i in tests:
        print(i,'\t'*3,str(tests[i]).replace('\n',' '))
        
