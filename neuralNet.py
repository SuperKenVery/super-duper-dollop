import numpy as np
import time,f,pdb



class layer:
    def __init__(self,inputs,neurons,w=None,b=None,activate=f.relu,loss=f.loss):
        self.inputs=inputs
        self.neurons=neurons
        self.activate=activate
        self.loss=loss
        if not w:
            self.w=np.random.randn(neurons,inputs)*0.01
        else:
            self.w=w.reshape((neurons,inputs))

        if not b:
            self.b=np.zeros((neurons,1))
        else:
            self.b=b.reshape((neurons,1))

        self.data=np.zeros(inputs)
        self.output=np.zeros(self.neurons)
        self.answers=np.zeros(self.output.shape)
    def forward(self):
        self.wx=np.dot(self.w,self.data)
        self.z=self.wx+self.b
        a=self.activate(self.z)
        self.output=a
    def reverse(self,nextLayer,lr):
        #lr:learning rate
        self.da=np.dot(nextLayer.w.T,nextLayer.dz)
        #最后一层：last layer.dz = last layer.a-y
        self.dz=self.da*self.activate.reverse(self.output,self.z)
        self.db=self.dz.sum(axis=1,keepdims=True)/self.dz.shape[1]
        self.dw=np.dot(self.dz,self.data.T)/self.dz.shape[1]
        self.w-=lr*self.dw
        self.b-=lr*self.db

class neuralNet:
    def __init__(self,inputs,neurons):
        #neurons: a list indicating the /*number*\ of neurons of /*each layer*\
        #So if inputs==2, neurons==[4,6], the net looks like this:
        #     O
        #  O  O
        #  O  O
        #  O  O
        #  O  O
        #     O
        self.layers=[layer(inputs,neurons[0])] + [layer(neurons[index],i) for index,i in enumerate(neurons[1:])]
    def forward(self):
        self.layers[0].forward()
        for i in range(1,len(self.layers)):
            self.layers[i].data=self.layers[i-1].output
            self.layers[i].forward()
        self.output=self.layers[-1].output
        return self.output
    def loss(self):
        return f.loss(self.output,self.y)
    def reverse(self,y,lr=0.01):
        '''
            y:  the answer
            lr: learning rate
        '''
        self.y=y
        last=self.layers[-1]
        last.dz=last.output-y
        last.dw=np.dot(last.dz,last.data.T)/last.dz.shape[1]
        last.db=last.dz
        for i in range(len(self.layers)-2,0,-1):#layers[-2] to layers[1]
            self.layers[i].reverse(self.layers[i+1],lr)
    def train(self,data,y,times,step=None):
        '''
            data:   train data
            y:      the answer
            times:  how many times to train
            step:   how many times to train before printing loss
                    leave None means never print
        '''
        self.layers[0].data=data
        for time in range(1,times+1):
            self.forward()
            self.reverse(y)
            if step and time%step==0:
                print("Trained %d times. Loss is %s. Output is %s. Answer is %s. "%(time,self.loss(),str(self.layers[-1].output),str(y)))
            
if __name__=='__main__':
    import h5py
    np.random.seed(1)
    data=(np.array([11,36,33,20,29])*100).reshape(-1,1)
    answer=np.array([0.5]).reshape(-1,1)
    net=neuralNet(5,[10,10,20,1])
    #5 inputs, 3 layers

    net.train(data,answer,1000,100)
    
    #net.train(data,answer,100,10)
