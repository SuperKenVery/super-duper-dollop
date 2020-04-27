import numpy as np
import time,f,pdb
'''
目前存在的问题：
    反向传播正确的前提，是最后一层使用sigmoid激活函数。
'''


class layer:
    def __init__(self,inputs,neurons,w=None,b=None,activate=f.relu,loss=f.loss):
        self.inputs=inputs
        self.neurons=neurons
        self.activate=activate
        self.loss=loss
        if not w:
            self.w=np.random.randn(neurons,inputs) / np.sqrt(inputs)
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
        #a=self.activate(self.z)
        self.output=self.activate(self.z)
    def reverse(self,nextLayer,lr):
        #lr:learning rate
        self.da=np.dot(nextLayer.w.T,nextLayer.dz)
        #最后一层：last layer.dz = last layer.a-y
        self.dz=self.da*self.activate.reverse(self.output,self.z)
        self.db=self.dz.sum(axis=1,keepdims=True)/self.dz.shape[1]
        self.dw=np.dot(self.dz,self.data.T)/self.dz.shape[1]
        self.w=self.w-lr*self.dw
        self.b=self.b-lr*self.db

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
        self.layers[-1].activate=f.sigmoid
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
        last.db=last.dz.sum(axis=1,keepdims=True)/last.dz.shape[1]
        for i in range(len(self.layers)-2,0,-1):#layers[-2] to layers[1]
            self.layers[i].reverse(self.layers[i+1],lr)
    def train(self,data,y,times,step=None,lr=0.01):
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
            self.reverse(y,lr)
            if step and time%step==0:
                trainMessage="Trained %d times. "%time
                lossMessage="Loss is %s. "%self.loss()
                outputMessage="Output is %s. "%str(self.layers[-1].output)
                answerMessage="Answer is %s. "%str(y)
                start=stop='\n==============================\n'
                print(start,lossMessage,stop)

def create_net_to_train(filename,layers,neuron_per_layer,datakey='train_set_x',answerkey='train_set_y'):
    import loader
    data,answer,groups=loader.load(filename,datakey,answerkey)
    data=data/255
    class fixed_io_neural_net(neuralNet):
        def train(self,times,step=None):
            neuralNet.train(self,data,answer,times,step)
    layers=[neuron_per_layer for i in range(layers)]
    layers[-1]=answer.shape[0]
    net=fixed_io_neural_net(data.shape[0],layers)
    return net

if __name__=='__main__':
    def neural_net_test():
        np.random.seed(1)
        data=(np.array([11,36,33,20,29])*100).reshape(-1,1)
        answer=np.array([0.5]).reshape(-1,1)
        net=neuralNet(5,[10,10,20,1])
        #5 inputs, 3 layers
        net.train(data,answer,1000,100)
        #net.train(data,answer,100,10)
    
    def data_test():
        net=create_net_to_train('examples/dnn/datasets/train_catvnoncat.h5',30,30)
        net.train(100,10)
    def data_test_without_createnettotrain():
        import loader
        print("loading data")
        data,answer,groups=loader.load('examples/dnn/datasets/train_catvnoncat.h5',
                                    'train_set_x','train_set_y')
        #data=data[:,0].reshape(-1,1)
        #answer=answer[:,0].reshape(-1,1)
        groups=1
        outputs=answer.shape[0]
        inputs=data.shape[0]
        print("creating the net")
        neurons=[100,200,outputs]
        net=neuralNet(inputs,neurons)
        print("start training")
        net.train(data,answer,1000,100,0.001)

    data_test_without_createnettotrain()
        

