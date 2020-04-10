import h5py
import numpy as np
def load(filename,datakey,answerkey):
    dataset=h5py.File(filename,mode='r')
    data=np.array(dataset[datakey][:])
    answer=np.array(dataset[answerkey][:])

    groups=data.shape[0]
    data=data.reshape(-1,groups)
    answer=answer.reshape(-1,groups)
    
    return data,answer

if __name__=='__main__':
    data,answer=load('/home/ken/Codes/AI/examples/dnn/datasets/train_catvnoncat.h5','train_set_x','train_set_y')
    print('data shape:',data.shape)
    print('answer shape:',answer.shape)