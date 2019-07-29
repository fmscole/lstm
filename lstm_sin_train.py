import numpy as np
from LSTM.lstm import LstmParam, LstmNetwork
import pickle as pk

class Loss:
    def __init__(self,mem_cell_ct):
        self.v=np.zeros(mem_cell_ct)
    
    def value(self, pred):
        out=self.v.dot(pred)  
        return out
    def loss(self, pred, label):
        out=self.value(pred)  
        return (out- label) ** 2


    def bottom_diff(self, pred, label):
        out=self.value(pred)
        df = 2 * (out - label)

        diff=df*self.v
        self.v-=0.01*pred*df
        return diff
T = 50
x_dim = 40

def train():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    np.random.seed(2)

    
    L = 100
    N = 500

    x = np.empty((N, L), 'int64')
    t=np.arange(N)
    np.random.shuffle(t)
    x = np.array(range(L)) +t.reshape(N, 1)
    w=2*np.pi/T
    data = np.sin(w*x).astype('float64')

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    loss=Loss(mem_cell_ct)
    epoch=1
    for cur_iter in range(epoch):
        for n in range(N):
            input_val_arr = data[n, :len(data)-x_dim]
            y_list =data[n, x_dim:]
            print("iter", "%2s" % str(cur_iter), end=": ")
            for ind in range(len(y_list)):
                lstm_net.x_list_add(input_val_arr[ind:ind+x_dim])

            lossv = lstm_net.y_list_is(y_list, loss)
            print("loss:", "%.3e" % lossv)
            lstm_param.apply_diff(lr=0.01)
            lstm_net.x_list_clear()
        #模型保存
        if cur_iter % 10 ==0:
            fs = open('model/model_%d.pkl'%cur_iter, 'wb')
            pk.dump(lstm_param,fs)
            pk.dump(loss,fs)
            fs.close()

if __name__ == "__main__":
    train()