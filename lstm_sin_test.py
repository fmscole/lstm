import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LSTM.lstm import LstmParam, LstmNetwork
from lstm_sin_train import T,x_dim
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
        df = 2 * (out - label)/self.v.shape[0] 
        diff=df*self.v
        self.v-=1.0*pred*df
        return diff


def test():
    '''
    结果保存在当前目录下的pdf文件predictXXX-XXX.pdf中
    '''
    fl = open('model/model_0.pkl', 'rb')
    lstm_param=pk.load(fl)
    loss=pk.load(fl)
    fl.close()

    lstm_net = LstmNetwork(lstm_param)
    
    L = 100
    F=1000
    x= np.array(range(L))
    w=2*np.pi/T
    input_val_arr =0.5*np.sin(w*x).astype('float64')
    y_list =input_val_arr
    L=len(input_val_arr)
    
    for i in range(F):
        
        for ind in range(L-x_dim):
            lstm_net.x_list_add(input_val_arr[ind:ind+x_dim])
        
        y=loss.value(lstm_net.lstm_node_list[-1].state.h)
        input_val_arr=np.hstack((input_val_arr,[y]))
        input_val_arr=input_val_arr[1:]
        lstm_net.x_list_clear()
        y_list=np.hstack((y_list,[y]))
        # print(i,end=" ")

    print("here1")
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    def draw(yi, color):
        plt.plot(np.arange(L), yi[:L], color, linewidth = 2.0)
        plt.plot(np.arange(L, L+F), yi[L:], color + ':', linewidth = 2.0)
    draw(y_list, 'r')
    
    plt.savefig(r'predict%d-%d.pdf'%(L,F))
    plt.show()
    plt.close()

if __name__ == "__main__":
    test()