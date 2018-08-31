import numpy as np

class Perceptron:

    def __init__(self,input):
        self.W = np.zeros(len(input[0])+1) ## weight vector
        ## +1 slot cause of bias in input
        self.bias = 1
        self.lr = 1 ##learning rate
        self.epoch = 100

    def predict(self,value):
        return 1 if value >= 0 else 0

    def fit(self,input,d):
        for i in range(self.epoch):
            for j in range(d.shape[0]):
                x = np.insert(input[j],0,1) # x0 = 1, for bias
                y = self.predict( np.inner(self.W,x) )
                self.W = self.W + self.lr * (d[j]-y)*x

if __name__ == '__main__':
    X = np.array([
    [0,0],[0,1],[1,0],[1,1]
    ])
    d = np.array([0,0,0,1])
    perceptron = Perceptron(X)
    perceptron.fit(X,d)
    print(perceptron.W)
