import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90

class Net(object):
    def __init__(self, num_layers, num_units):
        self.ni = NUM_FEATS
        self.no = 1
        self.nh = num_layers
        self.sizes = [self.ni] + [num_units]*num_layers + [1]
        self.W = {}
        self.B = {}
        for i in range(self.nh+1):
            self.W[i+1] = np.random.uniform(-1,1,size = (self.sizes[i], self.sizes[i+1]))
            self.B[i+1] = np.random.uniform(-1,1,size = (1, self.sizes[i+1]))
  
    def relu(self, X):
        return np.maximum(0,X)
    
    def leakyrelu(self, X):
        return np.maximum(0.01*X,X)
    
    def grad_relu(self,X):
        return (X>0) * 1.0

    def grad_leakyrelu(self,X):
    	out = np.zeros_like(X)
    	out[X<=0]=0.01
    	out[X>0]=1.0
    	return out
        # out = np.ones_like(X)
        # out[X < 0] *= 0.0001
        # return out

    def __call__(self, X):
        self.PrA = {} #preactivation
        self.PsA = {} #postactivation
        #self.PrA[0] = X
        self.PsA[0] = X
        for i in range(self.nh+1):
            self.PrA[i+1] = np.matmul(self.PsA[i], self.W[i+1]) + self.B[i+1] # (M,N) * (N,U) => (M, U)
            if i < self.nh:
                self.PsA[i+1] = self.leakyrelu(self.PrA[i+1]) #(M,U)
            elif i == self.nh:
                self.PsA[i+1] = self.PrA[i+1] # (M,1)
        return self.PsA[self.nh+1]

    def backward(self, X, Y, lamda):
        self(X) #forward pass
        self.dW = {}
        self.dB = {}
        self.dPsA = {}
        self.dPrA = {}
        L = self.nh + 1                                                               # output layer number
        self.dPrA[L] = 2*(self.PsA[L] - Y)                                            # (M,1) - (M,1) -> (M,1)
        # print(self.PsA[L].shape,Y.shape, self.dPrA[L].shape)
        for k in range(L, 0, -1):
            self.dW[k] = np.matmul(self.PsA[k-1].T, self.dPrA[k])                       # (U,M)*(M,1) => (U,1) for last layer (U,M)*(M,U) = >(U,U) for hidden
            self.dB[k] = np.sum(self.dPrA[k],axis=0).reshape(1,-1)                      # (1,1) or (1,U)
            # print(self.W[k].T.shape)
            if (k > 1):
                self.dPsA[k-1] = np.matmul(self.dPrA[k], self.W[k].T)                       # (M,1) * (1,U)=>(M,U) for output otherwise (M,U)*(U,U)=>(M,U)
                self.dPrA[k-1] = np.multiply(self.dPsA[k-1], self.grad_leakyrelu(self.PsA[k-1])) # (M,1) * (1,U)=>(M,U) for output otherwise (M,U)*(U,U)=>(M,U)

    def predict(self, X):
        Y_pred = self(X)
        return np.array(Y_pred).squeeze()


class Optimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, weights, biases, delta_weights, delta_biases):
        # m = X.shape[1]
        nh = len(weights)
        for i in range(nh):
            weights[i+1] -= self.learning_rate * delta_weights[i+1]
            biases[i+1] -= self.learning_rate * delta_biases[i+1]


def loss_mse(y, y_hat):
    '''
    Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
        MSE loss between y and y_hat.
    '''
    mse_loss = np.square(np.subtract(y,y_hat)).mean()
    return mse_loss

def loss_regularization(weights, biases):
    '''
    Compute l2 regularization loss.

    Parameters
    ----------
        weights and biases of the network.

    Returns
    ----------
        l2 regularization loss 
    '''
    total_norm=0
    nh = len(weights)
    for i in range(nh):
        total_norm += np.square(weights[i+1]).sum()
        total_norm += np.square(biases[i+1]).sum()
    return (total_norm)

def loss_fn(y, y_hat, weights, biases, lamda):
    '''
    Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1
        weights and biases of the network
        lamda: Regularization parameter

    Returns
    ----------
        l2 regularization loss 
    '''
    loss = loss_mse(y,y_hat) + (lamda * loss_regularization(weights,biases))
    return loss

def rmse(y, y_hat):
    '''
    Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
        RMSE between y and y_hat.
    '''
    mse_loss = np.square(np.subtract(y,y_hat)).mean()
    rmse_loss = np.sqrt(mse_loss)
    return rmse_loss


def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target,plot=False
):
    '''
    In this function, you will perform following steps:
        1. Run gradient descent algorithm for `max_epochs` epochs.
        2. For each bach of the training data
            1.1 Compute gradients
            1.2 Update weights and biases using step() of optimizer.
        3. Compute RMSE on dev data after running `max_epochs` epochs.
    '''
    if plot:
        iter = []
        loss = []
        dev_loss = []
        for p in range(max_epochs):
            loss.append(0)
            dev_loss.append(0)
            iter.append(p)

    nh = net.nh
    m = train_input.shape[0]
    for j in range(max_epochs):
        for k in range(0,m,batch_size):
            net.backward(train_input[k:k+batch_size],train_target[k:k+batch_size],0)
            dW = {}
            dB = {}
            for i in range(nh+1):
                dW[i+1] = (net.dW[i+1] / batch_size) + ((2*lamda)*(net.W[i+1])) 
                dB[i+1] = (net.dB[i+1] / batch_size) #+ ((2*lamda)*(net.B[i+1])) 
            optimizer.step(net.W,net.B,dW,dB)
        if plot:
            dev_pred = net.predict(dev_input)
            #train_pred = net.predict(train_input)
            dev_loss[j] = loss_fn(dev_target,dev_pred,net.W,net.B,lamda)
            #loss[j] = loss_fn(train_target,train_pred,net.W,net.B,lamda)
    dev_predict = net.predict(dev_input)
    rmse_score =  rmse(dev_target,dev_predict)
    print("Dev - rmse score:", rmse_score)
    train_loss=0
    for k in range(0,m,batch_size):
        train_batch_pred = net.predict(train_input[k:k+batch_size])
        train_loss += np.sum(np.square(np.subtract(train_target[k:k+batch_size],train_batch_pred)))

    rmse_train = np.sqrt(train_loss/m)
    print("Train - rmse score:", rmse_train)
    if plot:
        #plt.plot(iter,loss)
        plt.plot(iter,dev_loss)
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.show()
        return (dev_loss,dev_loss)

def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	return net.predict(inputs)

def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''
    X = pd.read_csv(csv_path)
    if 'label' in X.columns:
        X.drop('label',axis=1,inplace=True)
    X_arr = np.asarray(X,dtype=np.float64)
    # feature scaling later
    if scaler is not None:
        if is_train:
            scaler.fit(X_arr)
        X_norm = scaler.transform(X_arr)
        return X_norm
    #If scaling is disabled
    return X_arr

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    X = pd.read_csv(csv_path)
    Y = X.iloc[:,0]
    Y_arr = np.asarray(Y,dtype=np.float64)
    return Y_arr.reshape(-1,1)


def read_data():
    '''
    Read the train, dev, and test datasets
    '''
    input_dir = './dataset/'
    train_input, train_target = get_features(input_dir + 'train.csv',True,None), get_targets(input_dir + 'train.csv')
    dev_input, dev_target = get_features(input_dir+'dev.csv',False,None), get_targets(input_dir + 'dev.csv')
    test_input = get_features(input_dir + 'test.csv',False,None)
    print(train_input.shape)
    return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 128


	learning_rate = 0.0001
	num_layers = 1
	num_units = 64
	lamda = 0.01 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	#get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()
