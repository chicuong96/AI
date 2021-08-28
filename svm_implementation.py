import os
import numpy as np
import pandas as pd
from time import time
import cvxopt.solvers
import numpy.linalg as la
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
os.chdir(os.path.dirname(__file__))

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class Kernel(object):
    @staticmethod
    def polykernel(dimension, offset, gamma):
        return lambda x, y: ((offset + gamma*np.dot(x.T,y)) ** dimension)
    @staticmethod
    def linear():
        return lambda x,y:np.dot(x.T,y)

    @staticmethod
    def sigmoid(gamma, offset):
        return lambda x, y: np.tanh(gamma*np.dot(x.T,y) + offset)

    @staticmethod
    def radial_basis(gamma):
        return lambda x, y: np.exp(-gamma*la.norm(np.subtract(x, y)))

class SVMTrainer(object):

    def __init__(self, kernel, c):
        self.kernel = kernel
        self.c = c

    def train(self, X, y):
        lagrange_multipliers = self.compute_multipliers(X, y)
        return self.construct_predictor(X, y, lagrange_multipliers)

    def kernel_matrix(self, X, n_samples):
        K = np.zeros((n_samples, n_samples))
        # f = open("K.csv", "w+")
        # f.write('header \n')
        for i, x_i in enumerate(X):
            temp = ""
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
                temp = temp + str(K[i,j]) + ','
            # f.write(temp + '\n')
        # f.close()
        return K

    def construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self.kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)[0]
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            kernel=self.kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def compute_multipliers(self, X, y):
        n_samples, n_features = X.shape
        # K = pd.read_csv('K.csv', header=0, dtype='d').values
        K = self.kernel_matrix(X,n_samples)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples)) #[1, n] = all 0

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.c) #[1, n] = all c

        G = cvxopt.matrix(np.vstack((G_std, G_slack))) #7240x3620
        h = cvxopt.matrix(np.vstack((h_std, h_slack))) #[2, n]

        A = cvxopt.matrix(y, (1, n_samples) , 'd')
        b = cvxopt.matrix(0.0)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])

class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)

    def predict(self, x):
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            # result += w * support_vector_labels * K
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item(), result

def calculate(true_positive,false_positive, false_negative, true_negative):
    result = {}
    result['precision'] = true_positive / (true_positive + false_positive)
    result['recall'] = true_positive / (true_positive + false_negative)
    return result

def confusion_matrix(true_positive,false_positive,false_negative, true_negative, y_pred):
    matrix = PrettyTable([' ', 'Ham' , 'Spam'])
    matrix.add_row(['Ham', true_positive , false_positive])
    matrix.add_row(['Spam', false_negative , true_negative])
    return matrix , y_pred, calculate(true_positive,false_positive,false_negative, true_negative)

def implementSVM(X_train,Y_train,X_test,Y_test,parameters,type, c, y_pred_prob, k):
    ham_spam = 0
    spam_spam = 0
    ham_ham = 0
    spam_ham = 0
    if(type == "polykernel"):
        dimension = parameters['dimension']
        offset = parameters['offset']
        gamma = parameters['gamma']
        trainer = SVMTrainer(Kernel.polykernel(dimension, offset, gamma),float(c))
        predictor = trainer.train(X_train,Y_train)
    elif(type=="linear"):
        trainer = SVMTrainer(Kernel.linear(),float(c))
        predictor = trainer.train(X_train,Y_train)
    elif(type == "sigmoid"):
        gamma = parameters['gamma']
        offset = parameters['offset']
        trainer = SVMTrainer(Kernel.sigmoid(gamma,offset),float(c))
        predictor = trainer.train(X_train,Y_train)
    elif(type == "rbf"):
        gamma = parameters['gamma']
        trainer = SVMTrainer(Kernel.radial_basis(gamma),float(c))
        predictor = trainer.train(X_train,Y_train)

    y_pred_prob[str(type)][k] = []
    for i in range(X_test.shape[0]):
        ans = predictor.predict(X_test[i])[0]
        if(ans==-1 and Y_test[i]==-1):
            spam_spam+=1
        elif(ans==1 and Y_test[i]==-1):
            spam_ham+=1
        elif(ans==1 and Y_test[i]==1):
            ham_ham+=1
        elif(ans==-1 and Y_test[i]==1):
            ham_spam+=1
        y_pred_prob[str(type)][k].append(predictor.predict(X_test[i])[1])
    if(type == "polykernel"):
        y_pred_prob[str(type)][k].append('b')
    elif(type == "linear"):
        y_pred_prob[str(type)][k].append('r')
    elif(type == "sigmoid"):
        y_pred_prob[str(type)][k].append('k')
    elif(type == "rbf"):
        y_pred_prob[str(type)][k].append('m')
    
    return confusion_matrix(ham_ham,ham_spam,spam_ham,spam_spam, y_pred_prob[str(type)][k])

def write_to_file(matrix, result, parameters, type, start_time, auc):
    F1_score = 2*(float(result['precision'])*float(result['recall']))/(float(result['precision']) + float(result['recall']))
    f = open("results.txt","a")
    if(type=="polykernel"):
        f.write("Polykernel model parameters")
        f.write("\n")
        f.write("Dimension : " + str(parameters['dimension']))
        f.write("\n")
        f.write("Offset : " + str(parameters['offset']))
        f.write("\n")
        f.write("Gamma : " + str(parameters['gamma']))
    elif(type=="linear"):
        f.write("Linear model")
    elif(type=="sigmoid"):
        f.write("sigmoid model")
    elif(type=="rbf"):
        f.write("rbf model")
        f.write("\n")
        f.write("Gamma : " + str(parameters['gamma']))
        
    f.write("\n")
    f.write(matrix.get_string())
    f.write("\n")
    f.write("Precision : " + str(round(result['precision'],4)))
    f.write("\n")
    f.write("Recall : " + str(round(result['recall'],4)))
    f.write("\n")
    f.write("AUC : " + str(round(auc,4)))
    f.write("\n")
    f.write("F1 score : " + str(round(F1_score,4)))
    f.write("\n")
    f.write("Time spent for model : " + str(round(time()-start_time,2)))
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.close()

def read_data(wordlist, frequency, percent):
    df1 = pd.read_csv(wordlist)
    df2 = pd.read_csv(frequency, header=0)
    input_output = df2.values
    X = input_output[:,:-1] # ignore final colummn
    Y = input_output[:,-1:] # output
    total = X.shape[0]
    train = int(X.shape[0] * int(percent) / 100)
    X_train = X[:train,:]
    Y_train = Y[:train,:]
    X_test = X[train:,:]
    Y_test = Y[train:,:]
    return X_train, Y_train, X_test, Y_test

def get_parameter_hernel(type):
    if type == 'polykernel':
        dimension = 4
        offset = 7
        gamma = [1, 2, 3]
        c = [0.1]
    elif type == 'linear':
        dimension = 3
        offset = 7
        gamma = [1]
        c = [0.1]
    elif type == 'sigmoid':
        dimension = 3
        offset = 7
        gamma = [1, 2]
        c = [0.1]
    elif type == 'rbf':
        dimension = 3
        offset = 1
        gamma = [1, 2, 3]
        c = [0.1]
    return dimension, offset, gamma, c

def run(X_train, Y_train, X_test, Y_test, type, dimension, offset, gamma, c, y_pred_prob):
    k=0
    t_type = type
    parameters = {}
    for i in range(2,int(dimension)):
        for j in range(0, int(offset), 2):
            for tc in c:
                for tgamma in gamma:
                    parameters['dimension'] = i
                    parameters['offset'] = j
                    parameters['gamma'] = tgamma
                    start_time = time()
                    matrix, y_pred, result = implementSVM(X_train, Y_train, X_test, Y_test, parameters, str(t_type), tc, y_pred_prob, k)
                    t_precision, t_recall, _  = precision_recall_curve(Y_test, y_pred[:-1])
                    t_auc = auc(t_recall, t_precision)
                    write_to_file(matrix, result, parameters, t_type, start_time, t_auc)
                    k +=1
                    print(str(type) + " done : " + str(k))

def model(model, global_start_time = None):
    if model == 'start':
        f = open("results.txt","w+")
        f.close()
    elif model == 'stop':
        f = open("results.txt","a")
        f.write("Time spent for entire code : " + str(round(time()-global_start_time,2)))
        f.close()
    else:
        pass

def plot_AUC(Y_test, y_pred_prob):
    
    for kernel, proba in y_pred_prob.items():
        arrange = {}
        best_Key_auc = 0
        best_value_auc = 0
        if len(proba) != 0:
            for i in range(len(proba)):
                t_precision, t_recall, _  = precision_recall_curve(Y_test, proba[i][:-1])
                t_auc = auc(t_recall, t_precision)
                if len(arrange) == 0:
                    arrange[i] = t_auc
                else:
                    arrange[i] = t_auc
            for key, value in arrange.items():
                if value > best_value_auc:
                    best_value_auc = value
                    best_Key_auc = key
                else:
                    pass
            plt.title('Receiver Operating Characteristic')
            plt.plot(t_recall, t_precision, str(y_pred_prob[kernel][best_Key_auc][-1]), label = 'AP' + kernel + '= %0.4f' % best_value_auc)
        else:
            print('Kernel ' + kernel + ' not found')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.03, 1.03])
    plt.ylim([-0.03, 1.03])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

if __name__ == '__main__':
    global_start_time = time()
    cvxopt.solvers.options['show_progress'] = False
    wordlistfile = 'wordslist.csv'
    frequenceyfile = 'frequency.csv'
    percent = 70
    y_pred_prob = {
                    'polykernel': {},
                    'linear': {},
                    'sigmoid': {},
                    'rbf': {}
                  }
    type = ['polykernel', 'linear', 'rbf']
    model('start')
    X_train, Y_train, X_test, Y_test = read_data(wordlistfile, frequenceyfile, percent)
    for i_type in type:
        dimension, offset, gamma, c = get_parameter_hernel(i_type)
        run(X_train, Y_train, X_test, Y_test, i_type, dimension, offset, gamma, c, y_pred_prob)
    model('stop', global_start_time)
    plot_AUC(Y_test, y_pred_prob)
