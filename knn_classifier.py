import os
import random
import string
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from time import time

class KNNModel():
    def __init__(self):
        self.TP =0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.correct = 0
        self.wrong = 0
        self.y_predict = []

    def getSimilarity(self, record1, record2):
        len1 = len(record1[0].split())
        len2 = len(record2[0].split())
        num_common = 0
        d = dict()
        for word in record1[0].split():
            if word not in d:
                d[word] = 1
        for word in record2[0].split():
            if word in d:
                num_common += 1
        similarity = num_common / (len1 * len2) ** 0.5
        return similarity

    def findKNN(self, train_data, record, k):
        for i in range(0,len(train_data)):
            sim = self.getSimilarity(train_data[i], record)
            train_data[i][-1] = sim
        # sort the train_data by similarity
        # from operator import itemgetter
        # train_data.sort(key = itemgetter(-1))
        # return the k nearest neighbor
        res = []
        for i in range(k):
            max_sim = 0
            max_sim_index = 0
            for i in range(0, len(train_data)):
                if train_data[i][-1] > max_sim:
                    max_sim = train_data[i][-1]
                    max_sim_index = i
            train_data[max_sim_index][-1] = 0
            res.append(train_data[max_sim_index])
        return res

    def judge(self, knn):
        num_ham = 0
        num_spam = 0
        for r in knn:
            if r[1] == 'ham':
                num_ham += 1
            else:
                num_spam += 1
        return "ham" if num_ham > num_spam else "spam", num_ham, num_spam

    def predict(self, train_data, test_data):
        k = 1000
        # k = len(test_data)

        for d in test_data:
            knn = self.findKNN(train_data, d, k)  
            state, num_ham, num_spam = self.judge(knn)
            if state == d[1]:
                self.correct += 1
                if d[1] == 'ham':
                    self.TP += 1
                    self.y_predict.append(num_ham/(num_ham + num_spam))
                elif d[1] == 'spam':
                    self.TN += 1
                    self.y_predict.append(num_spam/(num_ham + num_spam))
            else:
                self.wrong += 1
                if d[1] == 'ham':
                    self.FN += 1
                    self.y_predict.append(num_spam/(num_ham + num_spam))
                elif d[1] == 'spam':
                    self.FP += 1
                    self.y_predict.append(num_ham/(num_ham + num_spam))

def read_data(dir_ham, dir_spam):
    files_ham = os.listdir(dir_ham)
    files_spam = os.listdir(dir_spam)
    data = []
    for file_path in files_ham:
        f = open(dir_ham + file_path, "r")
        text = f.read()
        text_without_punctuation = [c for c in text if c not in string.punctuation]
        text_without_punctuation = ''.join(text_without_punctuation)
        text_without_stopwords = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]
        text_without_stopwords = ' '.join(text_without_stopwords)
        data.append([text_without_stopwords, "ham", 0.0])
    for file_path in files_spam:
        f = open(dir_spam + file_path, "r")
        text = f.read()
        text_without_punctuation = [c for c in text if c not in string.punctuation]
        text_without_punctuation = ''.join(text_without_punctuation)
        text_without_stopwords = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]
        text_without_stopwords = ' '.join(text_without_stopwords)
        data.append([text_without_stopwords, "spam", 0.0])
    return data

def separate(data):
    random.shuffle(data)
    train_data = data[0 : int(len(data)/2)]
    # test_data = data[int(len(data)/2) + 1 : -1]
    test_data = data[-1001:-1]
    return train_data, test_data

def write_to_file(type, result, precision, recall, auc, F1_score, start_time):
    matrix = PrettyTable([' ', 'Ham' , 'Spam'])
    matrix.add_row(['Ham', result[0] , result[2]])
    matrix.add_row(['Spam', result[3] , result[1]])
    f = open("results.txt","a")
    f.write("\n")
    f.write(type + '\n')
    f.write(matrix.get_string())
    f.write("\n")
    f.write("Precision : " + str(round(precision,4)))
    f.write("\n")
    f.write("Recall : " + str(round(recall,4)))
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
    
def rating_parameter(test_data, y_predict):
    y_test = []
    for data in test_data:
        if data[1] == 'ham':
            y_test.append(1)
        elif data[1] == 'spam':
            y_test.append(-1)
    t_precision, t_recall, _  = precision_recall_curve(y_test, y_predict)
    t_auc = auc(t_recall, t_precision)
    return t_precision, t_recall, t_auc

def plot(t_precision, t_recall, ap):
    plt.figure()
    plt.title('Average Precision')
    plt.plot(t_recall, t_precision, marker='.', label='AP %.4f' % ap)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # plt.show()
    plt.savefig('KNN_dataset_'+ str(ap) +'.png')

if __name__ == '__main__':
    start_time = time()
    os.chdir(os.path.dirname(__file__))
    dir_path = [["dataset/ham1/", "dataset/spam1/"], ["dataset/ham2/", "dataset/spam2/"], ["dataset/ham3/", "dataset/spam3/"]]
    for dir in dir_path:
        data = read_data(dir[0], dir[1])
        train_data, test_data = separate(data)
        KNN = KNNModel()
        KNN.predict(train_data, test_data)
        accuracy = KNN.correct / (KNN.correct + KNN.wrong)
        precision = KNN.TP/(KNN.TP + KNN.FP)
        recall = KNN.TP/(KNN.TP + KNN.FN)
        F1_score = 2*(precision*recall)/(precision + recall)
        result = [KNN.TP, KNN.TN, KNN.FP, KNN.FN]
        pre, rec, ap = rating_parameter(test_data, KNN.y_predict)
        write_to_file('KNN Classifier', result, precision, recall, ap, F1_score, start_time)
        print("accuracy",accuracy)
        print("precision",precision)
        print("recall",recall)
        plot(pre, rec, ap)
