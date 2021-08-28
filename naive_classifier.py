import os
import random
import string
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from time import time

class Naive_bayes():
    def __init__(self, data):
        self.data = data
        self.TP =0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.correct = 0
        self.wrong = 0
        self.y_predict = []

    def unique(self):
        random.shuffle(self.data)
        train_data = self.data[0 : int(len(self.data)/2)]
        test_data = self.data[int(len(self.data)/2) + 1 : -1]
        ham_dict = dict()
        spam_dict = dict()
        for d in train_data:
            if d[-1] == "ham":
                for word in d[0].split():
                    if word in ham_dict:
                        ham_dict[word] += 1
                    else:
                        ham_dict[word] = 1
            elif d[-1] == "spam":
                for word in d[0].split():
                    if word in spam_dict:
                        spam_dict[word] += 1
                    else:
                        spam_dict[word] = 1
        return test_data, ham_dict, spam_dict

    def predict(self, test_data, ham_dict, spam_dict):
        prior_ham = 0.5
        prior_spam = 0.5
        for d in test_data:
            text = d[0]
            p_ham = 1
            p_spam = 1
            for word in text.split():
                num_ham = ham_dict[word] if word in ham_dict else 0.000001
                num_spam = spam_dict[word] if word in spam_dict else 0.000001
                likily_ham = num_ham / (num_ham + num_spam)
                likily_spam = num_spam / (num_ham + num_spam)
                p_ham *= (likily_ham * prior_ham) / (likily_ham * prior_ham + likily_spam * prior_spam)
                p_spam *= (likily_spam * prior_spam) / (likily_ham * prior_ham + likily_spam * prior_spam)
            if p_spam > p_ham and d[-1] == "spam":
                self.TN += 1
                self.correct += 1
                self.y_predict.append(p_spam)
            elif p_spam < p_ham and d[-1] == "ham":
                self.TP += 1
                self.correct += 1
                self.y_predict.append(p_ham)
            else:
                if d[-1] == "spam":
                    self.FP += 1
                    self.y_predict.append(p_ham)
                if d[-1] == "ham":
                    self.FN += 12
                    self.y_predict.append(p_spam)
                self.wrong += 1
                
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
        data.append([text_without_stopwords, "ham"])
    for file_path in files_spam:
        f = open(dir_spam + file_path, "r")
        text = f.read()
        text_without_punctuation = [c for c in text if c not in string.punctuation]
        text_without_punctuation = ''.join(text_without_punctuation)
        text_without_stopwords = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]
        text_without_stopwords = ' '.join(text_without_stopwords)
        data.append([text_without_stopwords, "spam"])
    return data

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
        if data[-1] == 'ham':
            y_test.append(1)
        elif data[-1] == 'spam':
            y_test.append(-1)
    t_precision, t_recall, _  = precision_recall_curve(y_test, y_predict)
    t_auc = auc(t_recall, t_precision)
    return t_precision, t_recall, t_auc

def plot(t_precision, t_recall, ap):
    plt.figure()
    plt.title('Average Precision')
    plt.plot(t_recall, t_precision, 'r', marker='.', label='AP %.4f' % ap)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # plt.show()
    plt.savefig('Naive_dataset_'+ str(ap) +'.png')

if __name__ == '__main__':
    start_time = time()
    os.chdir(os.path.dirname(__file__))
    dir_path = [["dataset/ham1/", "dataset/spam1/"], ["dataset/ham2/", "dataset/spam2/"], ["dataset/ham3/", "dataset/spam3/"]]
    for dir in dir_path:
        data = read_data(dir[0], dir[1])
        Naive = Naive_bayes(data)
        test_data, ham_dict, spam_dict = Naive.unique()
        Naive.predict(test_data, ham_dict, spam_dict)
        accuracy = Naive.correct / (Naive.correct + Naive.wrong)
        precision = Naive.TP/(Naive.TP + Naive.FP)
        recall = Naive.TP/(Naive.TP + Naive.FN)
        F1_score = 2*(precision*recall)/(precision + recall)
        result = [Naive.TP, Naive.TN, Naive.FP, Naive.FN]
        pre, rec, ap = rating_parameter(test_data, Naive.y_predict)
        write_to_file('Naive Bayes Classifier', result, precision, recall, ap, F1_score, start_time)
        print('accuracy %.4f' % accuracy)
        print('precision %.4f' % precision)
        print('recall %.4f' % recall)
        plot(pre, rec, ap)
