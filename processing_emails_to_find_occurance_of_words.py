import os
import string
import numpy as np
import pandas as pd
from time import time
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
os.chdir(os.path.dirname(__file__))

start_time = time()

df = pd.read_csv('wordslist.csv',header=0)
words = df['word']

lmtzr = WordNetLemmatizer()

directory_in_str = "emails/"
directory = os.fsencode(directory_in_str)

f = open("frequency.csv","w+")
for i in words:
    f.write(str(i) + ',')
f.write('output')
f.write('\n')
f.close()

k=0

for file in os.listdir(directory):
    file_name = file.decode("utf-8")
    path_file = str(os.getcwd()) + '/emails/'
    path_file = path_file + file_name

    k += 1
    file_reading = open(path_file,"r",encoding='utf-8', errors='ignore')
    words_list_array = np.zeros(words.size)
    for word in file_reading.read().split():
        word = lmtzr.lemmatize(word.lower())
        if(word in stopwords.words('english') or word in string.punctuation or word.isdigit()==True): #or len(word)<=2 
            continue
        for i in range(words.size):
            if(words[i]==word):
                words_list_array[i] = words_list_array[i]+1
                break

    f = open("frequency.csv","a") # append
    for i in range(words.size):
        f.write(str(int(words_list_array[i])) + ',')
    ######################################## Note: research######################################
    if(len(file_name)==27):
        f.write("-1")
    elif (len(file_name)==30):
        f.write("1")
    else:
        pass
    
    f.write('\n')
    f.close()
    if(k % 100 == 0):
        print("Done " + str(k))

print("Time (in seconds) to segregate entire dataset to form input vector " + str(round(time() - start_time,2)))