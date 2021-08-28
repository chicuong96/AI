import os
from re import X
import nltk
import time
import string
import operator
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# nltk.download()
os.chdir(os.path.dirname(__file__))

def text_cleanup(text):
    text_without_punctuation = [c for c in text if c not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)
    # stopwords high frequency
    text_without_stopwords = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]
    text_without_stopwords = ' '.join(text_without_stopwords)
    cleaned_text = [word.lower() for word in text_without_stopwords.split()]
    return cleaned_text

start_time = time.time()
# Create object 
lmtzr = WordNetLemmatizer()
k=0
count = {}

directory_in_str = "emails/"
# encode path
directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    file = file.decode("utf-8")
    # This method returns current working directory of a process.
    file_name = str(os.getcwd()) + '/emails/'
    file_name = file_name + file
    file_reading = open(file_name,"r",encoding='utf-8', errors='ignore')
    words = text_cleanup(file_reading.read())
    for word in words:
        if (word.isdigit() == False): # and len(word)>2):
            word = lmtzr.lemmatize(word)
            # count quantity of word
            if word in count:
                count[word] += 1
            else:
                count[word] = 1
    k +=1
    if(k % 100 == 0):
        print("Done " + str(k))

sorted_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
sorted_count = dict(sorted_count)

f= open("wordslist.csv","w+")
f.write('word,count')
f.write('\n')
for word , times in sorted_count.items():
    if times < 100:
        break
    f.write(str(word) + ',' + str(times))
    f.write('\n')
f.close()

print('Time (in seconds) to pre process the emails ' + str(round(time.time() - start_time,2)))