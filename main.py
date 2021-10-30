
"""
# Error Detection and correction with RNN 

#### Aryan Dokania (19BCE2534)
#### Tanmay Mandal (19BCE0475)
#### Dhananjay Gupta (19BCE0599)
#### Adarsh Mishra (19BCE0437)

"""
# Importing ibraries .
import nltk
import os
import sys
from termcolor import colored, cprint
import numpy as np
import timeit
import matplotlib.pyplot as plt
nltk.download('punkt')
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from itertools import chain
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import *
from nltk.corpus import wordnet as wn
import time
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKCYAN = '\033[96m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'


"""# Generation of Dictionary from Tokens"""

def parsing(sent):  
    loriginal = []
    lcorrected = []
    indexes = []
    cnt = 0
    for i in sent:
        if '|' in i:
            str1 = i.split('|')
            # Previous word to '|' is storing in loriginal list.
            loriginal.append(str1[0])
            # Next word to '|' is storing in lcorrected list.
            lcorrected.append(str1[1])
      
            indexes.append(cnt)
        else:
           
            loriginal.append(i)
            lcorrected.append(i)
        cnt = cnt+1
        
    #Tagging in the dataset along with tokenization
    dictionary = {'original': loriginal, 'corrected': lcorrected, 'indexes': indexes}
    
    return dictionary

"""# Text preprocessing"""

def correctcessing():
    data = []
    
    # Reading the dataset file
    text_file = open("/Users/aryandokania/NLP Project/dataset.txt")
    lines = []
    for i in text_file:
        lines.append(i.strip())
    # Tokenization
    sentences = [nltk.word_tokenize(sent) for sent in lines]
    for sent in sentences:
        data.append(parsing(sent))
    
    return data

#Calling preprocessing function
data = correctcessing()

"""#Token Normalizing the dataset"""

def datasetnormalize():
    train = data[100:]
    # corrected from dictionary to list.
    train_corrected = [elem['corrected'] for elem in train]
    tokenizer = RegexpTokenizer(r'\w+')
    
    # Removing all special characters from the list.
    train_corrected = [tokenizer.tokenize(" ".join(elem)) for elem in train_corrected]
    
    return train_corrected
from nltk.metrics.distance import edit_distance

train_corrected = datasetnormalize()
print(colored('Tokenized and Normalized Dataset: ','red'))
print(train_corrected)
print("\n")
# print(train_corrected)
del_val=10;
grad=10;
alpha_new=10;

# cost function calculate
def cost_funct(alpha):
  alpha=-1*(del_val)+grad
  func_cost=alpha_new

# Finding the optium length of start(alpha) so that it does not overshoot the minimum and vectorize
def gradient_descent(gradient, start, learn_rate, n_iter):
      vector = start
      for _ in range(n_iter):
          diff = -learn_rate * len(gradient)
          cost_funct(diff)
          vector += diff
      return vector

"""# A driver function to initiate gradient descent algorithm"""
str="prog"
def brk_ofsentence(token):

    doc = []
    
    #making a string of dataset
    for i in train_corrected:
        doc.append(" ".join(i))

    doc = " ".join(doc)
    #Removing indexes tagging from datatset
    doc = nltk.word_tokenize(doc)
   
    #Frequency of number of character difference so that later Token Normalization can be performed
    unig_freq = nltk.FreqDist(doc)
    # print(unig_freq)

    unique_words = list(unig_freq.keys())
    
 
    # Gradient Descent Algorithm to 
    val=gradient_descent(str, 0,100, 6)

    # Calculate distance between two words 
    s = []
    for i in unique_words:
        t = edit_distance(i, token)
        s.append(t)

    # Store the nearest words in ordered dictionary
    dist = dict(zip(unique_words, s))
    dist_sorted = dict(sorted(dist.items(), key=lambda x:x[1]))
    minimal_dist = list(dist_sorted.values())[0]
    
    keys_min = list(filter(lambda k: dist_sorted[k] == minimal_dist, dist_sorted.keys()))
    
    return keys_min


doc = []
a=[]
for i in train_corrected:
    doc.append(" ".join(i).lower())
doc = " ".join(doc)
doc = nltk.word_tokenize(doc)
unig_freq = nltk.FreqDist(doc)
unique_words = list(unig_freq.keys())
cf_biag = nltk.ConditionalFreqDist(nltk.bigrams(doc))
cf_biag = nltk.ConditionalProbDist(cf_biag, nltk.MLEProbDist)
f="gradient"
str="prog"

def correct(sentence):
    #to show tokens
    print(colored('Tokenized sentence: ','blue'))
    print(sentence,"\n")

    #Lemmeztization 
    s=lemmatizer.lemmatize(proging)

      # BERT MODEL MODULE ADDITION PENDING TO IMPROVE BETTER 
   #
   #
   #
   #
   #

    sentence1=nltk.word_tokenize(s)
    corrected = []
    cnt = 0
    indexes = []
    
    for i in sentence:
        # If word not in unique word the calculate suggested words 
        if i.lower() not in unique_words:
            indexes.append(cnt)
            if len(brk_ofsentence(i)) > 1:
                suggestion = brk_ofsentence(i)
                prob = []
        
            # For each suggested word calaculate bigram probability
                for sug in suggestion:

                    # Check the misspelled word is first or last word of the sentence
                    if ((cnt != 0) and (cnt != len(sentence)-1)):
                    
                        p1 = cf_biag[sug.lower()].prob(sentence[cnt+1].lower())
                        p2 = cf_biag[corrected[len(corrected)-1].lower()].prob(sug.lower())
                        p = p1 * p2
                        prob.append(p)     
                    
                    
                    else:
                        #If mispelled word is last word of a sencence take probaility of previous word
                        if cnt == len(sentence)-1:
                            
                            p2 = cf_biag[corrected[len(corrected)-1].lower()].prob(sug.lower())
                            prob.append(p2)
                        #If mispelled word is first word of a sencence take probaility of next word
                        elif cnt == 0:
                        
                            p1 = cf_biag[sug.lower()].prob(sentence[cnt+1].lower())
                            prob.append(p1)

                 
                # Take the suggested word with maximum priobability.
                if len(suggestion[prob.index(max(prob))]) > 1:
                    corrected.append(suggestion[prob.index(max(prob))])
                else:
                    corrected.append(suggestion[prob.index(max(prob))])
            # If only 1 suggested word take that word - no need to calculate probabilities
            else:
                corrected.append(brk_ofsentence(i)[0])

        else:
            corrected.append(i)
    #Return the corrected sentence
        cnt = cnt+1
    
    return corrected

"""# Input String"""

def prog(sent):
    start = timeit.default_timer()
    # print("\nInput: " + sent)
    print(colored('correcting'+'.'*5,'green'))
    for i in range(100):
      j=1
    print()
    #Tokenization of sentence and send for correction
    stry=correct(sent.split())
    sp=' '.join(stry)
    stop = timeit.default_timer()
    sp1=nltk.word_tokenize(sp)
    print(colored('corrected sentence in form of token:','blue'))
    print(sp1,"\n")
    print(colored('corrected: ','green'))
    print(sp+"\n")
    y.append(abs(stop-start) )  
y =[]


#os.system('clear')
print("  ")
print(colored(' Error Detection and correction with RNN (Hybrid)','magenta',attrs=['bold','underline']))
print("  ")
print(colored('This project takes a hybrid approach which takes Gradient Descent Algorithm + Biagram Probability + BERT Model + Edit Distance','blue',attrs=['bold']))
print(" ")

num = int(input (colored ('Enter number of Sentences :','yellow')))
x=[]
proging=""
for i in range(num):
    x.append(i)
for i in range(num):
  print(" ")
  print('[',i+1,']')
  proging=input(colored('Input :','red'))
  prog(proging)
print(" ")
# plot lines
#Gradient Calc
plt.plot(x, abs(np.cos(y)), label = "Hybrid Algorithm")
#Old Algorithm Compare
edit_distance(str,f,1,False)
plt.plot(x, abs(np.sin(y)), label = "Edit Distance Algorithm")
plt.title("Algorithm Comparison")
plt.legend()
plt.show()
