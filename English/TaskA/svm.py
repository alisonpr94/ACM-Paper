# -*- coding: utf-8 -*-

#@author: alison

import re
import string
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split

# Etapa de pré-processamento

def clean_tweets(tweet):
    tweet = re.sub('@(\\w{1,15})\b', '', tweet)
    tweet = tweet.replace("via ", "")
    tweet = tweet.replace("RT ", "")
    tweet = tweet.lower()
    return tweet
    
def clean_url(tweet):
    tweet = re.sub('http\\S+', '', tweet, flags=re.MULTILINE)   
    return tweet
    
def remove_stop_words(tweet):
    stops = set(stopwords.words("english"))
    stops.update(['.',',','"',"'",'?',':',';','(',')','[',']','{','}'])
    toks = [tok for tok in tweet if not tok in stops and len(tok) >= 3]
    return toks
    
def stemming_tweets(tweet):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in tweet]
    return stemmed_words

def remove_number(tweet):
    newTweet = re.sub('\\d+', '', tweet)
    return newTweet

def remove_hashtags(tweet):
    result = ''

    for word in tweet.split():
        if word.startswith('#') or word.startswith('@'):
            result += word[1:]
            result += ' '
        else:
            result += word
            result += ' '

    return result

def preprocessing(tweet, swords = True, url = True, stemming = True, ctweets = True, number = True, hashtag = True):

    if ctweets:
        tweet = clean_tweets(tweet)

    if url:
        tweet = clean_url(tweet)

    if hashtag:
        tweet = remove_hashtags(tweet)
    
    twtk = TweetTokenizer(strip_handles=True, reduce_len=True)

    if number:
        tweet = remove_number(tweet)
    
    tokens = [w.lower() for w in twtk.tokenize(tweet) if w != "" and w is not None]

    if swords:
        tokens = remove_stop_words(tokens)

    if stemming:
        tokens = stemming_tweets(tokens)

    text = " ".join(tokens)

    return text

def bag_of_words(train, test):
    vec = CountVectorizer(analyzer='word', binary=True, min_df=1, ngram_range=(1, 3), max_features=25000)
    train = vec.fit_transform(train).toarray()
    test = vec.transform(test).toarray()
    return train, test

def save_files(y_test, y_pred, ID):
    with open("ClassPred/SVM_BoW.tsv", "w") as file:
        for i in range(len(y_pred)):
            file.write(str(ID[i]))
            file.write('\t')
            file.write(str(y_pred[i]))
            file.write('\n')

    with open("input/res/en_a.tsv", "w") as file:
        for i in range(len(y_pred)):
            file.write(str(ID[i]))
            file.write('\t')
            file.write(str(y_pred[i]))
            file.write('\n')

def main():

	train = pd.read_csv('Dataset/train_en.tsv', delimiter='\t',encoding='utf-8')
	dev = pd.read_csv('Dataset/dev_en.tsv', delimiter='\t',encoding='utf-8')
	test = pd.read_csv('Dataset/test_en.tsv', delimiter='\t',encoding='utf-8')

	###########################################################################################################
    
	# Pré-processamento dos tweets

	train_text = train['text'].map(lambda x: preprocessing(x, swords = True, url = True, stemming = True, ctweets = True, number = True, hashtag = True))
	y_train    = train['HS']
	id_train   = train['id']
	
	dev_text   = dev['text'].map(lambda x: preprocessing(x, swords = True, url = True, stemming = True, ctweets = True, number = True, hashtag = True))
	y_dev      = dev['HS']
	id_dev     = dev['id']

	test_text  = test['text'].map(lambda x: preprocessing(x, swords = True, url = True, stemming = True, ctweets = True, number = True, hashtag = True))
	y_test     = test['HS']
	id_test    = test['id']

	###########################################################################################################

	# Bag-of-Words
	teste  = test_text[1000:]
	labels = y_test[1000:]

	train_text  = np.concatenate((train_text, dev_text, teste), axis=0)
	y_train     = np.concatenate((y_train, y_dev, labels), axis=0)

	x_train, x_test = bag_of_words(train_text, test_text)

	###########################################################################################################

	# Fase de classificação de sentimentos

	clf = LinearSVC(C=10, verbose=1, max_iter=2000, random_state=None, penalty='l2')

	clf.fit(x_train, y_train)   # Fase de treinamento

	# Criando arquivo para salvar modelo treinado
	filename = 'Models/modelSVM.sav'
	pickle.dump(clf, open(filename, 'wb'))

	y_pred = clf.predict(x_test)    # Fase de predição, testando dados novos

	# Salvando arquivos para a avalição em evaluation.py
	#save_files(y_test, y_pred, id_test)

	print("Treinamento finalizado! Testando modelo...")

	#print("F1.........: %f" %(f1_score(y_test, y_pred, average="macro")))
	#print("Precision..: %f" %(precision_score(y_test, y_pred, average="macro")))
	#print("Recall.....: %f" %(recall_score(y_test, y_pred, average="macro")))
	#print("Accuracy...: %f" %(accuracy_score(y_test, y_pred)))
 
if __name__ == '__main__':
    main()