import pickle
import pandas as pd
import numpy as np

import os
import re
import pickle
import string
import unicodedata
from random import randint
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.ensemble
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from wordcloud import STOPWORDS, WordCloud
from sklearn.naive_bayes import MultinomialNB,GaussianNB,CategoricalNB

# from main import Vectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import _gradient_boosting ,AdaBoostClassifier
# _gradient_boosting.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def transform(y_train):

    y_train_vals = []
    for i in y_train:
        if i == 'love':
            y_train_vals.append(0)
        elif i == 'anger':
            y_train_vals.append(1)
        elif i == 'surprise':
            y_train_vals.append(2)
        elif i == 'joy':
            y_train_vals.append(3)
        elif i == 'sadness':
            y_train_vals.append(4)
        elif i == 'fear':
            y_train_vals.append(5)
    return y_train_vals


def transform_decode(val):
    y_vals = []
    for i in val:
        if i == 0:
            y_vals.append("love🥰")
        elif i == 1:
            y_vals.append("anger😡")
        elif i == 2:
            y_vals.append('surprise 😮')
        elif i == 3:
            y_vals.append('joy 😃')
        elif i ==4:
            y_vals.append( 'sadness 😔')
        elif i == 5:
            y_vals.append("fear 😨")
    return y_vals

data_train = pd.read_csv("train.txt",sep=";")
# print(data_train)
data_train = data_train.rename(columns={'i didnt feel humiliated':'text','sadness':'emotions'})
# data_train.text = data_train['text']
data_train.text = data_train.text.apply(str.lower)
data_train.emotions = data_train.emotions.apply(str.lower)
x_train = data_train['text']
y_train = data_train['emotions']

data_test = pd.read_csv("test.txt",sep=";")
data_test = data_test.rename(columns={'im feeling rather rotten so im not very ambitious right now':'text','sadness':'emotions'})
# print(data_test)
data_test.text = data_test.text.apply(str.lower)
data_test.emotions = data_test.emotions.apply(str.lower)

x_test = data_test['text']
y_test = data_test['emotions']
from contractions import contractions_dict


def expand_contractions(text, contraction_map=contractions_dict):
    # Using regex for getting all contracted words
    contractions_keys = '|'.join(contraction_map.keys())
    contractions_pattern = re.compile(f'({contractions_keys})', flags=re.DOTALL)

    def expand_match(contraction):
        # Getting entire matched sub-string
        match = contraction.group(0)
        expanded_contraction = contraction_map.get(match)
        if not expand_contractions:
            print(match)
            return match
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)

    return expanded_text


# expand_contractions("i'd think y'all can do this.")

x_train.apply(expand_contractions)
y_train.apply(expand_contractions)

from cleantext import clean


def rm_punc_from_word(word):
    clean_alphabet_list = [
        alphabet for alphabet in word if alphabet not in string.punctuation
    ]
    return ''.join(clean_alphabet_list)


print(rm_punc_from_word('#cool!'))


# Remove puncuation from text
def rm_punc_from_text(text):
    clean_word_list = [rm_punc_from_word(word) for word in text]
    return ''.join(clean_word_list)



import nltk

# nltk.download("stopwords")

def rm_number_from_text(text):
    text = re.sub('[0-9]+', '', text)
    return ' '.join(text.split())  # to rm `extra` white space

def rm_stopwords_from_text(text):
    _stopwords = stopwords.words('english')
    text = text.split()
    word_list = [word for word in text if word not in _stopwords]
    return ' '.join(word_list)

def clean_text(text):
    text = text.lower()
    text = rm_punc_from_text(text)
    text = rm_number_from_text(text)
    text = rm_stopwords_from_text(text)

    # there are hyphen(–) in many titles, so replacing it with empty str
    # this hyphen(–) is different from normal hyphen(-)
    text = re.sub('–', '', text)
    text = ' '.join(text.split())  # removing `extra` white spaces

    # Removing unnecessary characters from text
    text = re.sub("(\\t)", ' ', str(text)).lower()
    text = re.sub("(\\r)", ' ', str(text)).lower()
    text = re.sub("(\\n)", ' ', str(text)).lower()
    # remove accented chars ('Sómě Áccěntěd těxt' => 'Some Accented text')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode(
        'utf-8', 'ignore'
    )

    text = re.sub("(__+)", ' ', str(text)).lower()
    text = re.sub("(--+)", ' ', str(text)).lower()
    text = re.sub("(~~+)", ' ', str(text)).lower()
    text = re.sub("(\+\++)", ' ', str(text)).lower()
    text = re.sub("(\.\.+)", ' ', str(text)).lower()

    text = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(text)).lower()

    text = re.sub("(mailto:)", ' ', str(text)).lower()
    text = re.sub(r"(\\x9\d)", ' ', str(text)).lower()
    text = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(text)).lower()
    text = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM',
                  str(text)).lower()
    text = re.sub("(\.\s+)", ' ', str(text)).lower()
    text = re.sub("(\-\s+)", ' ', str(text)).lower()
    text = re.sub("(\:\s+)", ' ', str(text)).lower()
    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()

    try:
        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(text))
        repl_url = url.group(3)
        text = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(text))
    except Exception as e:
        pass

    text = re.sub("(\s+)", ' ', str(text)).lower()
    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()

    return text


clean_text("Mrs. Robinson, 343 -- you're trying to fool &^%me, aren't you?")
x_train = x_train.apply(clean_text)
x_test = x_test.apply(clean_text)
y_train = y_train.apply(clean_text)
y_test = y_test.apply(clean_text)
Vectorizer = TfidfVectorizer()
x_train = Vectorizer.fit_transform(x_train).toarray()
y_train = transform(y_train)
print("data set done ........")


Vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, stop_words='english')
print("vectorization complete.......")
x_train = Vectorizer.fit_transform(x_train).toarray()
y_train = transform(y_train)
# print(y_train)
x_test = Vectorizer.transform(x_test).toarray()
# x_points = [sum(i) for i in x_test]
y_test = transform(y_test)
# y_points = [j for j in y_test]
# data = zip(x_points,y_points)
# sns.catplot(data)
# plt.scatter(x_points,y_points,s = 6)
# plt.show()
# plt.close()
# pickle.dump(Vectorizer,open("vectoriser.pkl","wb"))
# print(x_test)
# model = AdaBoostClassifier(sklearn.ensemble.RandomForestClassifier(),algorithm='SAMME')
model = LinearSVC(random_state=0)
# model = SVC(kernel="rbf",random_state=0)
print("started training....")
model.fit(x_train,y_train)
print("Training complete......")
y_pred = model.predict(x_test)
# from sklearn.model_selection import GridSearchCV
# #
# # # defining parameter range
# # param_grid = {'C': [0.1, 1, 10, 100, 1000],
# #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
# #               'kernel': ['rbf']}
# #
# # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
# #
# # # fitting the model for grid search
# # grid.fit(x_train, y_train)
# # print(grid.best_params_)
# #
# # # print how our model looks after hyper-parameter tuning
# # print(grid.best_estimator_)
# # grid_predictions = grid.predict(x_test)
# #
# # # print classification report
# # print(classification_report(y_test, grid_predictions))
# # print(x_test)
# #
# accuracy = accuracy_score(y_test,y_pred)
# # # accuracy = accuracy_score(y_train,y_pred)
# #
# print(f'{round(accuracy*100,2)}%')
# a = input("enter the text : ")
# a_p = Vectorizer.transform([a])
# val = model.predict(a_p)
# val = transform_decode(val)
# print(*val)
# error = confusion_matrix(y_test,y_pred)
# print(error)
# f1 = f1_score(y_test,y_pred,average="macro")
# print(f1)
# # pickle.dump(model,open("text_emotion.pkl",'wb'))

import paho.mqtt.client as mqtt
import time
#
# # Set up MQTT client
# client = mqtt.Client()
# client.connect("mqtt.eclipse.org", 1883, 60)
#
# # Send data to Android app
# def send_data(data):
#     client.publish("my_topic", data)
#
# # Send a sequence of numbers to the Android app
# for i in range(10):
#     send_data(str(i))
#     time.sleep(1)
#
# # Disconnect MQTT client
# client.disconnect()
import pickle
import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)


def print_items(items: Iterable) -> None:
    logger.debug("Printing each item of the iterable")
    try:
        iter(items)
    except TypeError as error:
        error_msg = f'items should be of type iter but {type(items)} were given.'
        logger.error(error_msg)
        raise TypeError(error_msg).with_traceback(error.__traceback__)
    else:
        if items:
            for item in items:
                print(item)
        else:
            error_msg = f'empty iterable was given.'
            logger.error(error_msg)
            raise ValueError(error_msg)


print_items([1, 2, 3])
print_items((1, 2, 3))
print_items({1, 2, 3})
print_items({1: 2, 2: 4, 3: 4})
print_items(4)
