"""new oos and preprocessing. """
import os
import re
import pickle
import string
import unicodedata
from random import randint

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from nltk.corpus import stopwords
from wordcloud import STOPWORDS, WordCloud
from sklearn.naive_bayes import MultinomialNB,GaussianNB,CategoricalNB

from sklearn.feature_extraction.text import TfidfVectorizer


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
print("data set done ........")
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
# print(y_train)
print("hello")
x_test = Vectorizer.transform(x_test).toarray()
print(len(x_test))
y_test = transform(y_test)
print(len(y_test))
model = LinearSVC()
model.fit(x_train,y_train)
print("model_trained")
from sklearn.model_selection import learning_curve, ShuffleSplit
# with open('text_emotions','wb') as f:
#     pickle.dump(model,f)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
emotions = ['love','anger','surprise','joy','sadness','fear']
# Define the emotions and their numerical labels
# emotions = ['joy', 'love', 'surprise', 'anger', 'sadness', 'fear']

# Generate a confusion matrix with labeled axes
cm = confusion_matrix(y_test, y_pred, labels=range(len(emotions)))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=emotions, yticklabels=emotions)
plt.show()
from sklearn.metrics import f1_score, plot_precision_recall_curve
#
# # assuming you have your predictions and true labels stored in y_pred and y_true variables
# f1 = f1_score(y_test, y_pred, average='micro')  # calculate F1 score

f1_scores = f1_score(y_test, y_pred, average=None)

# plot the f1 scores
plt.bar(range(len(f1_scores)), f1_scores)
plt.xticks(range(len(f1_scores)), emotions)
plt.xlabel('Class')
plt.ylabel('F1 Score')
plt.title('F1 Score per Class')
plt.show()