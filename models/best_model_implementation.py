#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel

import os
import joblib
from scripts.helpers_train import *

import tensorflow as tf 
import transformers




## Function that will run a linear model and return the predictions

def run_SVM(X_test):
    checker_pipeline = Pipeline([('vectorizer',  TfidfVectorizer().set_params(
            stop_words=None,
            max_features=100000,
            ngram_range=(1, 3))),
                                ('classifier', Pipeline([('feature_selection',
                SelectFromModel(LinearSVC(penalty="l1", dual=False))),
                ('classification', LinearSVC(penalty="l2"))]))])

    dir_name = os.path.dirname(__file__)

    pipeline = joblib.load(dir_name+"/../Resources/SVM_fit.joblib")
    
    y_pred = pipeline.predict(X_test)
    y_pred = [ -1 if y==0 else 1 for y in y_pred ]
    return y_pred




def build_model():
    import tensorflow as tf
    from transformers import TFBertForSequenceClassification
    model= TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model

def convert_example_to_feature(tweet):
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return tokenizer(tweet, add_special_tokens=True,
                                    max_length=None,
                                    pad_to_max_length=True,
                                    return_attention_mask=True,
                                    return_token_type_ids=False)
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def run_best_model(X_test):
    test = []
    for elem in X_test:
        tweet = ''
        for word in elem:
            tweet+=word+' '
        test.append(tweet)
    model = build_model()
    model.load_weights("./NN_WEIGHTS/full_training_bert_weights_1M.h5")
    bert_test = convert_example_to_feature(test)
    input_ids = tf.convert_to_tensor(bert_test.get('input_ids'))
    attention_mask =tf.convert_to_tensor(bert_test.get('attention_mask'))
    y_pred = model.predict([input_ids,attention_mask])
    predictions = [softmax(x) for x in y_pred[0]]
    output = []
    for elem in predictions:
        if elem[0]>elem[1] : x=-1
        else : x = 1
        output.append(x)
    return output