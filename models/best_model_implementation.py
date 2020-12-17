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
    return 0

def best_model(X_test):
    y_pred = model.predict(X)
    return y_pred