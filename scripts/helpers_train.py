#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np
import pickle
import pandas as pd
from cleaning.data_cleaning import *
import csv

import os.path



import csv
## Helper function that will perform cleaning methods to (full or sample) training data and test data 
## and return cleaned pos_tweets,neg_tweets,test_tweets.
## if load=True then just load the cleaned_data

def clean_reader(file_path):
        x = list(open(file_path, "r", encoding='utf-8').readlines())
        x = [s.strip() for s in x]
        tweets = []
        for elem in x:
            if elem!='':
                tweet=''
                for word in elem.split(','):
                    tweet+=word+' '
                tweets.append(tweet)
        return tweets

def load_cleaned_data(load=False, full=True,
                        emojis=True, repetitions=True,
                        numbers=True, hashtag=True,
                        apostrophes=False, tokenizing=True,
                        slang=True, spelling=True,
                        punctuations=True, stop_words=True,
                        stemming=True, lemmatizing=True,file_in=None):

    dir_name = os.path.dirname(__file__)
    dir_name = dir_name+'/../'

    if full==True: 
        pos_train= dir_name+'Data/train_pos_full.txt'
        neg_train= dir_name+'Data/train_neg_full.txt'
        output_pos= dir_name+'cleaned_data/cleaned_train_pos_full.txt'
        output_neg= dir_name+'cleaned_data/cleaned_train_neg_full.txt'
    else: 
        pos_train= dir_name+'Data/train_pos.txt'
        neg_train= dir_name+'Data/train_neg.txt'
        output_pos= dir_name+'cleaned_data/cleaned_train_pos.txt'
        output_neg= dir_name+'cleaned_data/cleaned_train_neg.txt'
    
    test_data= dir_name+'Data/test_data.txt'
    output_test= dir_name+'cleaned_data/cleaned_test_data.txt'

    if file_in:
        test_data= dir_name+file_in
        output_test= dir_name+'cleaned_data/cleaned_data.txt'
    

    if not load:

        pos_tweets = list(open(pos_train, "r", encoding='utf-8').readlines())
        pos_tweets = [s.strip() for s in pos_tweets]
        neg_tweets = list(open(neg_train, "r", encoding='utf-8').readlines())
        neg_tweets = [s.strip() for s in neg_tweets]
        test = list(open(test_data, "r", encoding='utf-8').readlines())
        test = [s.strip() for s in test]
        test = [x.split(',',1)[1] for x in test]

        # We will have to clean the data
        if repetitions:
            print("Ommiting repetitions")
            pos_tweets = [ommit_repetitions(tweet) for tweet in pos_tweets]
            neg_tweets = [ommit_repetitions(tweet) for tweet in neg_tweets]
            test = [ommit_repetitions(tweet) for tweet in test]
            
            
        if emojis:
            print("Translating emojis")
            pos_tweets = [translate_emoji(tweet) for tweet in pos_tweets]
            neg_tweets = [translate_emoji(tweet) for tweet in neg_tweets]
            test = [translate_emoji(tweet) for tweet in test]
        
        if slang:
            print('dealing with slang words')
            pos_tweets = [deal_slang(tokens) for tokens in pos_tweets]
            neg_tweets = [deal_slang(tokens) for tokens in neg_tweets]
            test = [deal_slang(tokens) for tokens in test]
    
        if numbers:
            print("removing numbers")
            pos_tweets = [remove_numbers(tweet) for tweet in pos_tweets]
            neg_tweets = [remove_numbers(tweet) for tweet in neg_tweets]
            test = [remove_numbers(tweet) for tweet in test]
            
        
        if hashtag:
            print("adding <tag> for hashtags")
            pos_tweets = [add_hashtag(tweet) for tweet in pos_tweets]
            neg_tweets = [add_hashtag(tweet) for tweet in neg_tweets]
            test = [add_hashtag(tweet) for tweet in test]
        
        
        if apostrophes:
            print("processing apostrophes")
            pos_tweets = [apostrophe(tweet) for tweet in pos_tweets]
            neg_tweets = [apostrophe(tweet) for tweet in neg_tweets]
            test = [apostrophe(tweet) for tweet in test]
        
        if spelling:
            print('correcting spelling mistakes')
            pos_tweets = [correct_spelling_from_dict(tweet) for tweet in pos_tweets]
            neg_tweets = [correct_spelling_from_dict(tweet) for tweet in neg_tweets]
            test = [correct_spelling_from_dict(tweet) for tweet in test]
            
        if tokenizing:
            print('tokenizing')
            pos_tweets = [text_processor.pre_process_doc(tweet) for tweet in pos_tweets]
            neg_tweets = [text_processor.pre_process_doc(tweet) for tweet in neg_tweets]
            test = [text_processor.pre_process_doc(tweet) for tweet in test]
        
        
        if punctuations:
            print("removing ponctuations")
            pos_tweets = remove_punctuations(pos_tweets)
            neg_tweets = remove_punctuations(neg_tweets)
            test = remove_punctuations(test)
        
    
        if slang:
            print('dealing with slang words')
            pos_tweets = [deal_slang(tokens) for tokens in pos_tweets]
            neg_tweets = [deal_slang(tokens) for tokens in neg_tweets]
            test = [deal_slang(tokens) for tokens in test]
            
            
        
        if stop_words:
            print("removing stop words")
            pos_tweets = [remove_stop_words(tweet) for tweet in pos_tweets]
            neg_tweets = [remove_stop_words(tweet) for tweet in neg_tweets]
            test = [remove_stop_words(tweet) for tweet in test]
        
        
        with open(output_pos,'w',encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerows(pos_tweets)
        
        with open(output_neg,'w',encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerows(neg_tweets)
            
        with open(output_test,'w',encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerows(test)

        pos_tweets = clean_reader(output_pos)
        neg_tweets = clean_reader(output_neg)
        test = clean_reader(output_test)

    else:

        pos_tweets = clean_reader(output_pos)
        neg_tweets = clean_reader(output_neg)
        test = clean_reader(output_test)

    return pos_tweets,neg_tweets,test


## This will load the created embeddings (after applying cooc.py and glove_template.py) and create a corresponding dataframe
def load_word_embeddings_df(path_embeddings,path_vocab):
    embeddings = np.load(path_embeddings)
    vocab = pickle.load(open(path_vocab, "rb")) #this file was generated by executing vocab.sh
    word_embedding = {}
    for key in vocab.keys():
        word_embedding[key] = embeddings[vocab.get(key)] # keys in this dict are not encoded 
    return pd.DataFrame(word_embedding).T


## A helper function that will calculate average word vectors for each tweet using the word embeddings
def average_word_vectors(tweets ,word_embedding):
    error = 0
    avg_word_vectors = np.zeros((len(tweets), word_embedding.shape[1] ))
    for i, tweet in enumerate(tweets):
        
        split_tweet = tweet.split()
        nb_words = 0
        
        for word in split_tweet:
            try:
                avg_word_vectors[i] += word_embedding.loc[word].to_numpy()
                nb_words += 1

            except KeyError: 
                continue
        if (nb_words != 0):
            avg_word_vectors[i] /= nb_words
        
    return avg_word_vectors



## This function will create a train_df containings both positive and negative tweets, and a test_df.
def create_train_test_dfs(sample_pos_tweets_sample,sample_neg_tweets_sample,test):

    pos_df = pd.DataFrame({"tweets":sample_pos_tweets_sample,"sign":np.ones(len(sample_pos_tweets_sample))})
    neg_df = pd.DataFrame({"tweets":sample_neg_tweets_sample,"sign":np.zeros(len(sample_neg_tweets_sample))})
    train_df = pd.concat([pos_df,neg_df])
    train_df = train_df.sample(frac = 1)
    train_df = train_df.reset_index()
    train_df = train_df.drop(columns='index')
    test_df = pd.DataFrame({'tweets': test})
    return train_df,test_df



def create_csv_submission(y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    ids=np.arange(1,10001)
    with open(name, 'w',newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
