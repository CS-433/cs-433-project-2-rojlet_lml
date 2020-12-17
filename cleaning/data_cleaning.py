#!/usr/bin/env python
# coding: utf-8
import os.path
dir_name = os.path.dirname(__file__)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

## Function that ommits repetitions in a tweet
def ommit_repetitions(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'\b(\w+)( \1\b)+', r'\1', tweet)
    return tweet


## Function that translates emojis to corresponding meaningful word
def translate_emoji(text):
    # HAPPY ---> :) | =) | :-) | : ) | :') | =') | = ) | (: | (= | (-: | ( : | (': | ('= | ( = 
    text = re.sub(r'(:\s?\)|\=\s?\)|:-\)|:\'\)|\=\'\)|\(\=|\(\s?:|\(-:|\(\s?:|\(\':|\(\'=|\(\s?\=)', ' happy ', text)
    # SAD -----> :( | =( | :-( | : ( | = ( | ): | )= | )-: | ) : | ) =
    text = re.sub(r'(:\s?\(|\=\s?\(|:-\(|\)\s?:|\)\s?\=|\)-:)', ' sad ', text)
    # Laugh emojis = :D | :-D | xd | x-d
    text = re.sub(r'(:\s?d|:-d|x-?d|xd|xdd)', ' laugh ', text)
    # LOVE ----> <3 
    text = re.sub(r'(<3)', ' love ', text)
    # KISS ----> :* | =* 
    text = re.sub(r'(:\*|:\*\*)', ' kiss ', text)
    # WINK ----> ;) | ;-) | ; ) | ;') | (; | (-; | ( ; | ('; | ;p | :p 
    text = re.sub(r'(;\s?\)|;-?\)|;\'\)|;-?d|\(\s?;|\(-;|\(\';|\(-?;|:p|;p)', ' wink ', text)
    # CRY -----> :'( | ='( | )': | )'= |
    text = re.sub(r'(:,\(|:\'\(|:"\(|=\'\(|\)\':|\)\'=)', ' cry ', text)
    return text

## this function will replace # with <tag>
def add_hashtag(text):
    text = re.sub(r'#', '<tag>', text)
    return text

## This function will remove numbers from a tweet
def remove_numbers(tweet):
    tweet = re.sub("\d+", "", tweet)
    return tweet


## these punctuations will keep #tags and <user> <link> 
punctuations = '!"$%&\'()*+,-./:;=@[\\]^_`{|}~<>'
## This function will remove punctutations from a tweet and replace .. and ... with multistop, ? with question
def remove_punctuations(tweets):
    tweets = [[token for token in tweet if token not in punctuations] for tweet in tweets]
    tweets = [['user' if token=='<user>' else 'number' if token=='<number>' else 'tag' if token=='<tag>' else 'url' if token=="<url>" else token for token in tweet] for tweet in tweets]
    tweets = [['multistop' if token=='..' or token=='...' else 'question' if token=='?' else token for token in tweet] for tweet in tweets]
    return tweets


##This methode transforms words with apostrophes at the end into two words
def apostrophe(text):
    
    # Apostrophe lookup
    text = re.sub(r"i\'m", "i am ", text) # i'm --> i am
    text = re.sub(r"I\'m", "i am ", text) # I'm --> I am
    text = re.sub(r"it\'s","it is",str(text)) #it's --> it is
    text = re.sub(r"he\'s","he is",str(text)) #he's --> he is
    text = re.sub(r"she\'s","she is",str(text)) #she's --> she is
    text = re.sub(r"we\'re","we are",str(text)) #we're --> we are
    text = re.sub(r"they\'re","they are",str(text)) #they're --> they are
    
    text = re.sub(r"there\'s","there is",str(text)) #there's --> there is
    text = re.sub(r"that\'s","that is",str(text)) #that's --> that is
    
    text = re.sub(r"i\'d","i would",str(text)) #i'd --> i would
    text = re.sub(r"he\'d","he would",str(text)) #he'd --> he would
    text = re.sub(r"it\'d","it would",str(text)) #it'd --> it would
    text = re.sub(r"she\'d","she would",str(text)) #she'd --> she would
    text = re.sub(r"we\'d","we would",str(text)) #we'd --> we would
    text = re.sub(r"they\'d","they would",str(text)) #they'd --> they would
    
    text = re.sub(r"i\'ll","i will",str(text)) #i'll --> i will
    text = re.sub(r"he\'ll","he will",str(text)) #he'll --> he will
    text = re.sub(r"it\'ll","it will",str(text)) #it'll --> it will
    text = re.sub(r"she\'ll","she will",str(text)) #she'll --> she will
    text = re.sub(r"we\'ll","we will",str(text)) #we'll --> we will
    text = re.sub(r"they\'ll","they will",str(text)) #they'll --> they will
    
    text = re.sub(r"don\'t","do not",str(text)) #don't --> do not    
    text = re.sub(r"can\'t", "can not", text) #can't --> can not
    text = re.sub(r"cannot", "can not ", text) #cannot --> can not
    text = re.sub(r"could\'t", "could not", text) #could't --> could not
    text = re.sub(r"should\'t", "should not", text) #should't --> should not
    text = re.sub(r"haven\'t", "have not", text) #haven't --> have not
    text = re.sub(r"didn\'t", "did not", text) #didn't --> did not    
    
    text = re.sub(r"what\'s", "what is", text) #what's --> what is
    text = re.sub(r"where\'s", "where is", text) #where's --> where is
    text = re.sub(r"when\'s", "when is", text) #when's --> when is
    text = re.sub(r"why\'s", "why is", text) #why's --> why is   
    
    return text



## EKphrasis text processor, to deal with contractions, unpack hashtags and spell correction for elongated words
## Using Potts tokenizer 
from cleaning.potts_tokenizer import Tokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

tok=Tokenizer(preserve_case=False)

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    #annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
    #annotate = {"emphasis"},
    
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=tok.tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)



## load slang dictionary
slang = list(open(dir_name+"/../Resources/slang_dict.txt", "r", encoding='utf-8').readlines())
slang = [s.lower().strip() for s in slang]
slang_dict={elem.split(':')[0] : elem.split(':')[1] for elem in slang}

## function to deal with slang words
def deal_slang(tokens):
    for token in tokens:
        for key in slang_dict.keys():
            if (token==key):
                token = slang_dict.get(key)
    return tokens


## load and create spelling dictionary
def load_dict(dict_path , spelling_dict) :
    '''
    Creates a dictionnary from a text file
    '''
    dict_ = open(dict_path, 'rb')
    for word in dict_:
        word = word.decode('utf8')
        word = word.split()
        spelling_dict[word[0]] = word[1]
    dict_.close()
    return spelling_dict
spelling_dict = dict()
spelling_dict = load_dict(dir_name+'/../Resources/spelling_dict1.txt', spelling_dict)
spelling_dict = load_dict(dir_name+'/../Resources/spelling_dict2.txt', spelling_dict)



## function to correct spelling from two dictionaries found in internet (spelling_dict1 and spelling_dict2)
def correct_spelling_from_dict(text , dic=spelling_dict):
    '''
    This method corrects words in the tweet using the slang/spelling dictionnaries aleready created
    '''
    text = text.split()
    for i in range(len(text)):
        if text[i] in dic.keys():
            text[i] = dic[text[i]]
    text = ' '.join(text)
    return text

# Loading stopwords list from NLTK
stoplist = set(stopwords.words("english"))
def remove_stop_words(text):
    '''
    This method removes stop words from the tweet
    '''
    new_text = text.copy()
    for word in new_text:
        if word in stoplist:
            new_text.remove(word)
    return new_text



#  Lemmatizing
lemmatizer = WordNetLemmatizer()
def perform_lemmatizing(text):
    '''
    lemmatize words: dances, dancing ,danced --> dance
    '''
    lemmatized = []
    for word in text:
        try:
            lemmatized.append(lemmatizer.lemmatize(word).lower())  #check problem doesnt work correctly
        except Exception:
             lemmatized.append(word)
    return lemmatized


# Stemming
stemmer = PorterStemmer()
def perform_stemming(text):
    '''
    stemm words: Car, cars --> car
    '''
    x = [stemmer.stem(t) for t in text]
    return x