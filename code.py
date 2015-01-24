
# coding: utf-8

## Import

# In[252]:

# General
import os, csv, time
import numpy as np
import pandas as pd

# Twitter API
import tweepy
from tweepy.streaming import StreamListener
from tweepy import Stream

# unicode tokenizer
import ucto

# scikit-learn for machine learning algortihms
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.cross_validation import LeaveOneOut
from sklearn import cross_validation

from sklearn.feature_selection import chi2

from random import shuffle

# Saving and Reading Python objects
import pickle 

# import logging # not sure


### Settings

# In[253]:

os.chdir("/home/pold/Dropbox/Uni/Radboud/Text_Mining/Endterm/hashpy")


### General Classifier

### Read Aggressive Tweets from TSV files

# In[254]:

def create_key(x, path = "tweets_better/aggressive_quest"):
    '''Create a file name from a number'''
    return os.path.join(path, "fold_" + str(x) + ".txt")

# Create a list of file names of the manually labeled tweets.
files = [create_key(x) for x in xrange(0,10)]

# Store everything in a pandas data frame.
list = []
for file in files:
    df = pd.read_table(file.encode('utf-8'),
                       names = ('category', 'user', 'date', 'time', 'message'), 
                       header = None,
                       index_col = None)
    list.append(df)
        
frame = pd.concat(list, axis=0, keys = files)


### Use Ucto for Dutch tokenization

# In[255]:

def remove_non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)

# Ucto settings
settingsfile = "/etc/ucto/tokconfig-nl-twitter"
tokenizer = ucto.Tokenizer(settingsfile)

def ucto_tokenizer(str):
    '''Tokenize string'''
    remove_non_ascii(str)
    tokenizer.process(unicode(str))
    tokens = [unicode(token).encode('utf-8') for token in tokenizer]
    for pos, token in enumerate(tokens): 
        # Create dummy values for users and urls.
        if token.startswith('http:'): tokens[pos] = 'url'
        if token.startswith('@'): tokens[pos] = 'user'
    return tokens


### Aggressive Hashtags - Wordlist

# In[256]:

# List a selected hashtags
with open("wordlists/wordlist_selection.txt") as f:
    selected_hashtags = f.read().splitlines()


## Twitter

### Authentication

# In[ ]:

consumer_key = 'zdprSqld7JxkSjaeqQppemW5y'
consumer_secret = ''
access_token = '1612711080-64a9w2e4BOTV4id4HTqLpzaO8Z6ag0pNVE9axZv'
access_token_secret = ''


### Connect to Twitter

# In[258]:

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


### Create a Query

# In[236]:

def create_query(wordlist):
    '''Create a string for an OR query on Twitter'''
    return " OR ".join(wordlist)

def create_negate_query(wordlist):
    '''Create a string for a negated query on Twitter'''
    return " ".join(["-" + word for word in wordlist])


### Collect Tweets

# In[ ]:

results_aggressive = []
results_non_aggressive = []

# Create queries
query_pos = create_query(selected_hashtags)
query_neg = create_negate_query(selected_hashtags)

# Connect to Twitter and collect aggressive tweets
for tweet in tweepy.Cursor(api.search, q = query_pos,  lang = "nl").items(1500):
    results_aggressive.append(tweet)
    
# Wait some time to get not rate limited    
time.sleep(60)

# Same for non-aggressive Tweets (equal amount to aggressive tweets)
for tweet in tweepy.Cursor(api.search
                         , q = query_neg
                         ,  lang = "nl").items(len(results_aggressive)):
    results_non_aggressive.append(tweet)   


### Count Hashtags

# In[259]:

hash_dict = { }

# Initialize dictionary
for word in selected_hashtags:
    hash_dict[word] = 0

# Count occurances of hashtags    
for tweet in results_aggressive:
    for word in selected_hashtags:
        if word in tweet.text: hash_dict[word] += 1

# Print collected same stats about collected tweets            
#print("Amount of aggressive tweets:", len(results_aggressive))
#print("Amount of non-aggressive tweets:", len(results_non_aggressive))
#print("Hashtags frequencies:")
#print(hash_dict)


### Additional Feature Extractor (Linguistic Markers)

# In[260]:

# Transformer for extracting linguistic features in the message        
# Inspired by (and uses code of) Andreas Mueller:
# https://github.com/amueller/kaggle_insults/
class LinguisticMarkers(BaseEstimator):

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        
        n_words = [len(ucto_tokenizer(remove_non_ascii(tweet))) for tweet in documents]
        n_chars = [len(tweet) for tweet in documents]
        
        # number of uppercase words
        allcaps = [np.sum([w.isupper() for w in comment.split()])
               for comment in documents]
        
        allcaps_ratio = (np.array(allcaps) / np.array(n_chars, dtype=np.float)) 
        
        # Total number of exlamation marks
        exclamation = [c.count("!") for c in documents]
        
        exclamation_ratio = (np.array(exclamation) / np.array(n_chars, dtype=np.float))
        
        # Total number of addressed users
        users = [c.count("@") for c in documents]

        return np.array([allcaps
                      , allcaps_ratio
                      , exclamation
                      , exclamation_ratio
                      , users
]).T 


### Additional Features (Swear Words)

# In[261]:

# Transformer for extracting swear words in the message        
# Inspired by (and uses code of) Andreas Mueller:
# https://github.com/amueller/kaggle_insults/
class SwearWords(BaseEstimator):
    
    def __init__(self):
        with open("wordlists/swearwords-nl.txt") as f:
            swearwords = [l.strip() for l in f.readlines()]
        self.swearwords_ = swearwords

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        
        # number of words
        n_words = [len(ucto_tokenizer(remove_non_ascii(tweet))) for tweet in documents]
        
        # number of swear words:
        n_swearwords = [np.sum([c.lower().count(w) for w in self.swearwords_])
                                                for c in documents]
        
        swearwords_ratio = np.array(n_swearwords) / np.array(n_words, dtype=np.float)
        
        return np.array([n_swearwords
                         , swearwords_ratio]).T 


### k-Fold Cross Validation

# In[262]:

def leave_one_out_evaluation(dataset
                           , testset = None
                           , pipe = 0
                           , add_selection_to_stopwords = False):
    
    if add_selection_to_stopwords:
        stopwords_path_to_file = "wordlists/dutch-stop-words-added.txt"
    else:
        stopwords_path_to_file = "wordlists/dutch-stop-words.txt" 
        
    with open(stopwords_path_to_file) as f:
        dutch_stop_words = f.read().splitlines()

    # Feature Extractors    
    count_vect = CountVectorizer(stop_words = dutch_stop_words
                                 , tokenizer = ucto_tokenizer
                                 , lowercase = True)

    linguisticmarkers = LinguisticMarkers()
    swearwords = SwearWords()

    # Combined features
    combined0 = count_vect

    combined1 = FeatureUnion([("feat1", count_vect)
                            , ("feat2", linguisticmarkers)])
    
    combined2 = FeatureUnion([("feat1", count_vect)
                            , ("feat3", swearwords)])    

    combined3 = FeatureUnion([("feat1", count_vect)
                            , ("feat2", linguisticmarkers)
                            , ("feat3", swearwords)])
    
    
    
    combined4 = linguisticmarkers
    combined5 = FeatureUnion([("feat2", linguisticmarkers)
                            , ("feat3", swearwords)])   
    
    combined6 = swearwords       

    
    pipe_list = [combined0, combined1, combined2
               , combined3, combined4, combined5, combined6]

    tfidf = TfidfTransformer(sublinear_tf = True
                           , use_idf = True)    


    # Create classifier
    clf = LogisticRegression(penalty = 'l2', dual = False) 


    # Create pipelines
    pipeline = Pipeline([('vect', pipe_list[pipe]),
                         ('tfidf', tfidf),
                         ('clf', clf)])
    
    # Create collectors
    predicted_per_fold = []
    target_per_fold = pd.Series()

    # If no test set is given, train the classifier do a k-fold cross-validation.
    if testset is None:
    
        # Create random test and fold sets
        kf = cross_validation.KFold(n=len(dataset), n_folds=10, shuffle=True)

        for train_num, test_num in kf:

            messages_train = dataset['message'][train_num]
            categories_train = dataset['category'][train_num]

            messages_test = dataset['message'][test_num]
            categories_test = dataset['category'][test_num]         


            target_per_fold = pd.concat([target_per_fold, categories_test])

            # train classifier 

            # Remember to chose combined 1, 2 or 3 in Pipeline
            pipeline = pipeline.fit(messages_train, categories_train)

            predicted = pipeline.predict(messages_test)
            predicted_per_fold.append(predicted)

        # flatten predicted_per_fold
        predicted_flattened = [item for sublist in predicted_per_fold for item in sublist]
        return (np.array(predicted_flattened), target_per_fold)
    
    
    # If a test set is available, train the classifier on the dataset and 
    # evaluate it on the test set
    else:                              
        
        # Remember to chose combined 1, 2 or 3 in Pipeline
        pipeline = pipeline.fit(dataset['message'], dataset['category'])
        
        predicted = pipeline.predict(testset['message'])
        
        return (predicted, testset['category'])
        
    


## Evaluation

# In[269]:

def labels2ints(ls):
    '''Convert aggressive to 1 and non-aggressive to 0'''
    out = []
    for label in ls:
        if label == 'aggressive': out.append(1)
        else: out.append(0) 
    return out

def evaluation(tpf, ppf):
    ''' Perform the complete evaluation'''
    predicted_per_fold_array = np.array(ppf)

    target_int = labels2ints(tpf)
    predicted_int = labels2ints(predicted_per_fold_array)
    
    #print("Accuracy")
    #print(metrics.accuracy_score(target_int, predicted_int))  
    
    #print("Classic Report")
    #print(metrics.classification_report(target_int, predicted_int))  


### Run Evaluation of manually labeled data set

### Run Cross-validation with function (distantly supervised) 

# In[ ]:

# Extract text from tweets
aggressive_train =  [tweet.text for tweet in results_aggressive]
non_aggressive_train = [tweet.text for tweet in results_non_aggressive]

messages_train = aggressive_train + non_aggressive_train

categories_train = ['aggressive'] * len(aggressive_train)                  + ['non_aggressive'] * len(non_aggressive_train)

distant_frame = pd.DataFrame()
distant_frame['message'] = messages_train
distant_frame['category'] = categories_train

for i in xrange(0,7):
    # print "i is", i
    # Pay attention to add_selection_to_stopwords!
    predicted_per_fold, target_per_fold =     leave_one_out_evaluation(frame
                           , distant_frame
                           , pipe = i
                           , add_selection_to_stopwords = False)
    evaluation(predicted_per_fold, target_per_fold)


### Validation: Distant Supervision -> Manually Labeled

### Interactive test for getting a feeling for the accuracy (using the manually labeled data)

# In[ ]:

interactive_test = [raw_input('Please type message that should be categorized: ')]

train = frame.ix[[create_key(y) for y in range(0,10)]]
    
messages_train = train['message']
categories_train = train['category']
    
# train classifier 
pipeline = pipeline.fit(messages_train, categories_train)

predicted = pipeline.predict(interactive_test)
#print(predicted)


### Inspect Bag of Words

# In[ ]:

aggressive_train =  [tweet.text for tweet in results_aggressive]

non_aggressive_train = [tweet.text for tweet in results_non_aggressive]

# messages_train = aggressive_train + non_aggressive_train

categories_train = ['aggressive'] * len(aggressive_train)                  + ['non_aggressive'] * len(non_aggressive_train)

matrix = count_vect.fit_transform(aggressive_train, ['aggressive'] * len(aggressive_train))
freqs = [(word, matrix.getcol(idx).sum()) for word, idx in count_vect.vocabulary_.items()]
#sort from largest to smallest
# print sorted (freqs, key = lambda x: -x[1])


### Pickle

# In[ ]:

# Getting back the objects:
with open('results_854.pickle') as f:
    results_aggressive, results_non_aggressive = pickle.load(f)


# In[ ]:

# Saving the objects:
with open('final_evaluation.pickle', 'w') as f:
    pickle.dump([results_aggressive, results_non_aggressive], f)

