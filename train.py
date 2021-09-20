


from azureml.core import Run
import os


from urllib import request
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import decomposition
from sklearn import preprocessing
sns.set()
# Python program to read 
# json file 
 
import json 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import TFOptimizer
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from keras.initializers import Constant
import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
stemming = PorterStemmer()
Lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words("english"))
stops2 = set(stopwords.words("french"))

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

# The following import and function call are the only additions to code required
# to automatically log metrics and parameters to MLflow.



import matplotlib.pyplot as plt
#%matplotlib inline


# Get the experiment run context
run = Run.get_context()





print("Prepare the dataset...")
# Prepare the dataset
commentData = pd.read_csv('P7_01_0_training.1600.processed.noemoticon.csv', encoding='ISO-8859-1')
df = commentData
train_df = df.drop(["id","date","flag","flag","user"], axis=1)
train_df = train_df.rename(columns={"target": "Sentiment"})

def clean_str(x):
    
    x = str(x)
   
    # Convert to lower case
    text = x.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    
    # Stemming
    #stemmed_words = [stemming.stem(w) for w in token_words]

    # lemmatizer    
    lemmatizer_words = [Lemmatizer.lemmatize(w) for w in token_words]
    
    # Remove stop words
    meaningful_words = [w for w in token_words if not w in stops]
    
    # Remove stop words
    meaningful_words2 = [w for w in meaningful_words if not w in stops2]   
    
    # Rejoin meaningful stemmed words
    joined_words = ( " ".join(meaningful_words2))

    
    # Return cleaned data
    return joined_words


train_df['text'] = train_df['text'].apply(clean_str)

df = train_df
df_0 = df[df['Sentiment'] == 0].sample(frac=1)
df_4 = df[df['Sentiment'] == 4].sample(frac=1)

# we want a balanced set for training against - there are 7072 `0` examples
sample_size = min(len(df_0), len(df_4))

data = pd.concat([df_0.head(sample_size), df_4.head(sample_size)]).sample(frac=1)
sentences = data['text']
tokenizer = Tokenizer(num_words = 4000)
tokenizer.fit_on_texts(sentences)
sequence = tokenizer.texts_to_sequences(sentences)

import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

index_of_words = tokenizer.word_index
embed_num_dims = 100
max_seq_len = 1000
padded_seq = pad_sequences(sequence , maxlen = max_seq_len )
Y = pd.get_dummies(data['Sentiment']).values

X_train,X_test,Y_train,Y_test = train_test_split(padded_seq,Y ,train_size = 0.55)

# Glove
f = open('glove.6B.100d.txt', encoding="utf8")
embedd_index = {}
for line in f:
    val = line.split()
    word = val[0]
    coff = np.asarray(val[1:],dtype = 'float')
    embedd_index[word] = coff

f.close()

embedding_matrix = np.zeros((len(index_of_words) + 1, embed_num_dims))

tokens = []
labels = []

for word,i in index_of_words.items():
    temp = embedd_index.get(word)
    if temp is not None:
        embedding_matrix[i] = temp
        
#for plotting
        tokens.append(embedding_matrix[i])
        labels.append(word)

#Embedding layer before the actaul BLSTM 
embedd_layer = Embedding(len(index_of_words) + 1 , embed_num_dims , input_length = max_seq_len , weights = [embedding_matrix])

print("Building model...")
# Bidirectional LSTM model avec GLOVE 
model = Sequential()
model.add(embedd_layer)
model.add(Bidirectional(LSTM(30 , return_sequences = True , dropout = 0.1 , recurrent_dropout = 0.1)))
model.add(GlobalMaxPooling1D())
model.add(Dense(30,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation = 'sigmoid'))

# Train LSTM model
add = Adam(lr = 0.01)
model.compile(loss = 'categorical_crossentropy' , optimizer = add , metrics = ['accuracy'])
hist = model.fit(X_train,Y_train,epochs = 5 , batch_size = 100, validation_data = (X_test,Y_test))

score = model.evaluate(X_test,Y_test, batch_size= 100, verbose=1)
print("Test score:", score[0])
print("Test accuracy:", score[1])
print("Accuracy: %.2f%%" % (score[1] * 100))
# Save the model 
#for heavy model architectures, .h5 file is unsupported.
weigh= model.get_weights();    pklfile= "./model1.pkl"
try:
    fpkl= open(pklfile, 'wb')    #Python 3     
    pickle.dump(weigh, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
    fpkl.close()
except:
    fpkl= open(pklfile, 'w')    #Python 2      
    pickle.dump(weigh, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
    fpkl.close()


from tensorflow.python.keras.optimizers import TFOptimizer
import tensorflow as tf
model_path = "./model"
tf.keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True, save_format=None)

import onnxmltools

onnx_model = onnxmltools.convert_keras(model) 

onnxmltools.utils.save_model(onnx_model, 'model.onnx')

# Save the model 
from tensorflow.python.keras.optimizers import TFOptimizer
import tensorflow as tf
model_path = "./model.h5"
tf.keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True, save_format='h5')

run.complete()
