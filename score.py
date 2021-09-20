

import numpy as np
import pandas as pd
import os
import pickle
import json
import time
from azureml.core.model import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_seq_len = 1000

# Called when the deployed service starts
def init():
    global model

    
 
    # load models
    #model = Model.get_model_path(model_name = 'my_model')
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './model.h5')
    model = model = load_model(model_path)    
    

# Handle requests to the service
def run(data):
    try:
        data = json.loads(data)    
        tokenizer = Tokenizer(num_words = 100)   
        _seq = tokenizer.texts_to_sequences(data['text'])
        x_test = pad_sequences(_seq, maxlen=max_seq_len)
        result = np.mean(np.array([[float(x)] for x in model.predict([x_test])[0]]))
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
