from flask import Flask , request , abort
import requests
import json
import numpy as np
import pandas as pd
from app.Config import *
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from pythainlp.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import pickle
model = load_model('app\my_model')
# Load the vectors
word2vec_model = KeyedVectors.load_word2vec_format('LTW2V_v0.1.bin', binary=True, unicode_errors='ignore')
'''max_leng = 0
for word_seq in tokenized_train_text:
  if len(word_seq) > max_leng:
    max_leng = len(word_seq)'''

app = Flask(__name__)
##Test Git
@app.route('/')
def hello():
    return 'Hello' , 200

@app.route('/webhook', methods=['POST','GET'])
def webhook():
    if request.method == 'POST':
        payload = request.json
        Reply_token = payload['events'][0]['replyToken']
        print(Reply_token)
        message = payload['events'][0]['message']['text']
        print(message)
        text = message
        df = pd.read_csv (r"C:\Users\bossi\Desktop\Clone ChatBot\LineChatBot\project data set.csv")
        print (df)
        df.columns =['input_text', 'labels']
        data_df=df
        print(data_df)
        # Lower case of all labels and replace to the old one
        data_df['cleaned_labels'] = data_df['labels'].str.lower()
        # we no longer need raw_label column
        data_df.drop('labels', axis=1, inplace=True)
        # Filter `garbage` entries
        data_df = data_df[data_df['cleaned_labels'] != 'garbage']
        # Strip the leading and trailing whitespaces
        cleaned_input_text = data_df['input_text'].str.strip()
        # Change the text into lowercased string
        cleaned_input_text = cleaned_input_text.str.lower()
        data_df['cleaned_input_text'] = cleaned_input_text
        # we are no longer need input_text column
        data_df.drop('input_text', axis=1, inplace=True)
        # Drop the duplicated entry and keep only the first one
        data_df = data_df.drop_duplicates("cleaned_input_text", keep="first")
        # Get cleaned input text as a list
        input_text = data_df["cleaned_input_text"].tolist()
        # Get labels from the cleaned dataset
        labels = data_df["cleaned_labels"].tolist()
        # Split train test
        train_text, test_text, train_labels, test_labels = train_test_split(input_text, labels, train_size=0.9, random_state=12345)        
        # tokenize
        word_seq = word_tokenize(text)
        # map index
        word_indices = map_word_index(word_seq)
        # padded to max_leng
        padded_wordindices = pad_sequences([word_indices], maxlen=10, value=0)
        # predict to get logit
        logit = model.predict(padded_wordindices, batch_size=32)
        # get prediction
        unique_labels = set(train_labels)
        index_to_label = [label for label in sorted(unique_labels)]
        predict = [ index_to_label[pred] for pred in np.argmax(logit, axis=1) ][0]
        print(np.argmax(logit, axis=1))
        ReplyMessage(Reply_token,predict,Channel_access_token)
        return request.json, 200
    elif request.method == 'GET' :
        return 'this is method GET!!!' , 200
    else:
        abort(400)

def ReplyMessage(Reply_token, TextMessage, Line_Acees_Token):
    LINE_API = 'https://api.line.me/v2/bot/message/reply'

    Authorization = 'Bearer {}'.format(Line_Acees_Token) ##ที่ยาวๆ
    print(Authorization)
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization':Authorization
    }

    data = {
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }]
    }

    data = json.dumps(data) ## dump dict >> Json Object
    r = requests.post(LINE_API, headers=headers, data=data) 
    return 200
  
def map_word_index(word_seq):
  """
    mapping word sequence to list of indices
  """
  indices = []
  for word in word_seq:
    if word in word2vec_model.key_to_index:
      indices.append(word2vec_model.key_to_index[word] + 1)
    else:
      # consider unknown word as '' (empty string); map to index 1
      indices.append(1)
  return indices  
  