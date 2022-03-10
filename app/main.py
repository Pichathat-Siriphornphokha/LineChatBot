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
from random import randint


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
        df = pd.read_csv (r"C:\Users\bossi\Desktop\Clone 2 ChatBot\GitHub\LineChatBot\project data set.csv")
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
        padded_wordindices = pad_sequences([word_indices], maxlen=32, value=0)
        # predict to get logit
        logit = model.predict(padded_wordindices, batch_size=32)
        # get prediction
        unique_labels = set(train_labels)
        index_to_label = [label for label in sorted(unique_labels)]
        predict = [ index_to_label[pred] for pred in np.argmax(logit, axis=1) ][0]
        print(np.argmax(logit, axis=1))
        if(predict == 'สิทธิการรักษา'):
          ReplyMessage(Reply_token,'''สิทธิการรักษาพยาบาล มี 3 ระบบใหญ่ คือ
1.สิทธิหลักประกันสุขภาพแห่งชาติ หรือที่รู้จักกันในนาม สิทธิ 30 บาท หรือ สิทธิบัตรทอง คุ้มครองบริการรักษาพยาบาลให้กับคนไทยทุกคนที่มีหมายเลขบัตรประชาชน 13 หลัก และต้องไม่เป็นผู้ที่มีสิทธิประกันสังคมและสวัสดิการรักษาพยาบาลของข้าราชการ เมื่อเจ็บป่วยสามารถเข้ารับบริการรักษาพยาบาลได้ที่โรงพยาบาลของรัฐ และสถานีอนามัย สำนักงานสาธารณสุขจังหวัด
2.สิทธิสวัสดิการการรักษาพยาบาลของราชการ คุ้มครองบริการรักษาพยาบาลให้กับข้าราชการและบุคคลในครอบครัว ได้แก่ บิดา มารดา คู่สมรส และบุตรที่ถูกต้องตามกฎหมาย เมื่อเจ็บป่วยสามารถเข้ารับบริการรักษาพยาบาลได้ที่โรงพยาบาลของรัฐ
3.สิทธิประกันสังคม คุ้มครองบริการ''',Channel_access_token)
        elif(predict == 'ทักทาย'):
          ReplyMessage(Reply_token,'''สวัสดีครับ มีข้อสงสัยอะไรสามารถสอบถามได้เลยครับ''',Channel_access_token)
        elif(predict == 'ช่องทางการติดต่อ '):
          ReplyMessage(Reply_token,'''ที่อยู่: หมู่ที่ 8 95 ทางคู่ขนาน ถ. พหลโยธิน ตำบล คลองหนึ่ง อำเภอคลองหลวง ปทุมธานี 12120 
โทรศัพท์: 02 926 9999 
เว็บไซต์: https://www.hospital.tu.ac.th/
''',Channel_access_token)
        elif(predict == 'จะได้เตียงวันไหน'):
          ReplyMessage(Reply_token,'''เนื่องจากโรงพยาบาลมีผู้ป่วยมาใช้บริการจำนวนมาก แต่เตียงนอนมีจำนวนจำกัด เราจึงมีการบริหารจัดการเตียงให้ผู้ป่วยได้เข้านอนโรงพยาบาลได้เร็วที่สุด โดยมีการตรวจสอบเตียงว่างทุกวัน เพื่อบริหารจัดการเตียงตามลำดับการจอง พร้อมทั้งพิจารณาถึงความเร่งด่วนของผู้ป่วยแต่ละราย โดยกำหนดระยะเวลาไม่เกิน 7 วันหลังจากวันจอง มีการโทรแจ้งผู้ป่วยล่วงหน้า เพื่อให้เตรียมความพร้อมสำหรับการมานอนโรงพยาบาล ในกรณีที่มีเตียงว่างเพิ่มเติมระหว่างวัน เพื่อลดการสูญเสียเตียงโดยเปล่าประโยชน์ จะโทรติดต่อ ผู้ป่วยที่มีความพร้อม และสามารถเดินทางมาได้สะดวกรวดเร็ว ให้เข้านอนโรงพยาบาลโดยทันท่วงที''',Channel_access_token)
        elif(predict == 'ถามเกี่ยวกับวิธีรักษา'):
          ReplyMessage(Reply_token,'''https://thaicancersociety.com/wp-content/uploads/lung-cancer-infographic-2020-16-2000x1500.jpg''',Channel_access_token)
        elif(predict == 'การกิน'):
          ReplyMessage(Reply_token,'''อาหารประเภทเนื้อสัตว์ ผัก ผลไม้ประเภทต่างๆ จะทำให้ร่างกายแข็งแรงมีส่วนช่วยให้ผู้ป่วยมะเร็งปอดทนต่อผลข้างเคียงจากการรักษา และทำให้มีการตอบสนองต่อการรักษาที่ดี  
หลีกเลี่ยงอาหารที่มีโอกาสทำให้ท้องเสีย 
งดการทานผักสดและผลไม้เปลือกบาง เช่น องุ่น ชมพู่ และผลไม้ไม่มีเปลือก เช่น สตรอว์เบอร์รี โดยเฉพาะในช่วง 14 วันแรก หลังได้รับยาเคมีบำบัด''',Channel_access_token)
        elif(predict == 'การนอน'):
          ReplyMessage(Reply_token,'''ควรนอนหลับพักผ่อนให้เพียงพออย่างน้อย 6-8 ชั่วโมง''',Channel_access_token)
        elif(predict == 'การออกกำลังกาย'):
          ReplyMessage(Reply_token,'''สามารถออกกำลังกายได้ตามแรงที่มี การออกกำลังกายที่เหมาะที่สุดคือ การเดินออกกำลังกายในตอนเช้า 30 นาที''',Channel_access_token)
        else:
          ReplyMessage(Reply_token,'''ขอโทษครับ ไม่เข้าใจคำถามครับ''',Channel_access_token)
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
  