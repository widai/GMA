import os
import numpy as np
from flask import Flask, render_template, redirect, url_for, request
import regex 
import pandas as pd
import nltk
from nltk import FreqDist
import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

def clean(sentence_token):
  inverse_words=['no','not','never']
  stop_list = STOPWORDS
  stopwords = [i for i in stop_list]
  for i in inverse_words:
    stopwords.remove(i)
  def remove_stopwords(text):
    clean_text = []
    for word in text.split():
        if word not in stopwords and (len(word) > 2):
            clean_text.append(word)
    return ' '.join(clean_text)
  sentence_token = sentence_token.lower()
  split_tokens = regex.split(r'however|but|yet|compared|though|although|either+', sentence_token)
  past = []
  for split_token in split_tokens:
    token_words = pos_tag(word_tokenize(split_token))
    for token_word in token_words:
      if token_word[1] == 'VBD' or token_word[1] == 'VBN':
        past.append(split_token)
  present_tense_tokens = list(set(split_tokens)^set(past))
  present_tense_tokens = list(set(split_tokens))
  for i in range(0,len(present_tense_tokens)):
    present_tense_tokens[i] = regex.sub(r'[^\w\s]', ' ', present_tense_tokens[i])
    present_tense_tokens[i] = remove_stopwords(present_tense_tokens[i])
    #present_tense_tokens[i] = TextBlob(present_tense_tokens[i])
    #present_tense_tokens[i] = str(present_tense_tokens[i].correct())
  return present_tense_tokens

def sentiment_scores(Text):
  sid_obj = SentimentIntensityAnalyzer()
  sentiment_dict = sid_obj.polarity_scores(Text)
  lst = list(sentiment_dict.items())
  return lst

def average(i,key,dict_,sent_final):
  comp = list()
  for d in range(0,len(dict_.get(key))):
    comp_ = sent_final[dict_.get(key)[d]][3][1]
    comp.append(comp_)
  compound = sum(comp)/len(dict_.get(key))
  if compound >= 0.05:
    return 'Positive'
  elif compound >= -0.05 and compound <=0.05:
    return 'Neutral'
  elif compound <= -0.05:
    return 'Negative'
  
def topics(text):
  sentence_tokens = sent = regex.split(r',|\.', text)
  cleaned_review = []
  for sentence_token in sentence_tokens:
    cleaned_tokens = clean(sentence_token)
    for cleaned_token in cleaned_tokens:
      if cleaned_token != '':
        cleaned_review.append(cleaned_token)
  token_count_list = []

  topics = pd.read_csv("C:\\Users\\User\\SentimentApp\\Topic Modelling Terms.csv")
  camera = list(topics['Camera'])
  price = list(topics['Price'])
  screen = list(topics['Screen'])
  battery = list(topics['Battery'])
  performance = list(topics['Performance'])
  software = list(topics['Software'])
  service = list(topics['Service'])
  topic_mapping = {"Camera":camera, "Price":price, "Screen":screen, "Battery":battery, "Performance":performance, "Software":software, "Service":service}

  def fetchTopic(x,topic_mapping):
    for key, values in topic_mapping.items():
      for value in values:
        if x==value:
          return key
      return 0
  feedback_words = {'trouble','improve','could'}
  def matchFeedbackWords(x):
    if re.search('trouble|improv|could', x):
        return 1
    else:
        return 0
  final = []
  for i in range(0,len(cleaned_review)):
    word_lst = cleaned_review[i].split()
    unit = []
    found_feedback = 0
    for k in range(0,len(word_lst)):
      unit.append(fetchTopic(word_lst[k],topic_mapping))
    final.append(unit)
  
  newdf4=pd.DataFrame()
  newdf4.at[0,'Review']= text
  newdf4['Topic: Camera']='0'
  newdf4['Topic: Price']='0'
  newdf4['Topic: Screen and Display']='0'
  newdf4['Topic: Battery']='0'
  newdf4['Topic: Processor']='0'
  newdf4['Topic: Software']='0'
  newdf4['Topic: Service']='0'

  for j in range(0,len(final)):
    for k in range(0,len(final[j])):
      if final[j][k] == 'Camera':
        newdf4.at[0,'Topic: Camera']='1'
      elif final[j][k] == 'Battery':
        newdf4.at[0,'Topic: Battery']='1'
      elif final[j][k] == 'Screen':
        newdf4.at[0,'Topic: Screen and Display']='1'
      elif final[j][k] == 'Price':
        newdf4.at[0,'Topic: Price']='1'
      elif final[j][k] == 'Performance':
        newdf4.at[0,'Topic: Processor']='1'
      elif final[j][k] == 'Software':
        newdf4.at[0,'Topic: Software']='1'
      elif final[j][k] == 'Service':
        newdf4.at[0,'Topic: Service']='1'
  sent_final = []
  for j in range(0,len(cleaned_review)):
    sent_final.append(sentiment_scores(cleaned_review[j]))
  ## CUSTOM SENTIMENTS
  camera_positive=[]
  camera_negative=[]
  price_positive=['value']
  price_negative=['expensive']
  screen_positive=['clear']
  screen_negative=['blurry']
  battery_positive=['long']
  battery_negative=[]
  performance_positive=['smooth','fast']
  performance_negative=['lag']
  software_positive=[]
  software_negative=[]
  service_positive=[]
  service_negative=[]

  custom_sentiment_mapping = {"Camera":{"Positive":camera_positive,"Negative":camera_negative}, 0:[], "Price":{"Positive":price_positive,"Negative":price_negative}, "Screen":{"Positive":screen_positive,"Negative":screen_negative}, "Battery":{"Positive":battery_positive,"Negative":battery_negative}, "Performance":{"Positive":performance_positive,"Negative":performance_negative}, "Software":{"Positive":software_positive,"Negative":software_negative}, "Service":{"Positive":service_positive,"Negative":service_negative}}
  sent_df = pd.DataFrame()
  sent_df.at[0,'Review'] = text
  sent_df['SENT: POSITIVITY (Camera)']= 0
  sent_df['SENT: NEGATIVITY (Camera)']=0
  sent_df['SENT: NEUTRAL (Camera)']=0
  sent_df['SENT: POSITIVITY (Price)']=0
  sent_df['SENT: NEGATIVITY (Price)']=0
  sent_df['SENT: NEUTRAL (Price)']=0
  sent_df['SENT: POSITIVITY (Screen and Display)']=0
  sent_df['SENT: NEGATIVITY (Screen and Display)']=0
  sent_df['SENT: NEUTRAL (Screen and Display)']=0
  sent_df['SENT: POSITIVITY (Battery)']=0
  sent_df['SENT: NEGATIVITY (Battery)']=0
  sent_df['SENT: NEUTRAL (Battery)']=0
  sent_df['SENT: POSITIVITY (Processor)']=0
  sent_df['SENT: NEGATIVITY (Processor)']=0
  sent_df['SENT: NEUTRAL (Processor)']=0
  sent_df['SENT: POSITIVITY (Software)']=0
  sent_df['SENT: NEGATIVITY (Software)']=0
  sent_df['SENT: NEUTRAL (Software)']=0
  sent_df['SENT: POSITIVITY (Service)']=0
  sent_df['SENT: NEGATIVITY (Service)']=0
  sent_df['SENT: NEUTRAL (Service)']=0

  top_dict = {"Camera":[], 0:[], "Price":[], "Screen":[], "Battery":[], "Performance":[], "Software":[], "Service":[]}
  for a in range(0,len(final)):
    for k in range(0,len(final[a])):
      top_dict[final[a][k]].append(a)
    lst = list(top_dict.values())
    flag = []
    for r in range(0,len(lst)):
      flag1 = list(set(lst[r]))
      flag.append(flag1)
    top_dict = {"Camera": flag[0], 0: flag[1], "Price": flag[2], "Screen": flag[3], "Battery": flag[4], "Performance": flag[5], "Software": flag[6], "Service": flag[7]}
    for key,values in top_dict.items():
      if top_dict.get(key) != []:
        if key == 'Price':
          sentiment = average(0,key, top_dict,sent_final)
          if sentiment == 'Neutral':
            sent_df.at[0,'SENT: NEUTRAL (Price)'] = 1
          elif sentiment == 'Positive':
            sent_df.at[0,'SENT: POSITIVITY (Price)'] = 1
          elif sentiment == 'Negative':
            sent_df.at[0,'SENT: NEGATIVITY (Price)'] = 1
        elif key == 'Battery':
          sentiment = average(0,key, top_dict,sent_final)
          if sentiment == 'Neutral':
            sent_df.at[0,'SENT: NEUTRAL (Battery)'] = 1
          elif sentiment == 'Positive':
            sent_df.at[0,'SENT: POSITIVITY (Battery)'] = 1
          elif sentiment == 'Negative':
            sent_df.at[0,'SENT: NEGATIVITY (Battery)'] = 1
        elif key == 'Camera':
          sentiment = average(0,key, top_dict,sent_final)
          if sentiment == 'Neutral':
            sent_df.at[0,'SENT: NEUTRAL (Camera)'] = 1
          elif sentiment == 'Positive':
            sent_df.at[0,'SENT: POSITIVITY (Camera)'] = 1
          elif sentiment == 'Negative':
            sent_df.at[0,'SENT: NEGATIVITY (Camera)'] = 1
        elif key == 'Screen':
          sentiment = average(0,key, top_dict,sent_final)
          if sentiment == 'Neutral':
            sent_df.at[0,'SENT: NEUTRAL (Screen and Display)'] = 1
          elif sentiment == 'Positive':
            sent_df.at[0,'SENT: POSITIVITY (Screen and Display)'] = 1
          elif sentiment == 'Negative':
            sent_df.at[0,'SENT: NEGATIVITY (Screen and Display)'] = 1
        elif key == 'Performance':
          sentiment = average(0,key, top_dict,sent_final)
          if sentiment == 'Neutral':
            sent_df.at[0,'SENT: NEUTRAL (Processor)'] = 1
          elif sentiment == 'Positive':
            sent_df.at[0,'SENT: POSITIVITY (Processor)'] = 1
          elif sentiment == 'Negative':
            sent_df.at[0,'SENT: NEGATIVITY (Processor)'] = 1
        elif key == 'Software':
          sentiment = average(0,key, top_dict,sent_final)
          if sentiment == 'Neutral':
            sent_df.at[0,'SENT: NEUTRAL (Software)'] = 1
          elif sentiment == 'Positive':
            sent_df.at[0,'SENT: POSITIVITY (Software)'] = 1
          elif sentiment == 'Negative':
            sent_df.at[0,'SENT: NEGATIVITY (Software)'] = 1
        elif key == 'Service':
          sentiment = average(0,key, top_dict,sent_final)
          if sentiment == 'Neutral':
            sent_df.at[0,'SENT: NEUTRAL (Service)'] = 1
          elif sentiment == 'Positive':
            sent_df.at[0,'SENT: POSITIVITY (Service)'] = 1
          elif sentiment == 'Negative':
            sent_df.at[0,'SENT: NEGATIVITY (Service)'] = 1
  Output_df = pd.merge(newdf4, sent_df, on="Review")

  return Output_df


@app.route('/')
def home():
    return render_template('sentimentUI.html')



@app.route('/', methods=("POST", "GET"))
def html_table():
    text = request.form['text_box']
    Output = topics(text)
    return render_template('sentimentUI.html',  tables=[Output.to_html(classes='data', header="true")])

if __name__=='__main__':
    app.run()