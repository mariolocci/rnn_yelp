#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:03:29 2017

@author: mario
"""
from rnn_classifier import RnnClassifier,Options,create_vocabulary
import tensorflow as tf
import os
import yelpreader
import re
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import csv
from sklearn.datasets import make_classification
import nltk
global global_x
global global_y
class MultiLabelRnnClassifier(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session,labels):
    self.options = options
    self.session = session
    self.labels=labels
    self.models={}
    for label in labels:
        self.models[label]=RnnClassifier(options,session)
        
  def create_graph_rnn(self,vocab_size):
     i=0
     for k in self.labels:
        #with tf.device("/cpu:"+str(i)):
         sc=k
         if (k=='+'): sc='PLUS'
         if (k=='-'): sc='MINUS'
         if (k=='='): sc='EQUALS'
         self.models[k].create_losses(vocab_size,saver=False,scope=sc)
         i=i+1
     self.saver=tf.train.Saver()

  def save(self,save_path):
      self.saver.save(self.session, os.path.join(save_path, "model.ckpt"))
  def restore(self,save_path):
       self.saver.restore(self.session,os.path.join(save_path, "model.ckpt"))
       
  def train(self,vocab_processor,text_data_train,text_data_target,retrain=False,logs=False):
         if not retrain:
            init = tf.global_variables_initializer()
            self.session.run(init)
            i=0
         for label in self.labels:
            #with tf.device("/cpu:"+str(i)):
            x_resampled,y_resampled=over_sample(text_data_train,text_data_target[label])
            self.models[label].train(vocab_processor,x_resampled,y_resampled,retrain=True,logs=logs)
            i=i+1
  def evaluate(self,vocab_processor,text_data):
       result=[]
       for label in self.labels:
            #with tf.variable_scope(label):
             #print(label)
             res=self.models[label].evaluate(vocab_processor,text_data)
             if res==1:
                result.append(label)
       return result
  def predict(self,vocab_processor,text_data):
       result={}
       for label in self.labels:
             res=self.models[label].predict(vocab_processor,text_data)
             if (res[1]>res[0]):
                result[label]=round(res[1],7)
             
       return result
def getTextAndLabel(lines, labels):
    text_data=[]
    text_labels={}
    for label in labels:
        text_labels[label]=[]
    for x in lines:
        splitted=x.split('|')
        text_data.append(splitted[0])
        print(splitted)
        raw=splitted[1]
        for label in labels:
                if label in raw:
                    text_labels[label].append(1)
                else:
                    text_labels[label].append(0)     
    return [text_data,text_labels]    
def clean_text(text_string,stopword=True):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    tokens=nltk.word_tokenize(text_string)
    if stopword:
        result=""
        for tk in tokens:
            if not tk in nltk.corpus.stopwords.words('english'):
                result=result+ tk+" "
        return result.strip()
    else:
       return text_string.strip()


def over_sample(text_data_train,text_data_target):
    # global global_x
     #global global_y
     ros = RandomOverSampler()
     aux=[]
     for x in text_data_train:
          aux.append([x])

     x_resampled, y_resampled = ros.fit_sample(aux,text_data_target)
     x_list=[]
     y_list=[]
     for i in range(0,len(x_resampled)):
         x_list.append(x_resampled[i][0])
         y_list.append(y_resampled[i])
    # global_x=x_list
    # global_y=y_list
     return x_list,y_list
def main():
      with tf.Graph().as_default(), tf.Session() as session:
          with tf.device("/cpu:0"):
              #labels=["FOOD","AMBIENCE","STAFF","SERVICE","FOODVARIETY","FOODDESSERT","FOODBEVERAGE","PRICE"]
              labels=["+","-"]
              opts = Options()
              opts.rnn_size=20
              opts.max_sequence_length=50
              opts.epochs=300
              opts.min_word_frequency=0
              opts.learning_rate=0.001
              model = MultiLabelRnnClassifier(opts, session,labels)
              lines=yelpreader.readYelpFromCsv("/home/mario/food.csv")
              
              [text_data_train,text_data_target]=getTextAndLabel(lines,labels)
              global global_x
              global_x=text_data_target
              text_data_train= [clean_text(x,stopword=False) for x in text_data_train]
              vocab_processor = create_vocabulary(text_data_train,opts.max_sequence_length,opts.min_word_frequency,opts.save_path)
              vocab_size = len(vocab_processor.vocabulary_)    
              model.create_graph_rnn(vocab_size)
              
             
              model.train(vocab_processor,text_data_train,text_data_target,logs=True)  # Eval analogies.
              text_data='At least you can get loaded and pretend the dinner you just waited too long for and paid too much for was worth it.'
              text_data=clean_text(text_data,stopword=False)
              print(text_data)
              test=model.evaluate(vocab_processor,[text_data])
              print(test)
              text_data='It is a chain place, and a bit over priced for fast food'
              text_data=clean_text(text_data,stopword=False)
              print (model.evaluate(vocab_processor,[text_data]))
              result=[]
              for text in text_data_train:
                  text_cleaned=clean_text(text,stopword=False)
                  ans=model.evaluate(vocab_processor,[text_cleaned])
                  result.append([text_cleaned,ans])
              filename="/home/mario/result_yelp.csv"   
              with open(filename, 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter='|')
                for row in result:
                   spamwriter.writerow(row)
                   
if __name__ =='__main__':
     main()

        