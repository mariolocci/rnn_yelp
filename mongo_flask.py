#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:23:24 2017

@author: mario
"""
import logging
from logging.handlers import TimedRotatingFileHandler

from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo
from flask import render_template
import nltk
from googletranslate import translate
from  rnn_classifier import RnnClassifier,Options,getTextAndLabel,labelsConverter,create_vocabulary
import os
import tensorflow as tf
import cPickle
import yelpreader
import csv
import re


app = Flask(__name__,static_url_path = "",static_folder = "static")

app.config['MONGO_DBNAME'] = 'yelp'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/yelp'

mongo = PyMongo(app)
def toDict(s):
    return {#'id' : s['_id'],
            'name':s['name'],
            'text' : s['text'], 
            'business_id':s['business_id'],
            'user_id':s['user_id'],
            'useful_votes':s['useful_votes']}
    
@app.route('/')
def serve_pages():
    return  render_template('index.html')


@app.route('/reviews', methods=['GET'])
def get_all_reviews():
  reviews = mongo.db.reviews
  output = []
  for s in reviews.find():
    output.append(toDict(s))
  return jsonify({'result' : output})

@app.route('/reviews/<name>', methods=['GET'])
def get_one_review(name):
  reviews= mongo.db.reviews
  s = reviews.find_one({'name' : name})
  if s:
    output = s
  else:
    output = "No such name"
  return jsonify({'result' : output})


@app.route('/reviews/next', methods=['GET'])
def get_next():
  app.logger.info(" get next review")
  reviews= mongo.db.reviews
  s = reviews.find_one({'classified':False})   
  #s = reviews.find_one()   
  if s:
    output = dict(s)
    del output['_id']
    if  not 'sentences' in s.keys():
      text=s['text']
      tokenized=nltk.sent_tokenize(text)     
      labelized=[[x,translate(x),'FOOD','NONE','NONE','+'] for x in tokenized]
      output['sentences']=labelized
      reviews.update({'name':s['name']},output)
    if not 'classified' in s.keys():
        output['classified']=False
  else:
    output = "No such name"
  return jsonify({'result' : output})

@app.route('/reviews', methods=['POST'])
def add_review():
  s={}
  reviews = mongo.db.reviews
  s['name'] = request.json['name']
  s['text'] = request.json['text']
  s['business_id'] = request.json['business_id']
  s['user_id'] = request.json['user_id']
  s['useful_votes'] = request.json['useful_votes']
  
  star_id = reviews.insert(s)
  new_star = reviews.find_one({'_id': star_id })

  #output = toDict(new_star)
  return jsonify({'result' : new_star})
@app.route('/reviews/update', methods=['POST'])
def update_review():
  s={}
  reviews = mongo.db.reviews
  s['name'] = request.json['name']
  s['text'] = request.json['text']
  s['sentences'] = request.json['sentences']
  s['classified'] = request.json['classified']
  app.logger.info("updated review"+str(s['name']))
  reviews.update({'name': s['name']},s)
  return jsonify({'result' : s})

@app.route('/classifier/train', methods=['GET'])
def train_classifier():
   
    #with tf.Graph().as_default(), tf.Session() as session:
             #with tf.device("/cpu:0"):
                 #opts=Options()
                 #model = RnnClassifier(opts, session)
                 text_data_train,text_data_target=read_train_data_from_db()
                 text_data=read_text_data_from_db()
                 text_data_target=labelsConverter(text_data_target,'FOOD')
                 app.logger.info("training data"+ str(text_data_train)+ str(text_data_target))
                 vocab_processor = create_vocabulary(text_data,opts.max_sequence_length,opts.min_word_frequency,opts.save_path)
                 #vocab_size = len(vocab_processor.vocabulary_)
                 #model.create_losses(vocab_size)
                 train_model(model,vocab_processor,opts,session,text_data_train,text_data_target,retrain=True)
                 return jsonify({'result' : 'ok'})

@app.route('/classifier/classify', methods=['GET'])
def classify():
      text= request.args.get('text')
      text_data=[]
      text_data.append(text)
      app.logger.info("classify text"+str(text_data))
      array_ones=model.evaluate(vocab_processor,text_data)
      app.logger.info( " logit out :"+str(array_ones))
      result=['FOOD' if x==1 else 'NO_FOOD'  for x in array_ones ]
      app.logger.info( "result"+str(result))
      return jsonify({'result' : result})

@app.route('/reviews/reset', methods=['GET'])
def reset():
    reviews = mongo.db.reviews
    rws=reviews.find({'classified':True})
    for rw in rws:
        rw['classified']=False
        reviews.update({'name': rw['name']},rw)
    return jsonify({'result' : "reset :"+ "str(len(rws))"+ " reviews" })
    
def read_train_data_from_db():
     reviews = mongo.db.reviews
    # model = RnnClassifier(opts, session)
   
     rws=reviews.find({'classified':True})
     text_data_train=[]
     text_data_target=[]
     for rw in rws:
     #app.logger.info(" training " +str(rw['sentences']))
         for x in rw['sentences']:
             if countValidWords(x[0])>2:
               app.logger.info(" training " + str(x)+"\n")
               text_data_train.append(x[0])
               text_data_target.append([x[2],x[3],x[4],x[5]])
      #source="/home/mario/tensorflow3/Yelp-Challenge-Dataset/Raw Data/yelp_dataset_challenge/yelp.csv"
   
     app.logger.info(" training " + str(text_data_train))
     app.logger.info(" data target "+ str(text_data_target))
     
     return text_data_train,text_data_target
 
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)
 
def read_text_data_from_db():
     reviews = mongo.db.reviews
     rws=reviews.find()
     text_data=[]
     i=0
     for rw in rws:
         i=i+1
         text_data.append(rw['text'])
     print(" numero di review" +str(i))
     return text_data

def countValidWords(text):
    tokens=nltk.word_tokenize(text)
    count=0
    for x in tokens:
        if not x in nltk.corpus.stopwords.words('english'):
            count=count+1
    return count
    
def train_model(model,vocab_processor,opts,session,text_data_train,text_data_target,retrain=False):
   
     model.train(vocab_processor,text_data_train,text_data_target,retrain)  # Eval analogies.
    # Perform a final save.
     app.logger.info("trained classifier"+ str( text_data_train))
     model.save(opts.save_path)
def saveCSV(listofsentences,target,filename,tras=False):
        #gs=goslate.Goslate()
        with open(filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='|')
            for s in range(0,len(listofsentences)):
                listofsentences[s]
                x=target[s][0] +","+target[s][1]+","+target[s][2]
                try:
                    spamwriter.writerow( [listofsentences[s],x,"fake"])
                except:
                    print "skipped "+s
def saveSpam(listofsentences,target,filename,tras=False):
        #gs=goslate.Goslate()
        with open(filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\t')
            for s in range(0,len(listofsentences)):
                listofsentences[s]
                x='ham' if 'FOOD' in target[s] else 'spam'
                try:
                    spamwriter.writerow( [x,listofsentences[s]])
                except:
                    print "skipped "+s
if __name__ == '__main__':
    #print(jsonify([{'r':3,'t':8}]))
     with tf.Graph().as_default(), tf.Session() as session:
             with tf.device("/cpu:0"):
                opts=Options()
                opts.epochs=500     
                opts.batch_size=250
                opts.max_sequence_length=25
                opts.min_word_frequency=1
                opts.learning_rate=0.0005
                opts.rnn_size=20
                print( str(opts))
                model = RnnClassifier(opts, session)
                #model_saved_path=os.path.join(opts.save_path, "model.ckpt")
                model_saved_path=opts.save_path
                if os.path.exists(model_saved_path) and not os.listdir(model_saved_path)==[]:
                #if False:
                  with app.app_context():
                    app.logger.info("restoring session")
                    vocab_processor=cPickle.loads(open(os.path.join(opts.save_path, "vocab.txt")).read())
                    vocab_size = len(vocab_processor.vocabulary_)
                    model.create_losses(vocab_size)
                    model.restore(opts.save_path)
                else:
                     with app.app_context():
                       text_data_train,text_data_target=read_train_data_from_db()
                       #text_data_train=[clean_text(x) for x in text_data_train]
                       text_data=read_text_data_from_db()
                       #text_data=[clean_text(x) for x in text_data]
                       #app.logger.info("training data"+ str(text_data_train))
                       #lines=yelpreader.readYelpFromCsv( "/home/mario/tensorflow3/Yelp-Challenge-Dataset/Raw Data/yelp_dataset_challenge/yelp.csv")
                       #[text_data_train,text_data_target,text_sent]=getTextAndLabel(lines)
                       #text_data_target=labelsConverter(text_data_target,'FOOD') 
                       saveCSV(text_data_train,text_data_target,"/home/mario/food.csv")
                       saveSpam(text_data_train,text_data_target,"/home/mario/spam.csv")
                       text_data_target=labelsConverter(text_data_target,'FOOD')
                       if len(text_data_train)>0:
                           vocab_processor = create_vocabulary(text_data,opts.max_sequence_length,opts.min_word_frequency,opts.save_path)
                           vocab_size = len(vocab_processor.vocabulary_)
                           print(" grandezza del vocabolario :",vocab_size)
                           model.create_losses(vocab_size)
                           train_model(model,vocab_processor,opts,session,text_data_train,text_data_target)
                handler = TimedRotatingFileHandler('logs/foo.log', when='midnight', interval=1)
                #handler.suffix = "%Y-%m-%d"
                formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(module)s:%(message)s")
                handler.setFormatter(formatter)
                handler.setLevel(logging.DEBUG)
                app.logger.addHandler(handler)
                app.logger.setLevel(logging.DEBUG)
                
                app.run(debug=True,host='0.0.0.0')
    
    

