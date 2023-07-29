#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:46:20 2017

@author: mario
"""
import json
#import csv
import nltk
import sqlalchemy

#from random import randint
from googletranslate import translate
#import htmllib as hlib
#from googleapiclient.discovery import build
from pymongo import MongoClient
#from flask import jsonify

class Review():
    def __init__ (self,review_id,text,business_id,user_id,useful_votes):
        self.text=text
        self.business_id=business_id
        self.user_id=user_id
        self.useful_votes=useful_votes
        self.review_id=review_id
    def setTranslation(self,translation):
        self.translation=translation
    def jsonify(self):
        aux=self.toDict()
        return json.dumps(aux)
    def toDict(self):
        aux={'review_id':self.review_id,'text':self.text,'business_id':self.business_id,'user_id':self.user_id,'useful_votes':self.useful_votes}
        return aux
    
    

def fetchFromDb():
    engine = sqlalchemy.create_engine('mysql://root:manu123@semtech.crs4.it/yelp') # connect to server
    connection=engine.connect()
    result=connection.execute("SELECT text,review.business_id,user_id,useful_votes FROM review,category WHERE (review.business_id=category.business_id) AND category.category_name='Restaurants' limit 10000") #create db
    out=[]
    index=0
    for r in result:
        text=r['text']
        business_id=r['business_id']
        user_id=r['user_id']
        useful_votes=r['useful_votes']
        review_id=None
        try:
         review_id=r['review_id']
        except Exception as e:
            print "skipped review_id" + str(type(e))
        if not review_id:
            review_id=str(index)
        rev=Review(review_id,text,business_id,user_id,useful_votes)
        out.append(rev)
        index=index+1
    connection.close()
    return out

def populateMongo(listOfReview,dbname):
    client = MongoClient('mongodb://localhost:27017/')
    db = client[dbname]
    db.reviews.drop()
    reviews = db.reviews
    i=0
    for rev in listOfReview:
         if i%100==0: print(i)
         try:
             revdict=rev.toDict();
             if i<0:
                 print(str(i))
                 text=revdict['text']
                 tokenized=nltk.sent_tokenize(text)     
                 labelized=[[x,translate(x),'FOOD','NONE','NONE'] for x in tokenized]
                 revdict['sentences']=labelized
             revdict['classified']=False
             reviews.insert_one(revdict)
             i=i+1
         except  Exception as inst:
             print('skipped' + str(type(inst)))
    
def fromJsonToMongo( jsonsource, colltype='reviews'):
     client = MongoClient('mongodb://localhost:27017/')
     db = client['completeYelp']
     if colltype=='reviews':
      db.reviews.drop()
      reviews = db.reviews
      coll=reviews
     if colltype=='business':
      db.business.drop()
      coll=db.business
     if colltype=='users':
      db.users.drop()
      coll=db.users
     with open(jsonsource) as f:
        for line in f:
          try:
            x=json.loads(line)
            coll.insert_one(x)
          except  Exception as inst:
             print('skipped' + str(x) +  str(type(inst)))
             
if __name__ =='__main__':
    #fromJsonToMongo("/home/mario/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json",colltype='reviews')
    #fromJsonToMongo("/home/mario/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_business.json",colltype='business')
    #fromJsonToMongo("/home/mario/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_user.json",colltype='users')
    l=fetchFromDb()
    populateMongo(l,"yelp_2")