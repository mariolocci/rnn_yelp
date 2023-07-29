#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:44:23 2017

@author: mario
"""
import json
import csv
import nltk
import sqlalchemy
import goslate
import time
from random import randint
import htmllib as hlib
from googleapiclient.discovery import build

def readYelp(source):
    with open(source) as f:
        # for each line in the json file
        for line in f:
            x=json.loads(line)
            text= x['text']
            sent_text = nltk.sent_tokenize(text)
            print sent_text
            for s in sent_text:
                tokenized_text = nltk.word_tokenize(s)
                print tokenized_text 
                print "\n"
            print '--------------'
    # close the reader
    f.close()
def readYelpFromCsv(source):
    with open(source) as f:
        result=[]
        for line in f:
            result.append(line)
    return result
            
def readYelpSent(source,limit):
    with open(source) as f:
        # for each line in the json file
        result=[]
        i=0
        for line in f:
            x=json.loads(line)
            text= x['text']
            sent_text = nltk.sent_tokenize(text)
            for s in sent_text:
                result.append(s)
            if(i>limit):
                break
            else:
                i=i+1
                #print "\n"
           
    # close the reader
    f.close()
    return result
def translate(frase):
  f=[]
  f.append(frase)
  # Build a service object for interacting with the API. Visit
  # the Google APIs Console <http://code.google.com/apis/console>
  # to get an API key for your own application.
  service = build('translate', 'v2',
            developerKey='AIzaSyC9UU3hzlh-bgHJatCAr_-UZr4uuZWQKgs')
  translated=service.translations().list(
      source='en',
      target='it',
      q=f,
      format='text'
    ).execute()
  #print(translated)
  di=translated['translations']
  return di[0]['translatedText']

def saveYelp(listofsentence,filename,tras=False):
        #gs=goslate.Goslate()
        with open(filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='|')
            for s in listofsentence:
                row=[]
                row.append(s)
                row.append('FOOD')
                row.append('+')
                #italian=gs.translate(s,'it')
                #row.append(italian)
                #t=randint(1,9)
                #time.sleep(t)
                if tras:
                    try:
                        italian=translate(s)
                        row.append(italian)
                    except:
                        row.append('')
                print row
                try:
                    spamwriter.writerow(row)
                except:
                    print "skipped "+s
def fetchFromDb():
    engine = sqlalchemy.create_engine('mysql://root:manu123@semtech.crs4.it/yelp') # connect to server
    connection=engine.connect()
    result=connection.execute("SELECT text FROM review,category WHERE (review.business_id=category.business_id) AND category.category_name='Restaurants'  limit 100") #create db
    out=[]
    for r in result:
        aux=r['text']
        out.append( aux)
    connection.close()
    return out

def sentenceTokenize(listofreview):
    result=[]
    for text in listofreview:
        try:
            sent_text = nltk.sent_tokenize(text)
            for s in sent_text:
                result.append(s)
        except UnicodeDecodeError:
            pass
        result.append("\n")
    return result
    
if __name__ =='__main__':
    print ("hello")
    result=fetchFromDb()

#result=readYelpSent("/home/mario/tensorflow3/Yelp-Challenge-Dataset/Raw Data/yelp_dataset_challenge/yelp_academic_dataset_review.json",100)
    tokenized=sentenceTokenize(result)

    saveYelp(tokenized,"/home/mario/tensorflow3/Yelp-Challenge-Dataset/Raw Data/yelp_dataset_challenge/yelp.csv")
