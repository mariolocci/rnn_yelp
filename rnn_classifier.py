#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:38:40 2017

@author: mario
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import yelpreader
import cPickle

flags = tf.app.flags
FLAGS = flags.FLAGS
try:
    flags.DEFINE_string("save_path", "/tmp/model/", "Directory to write the model.")

    flags.DEFINE_string(
    "train_data", "/home/mario/tensorflow3/Yelp-Challenge-Dataset/Raw Data/yelp_dataset_challenge/yelp.csv",
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")

    flags.DEFINE_integer("embedding_size", 50, "The embedding dimension size.")
    flags.DEFINE_integer(
    "epochs", 1000,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
    flags.DEFINE_float("learning_rate", 0.0005, "Initial learning rate.")

    flags.DEFINE_integer("batch_size", 250,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")


    flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")
    flags.DEFINE_integer("min_word_frequency", 1,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
    flags.DEFINE_integer("rnn_size", 10,
                     "the rnn size")
    flags.DEFINE_integer("max_sequence_length", 25,
                     "maximum number of words in a sentence")
    flags.DEFINE_boolean(
    "restore", False,
     "use the save_path for restoring the model")

   
except Exception:
    print ('ArgumentError')
def shuffle(text_processed,text_data_target):
    text_processed=np.array(text_processed)
    text_data_target=np.array(text_data_target)
    shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
    x_shuffled = text_processed[shuffled_ix]
    y_shuffled = text_data_target[shuffled_ix]
# Split train/test se
    ix_cutoff = int(len(y_shuffled)*0.80)
    x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
    y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
    #vocab_size = len(vocab_processor.vocabulary_)
    #print("Vocabulary Size: {:d}".format(vocab_size))
    print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))
    return [x_train,y_train,x_test,y_test]

class Options(object):
  """Options used by our word2vec model."""

  def __init__(self,embedding_size=50,learning_rate= 0.0005,batch_size=250,min_word_frequency=5,rnn_size=10,max_sequence_length=25):
    # Model options.

    if FLAGS:
        # Embedding dimension.
        self.embedding_size = FLAGS.embedding_size

        # Training options.

        # The training text file.
        self.train_data = FLAGS.train_data

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs= FLAGS.epochs


        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size

   

   
        # Where to write out summaries.
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        #The minimum number of word occurrences 
        self.min_word_frequency= FLAGS.min_word_frequency

        self.rnn_size=FLAGS.rnn_size
    
        self.max_sequence_length=FLAGS.max_sequence_length
  def __str__(self):
       result="embedding size:"+str(self.embedding_size)+"\n"
       result=result+"train data:"+str(self.train_data)+"\n"
       result=result+"learning rate:"+str(self.learning_rate)+"\n"
       result=result+"epochs:"+str(self.epochs)+"\n"
       result=result+"batch size:"+str(self.batch_size)+"\n"
       result=result+"min word frequency:"+str(self.min_word_frequency)+"\n"
       result=result+"rnn size:"+str(self.rnn_size)+"\n"
       result=result+"max sequence length"+str(self.max_sequence_length)+"\n"
       return result
class RnnClassifier(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self.options = options
    self.session = session
   
  
 

  def embeddedLayer(self,x_data,vocab_size):
      self.embedding_mat = tf.Variable(tf.random_uniform([vocab_size, self.options.embedding_size], -3.0, 3.0))
      embedding_output = tf.nn.embedding_lookup(self.embedding_mat, x_data)
      return embedding_output
  def create_graph_rnn(self,vocab_size,scope=None):
      # Create placeholders
      self.x_data = tf.placeholder(tf.int32, [None, self.options.max_sequence_length])
     
      self.dropout_keep_prob = tf.placeholder(tf.float32)
      #process vocabulary
      
     
      embedding_output=self.embeddedLayer(self.x_data,vocab_size)
      self.weight = tf.Variable(tf.truncated_normal([self.options.rnn_size, 2], stddev=0.1))
      self.bias = tf.Variable(tf.constant(0.1, shape=[2]))
      if tf.__version__[0]>='1':
          cell=tf.contrib.rnn.LSTMCell(num_units = self.options.rnn_size)
      else:
          cell = tf.nn.rnn_cell.LSTMCell(num_units = self.options.rnn_size)
      print(scope)
      output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32,scope=scope)
      output = tf.nn.dropout(output, self.dropout_keep_prob)
      # Get output of RNN sequence
      output = tf.transpose(output, [1, 0, 2])
      last = tf.gather(output, int(output.get_shape()[0]) - 1)
      self.logits_out = tf.nn.softmax(tf.matmul(last, self.weight) + self.bias)
      return self.logits_out
  def create_losses(self,vocab_size,saver=True,scope=None):
      self.y_output = tf.placeholder(tf.int32, [None])
      logits_out=self.create_graph_rnn(vocab_size,scope)
      self.argm=tf.argmax(self.logits_out,1)

      # Loss function
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_out, labels=self.y_output) # logits=float32, labels=int32
      self.loss = tf.reduce_mean(losses)
      self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(self.y_output, tf.int64)), tf.float32))
      self.optimizer = tf.train.AdamOptimizer(self.options.learning_rate)
      self.train_step = self.optimizer.minimize(self.loss)
      if saver:
       self.saver=tf.train.Saver()


 
  def save(self,save_path):
      self.saver.save(self.session, os.path.join(save_path, "model.ckpt"))
  def restore(self,save_path):
       self.saver.restore(self.session,os.path.join(save_path, "model.ckpt"))
      
  def train(self,vocab_processor,text_data_train,text_data_target,retrain=False, logs=True):
       
       text_processed=self.mapTextToInt(vocab_processor,text_data_train)
       [x_train,y_train,x_test,y_test]=shuffle(text_processed,text_data_target)
       sess=self.session
       if not retrain:
         init = tf.global_variables_initializer()
         sess.run(init)
       
       train_loss = []
       test_loss = []
       train_accuracy = []
       test_accuracy = []
       
       for epoch in range(self.options.epochs):
           # Shuffle training data
           shuffled_ix = np.random.permutation(np.arange(len(x_train)))
           x_train = x_train[shuffled_ix]
           y_train = y_train[shuffled_ix]
           num_batches = int(len(x_train)/self.options.batch_size) + 1
           # TO DO CALCULATE GENERATIONS ExACTLY
           for i in range(num_batches):
               # Select train data
               min_ix = i * self.options.batch_size
               max_ix = np.min([len(x_train), ((i+1) * self.options.batch_size)])
               x_train_batch = x_train[min_ix:max_ix]
               y_train_batch = y_train[min_ix:max_ix]
        
                # Run train step
               train_dict = {self.x_data: x_train_batch, self.y_output: y_train_batch, self.dropout_keep_prob:1}
               sess.run(self.train_step, feed_dict=train_dict)
        
               # Run loss and accuracy for training
               temp_train_loss, temp_train_acc = sess.run([self.loss, self.accuracy], feed_dict=train_dict)
               train_loss.append(temp_train_loss)
               train_accuracy.append(temp_train_acc)
               if logs: print('Epoch: {}, Train Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1,  temp_train_loss, temp_train_acc))
               # Run Eval Step
               test_dict = {self.x_data: x_test, self.y_output: y_test, self.dropout_keep_prob:1.0}
               temp_test_loss, temp_test_acc = sess.run([self.loss, self.accuracy], feed_dict=test_dict)
               test_loss.append(temp_test_loss)
               test_accuracy.append(temp_test_acc)
               
               if logs: print('Epoch: {}, Test Loss: {:.2}, Train Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
              
       #save_path = saver.save(sess, "/home/mario/tensorflow3/yelp_project/model.ckpt")
       #print("Model saved in file: %s" % save_path)
      
  def evaluate(self,vocab_processor,text_data):
     # print(text_data)
      text_processed=self.mapTextToInt(vocab_processor,text_data)
      #print(text_processed)
     # p=self.logits_out.eval(feed_dict={self.x_data:text_processed,self.dropout_keep_prob:1.0},session=self.session)
     # print("lllllllllllllllll--",str(p))
      #result=self.session.run(self.logits_out, feed_dict={self.x_data:text_processed,self.dropout_keep_prob:1.0})
      result =self.session.run(self.argm,feed_dict={self.x_data:text_processed,self.dropout_keep_prob:1.0})
      return result
  def predict(self,vocab_processor,text_data):
      text_processed=self.mapTextToInt(vocab_processor,text_data)
      result =self.session.run(self.logits_out,feed_dict={self.x_data:text_processed,self.dropout_keep_prob:1.0})
     
      return result[0]
  def mapTextToInt(self,vocab_processor, text_data_train):
      text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))
      text_processed = np.array(text_processed)
      return text_processed
  
    
    
def create_vocabulary(text_data,max_sequence_length,min_word_frequency,save_path):
       vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                     min_frequency=min_word_frequency)
      
       vocab_processor.fit_transform(text_data)
       vocab_processor.save( os.path.join(save_path, "vocab.txt"))
     
       return vocab_processor
def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns) 
  
def labelsConverter(data_target, label):
    result=[1 if label in x else 0  for x in data_target ]
    return result

def getTextAndLabel(lines):
    text_data=[]
    text_labels=[]
    text_sent=[]
    for x in lines:
        splitted=x.split('|')
        if len( splitted)==3 and len(splitted[0])>5 :
            text_data.append(splitted[0])
            text_labels.append(splitted[1])
            text_sent.append(splitted[2])
    return [text_data,text_labels,text_sent]    
def main():
  """Train a  model."""
  #if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
   # print("--train_data --eval_data and --save_path must be specified.")
    #sys.exit(1)
  opts = Options()
  opts.epochs=800
  if not FLAGS.restore:
      with tf.Graph().as_default(), tf.Session() as session:
          with tf.device("/cpu:0"):
              model = RnnClassifier(opts, session)
     
      #source="/home/mario/tensorflow3/Yelp-Challenge-Dataset/Raw Data/yelp_dataset_challenge/yelp.csv"
              #lines=yelpreader.readYelpFromCsv("/home/mario/tensorflow3/Yelp-Challenge-Dataset/Raw Data/yelp_dataset_challenge/yelp.csv")
              
              lines=yelpreader.readYelpFromCsv("/home/mario/food.csv")
              [text_data_train,text_data_target,text_sent]=getTextAndLabel(lines)
              text_data_target=labelsConverter(text_data_target,'FOOD')              
              vocab_processor = create_vocabulary(text_data_train,opts.max_sequence_length,opts.min_word_frequency,opts.save_path)
              vocab_size = len(vocab_processor.vocabulary_)    
              model.create_losses(vocab_size)
              model.train(vocab_processor,text_data_train,text_data_target)  # Eval analogies.
    # Perform a final save.
              #saver=tf.train.Saver()
              model.save(opts.save_path)
  if FLAGS.restore:
         with tf.Graph().as_default(), tf.Session() as session:
             with tf.device("/cpu:0"):
               
                model = RnnClassifier(opts, session)
                print(' restoring session')
                vocab_processor=cPickle.loads(open(os.path.join(opts.save_path, "vocab.txt")).read())
                vocab_size = len(vocab_processor.vocabulary_)
                model.create_losses(vocab_size)
                saver=tf.train.Saver()
                saver.restore(session,os.path.join(opts.save_path, "model.ckpt"))
               
                text_data=['It s great for dinner, but a little steep for a work day lunch.']
                print(text_data)
                test=model.evaluate(vocab_processor,text_data)
                print (test)
  if FLAGS.interactive:
      # E.g.,
      # [0]: model.analogy(b'france', b'paris', b'russia')
      # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
       print('hell')
       _start_shell(locals())
if __name__ =='__main__':
    print ("hello")
  #tf.app.run()
    main()