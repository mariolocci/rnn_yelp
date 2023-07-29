#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:24:21 2017

@author: mario
"""
import numpy as np
import tensorflow as tf
import yelpreader
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph
sess = tf.Session()

def mapTextToInt(vocab_processor, text_data_train):
  
     text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))
     text_processed = np.array(text_processed)
     return text_processed

def embeddedLayer(x_data,embedding_mat):
   
   # vocab_size = len(vocab_processor.vocabulary_)
    #x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
# Create embedding
    #embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
    return embedding_output

# Define the RNN cell
def rnnNet(rnn_size,embedding_output,dropout_keep_prob,weight,bias):

#tensorflow change >= 1.0, rnn is put into tensorflow.contrib directory. Prior version not test.
    if tf.__version__[0]>='1':
        cell=tf.contrib.rnn.BasicRNNCell(num_units = rnn_size)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)
    output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
    output = tf.nn.dropout(output, dropout_keep_prob)
# Get output of RNN sequence
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
  
    logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)
    return logits_out


# Shuffle data
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


def rnnClassify(text,label):
    text_processed=mapTextToInt(vocab_processor,text)
    #vocab_size = len(vocab_processor.vocabulary_)
    
    embedding_output= embeddedLayer(text_processed,embedding_mat)
    rnnNet(rnn_size,embedding_output,dropout_keep_prob,weight,bias)
    
    result=[]
    for x in embedding_output:
        if x==1:
            result.append(label)
        else:
            result.append('no_'+label)
    return result
# Set RNN parameters
epochs = 2
batch_size = 250
max_sequence_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 1
learning_rate = 0.0005


source="/home/mario/tensorflow3/Yelp-Challenge-Dataset/Raw Data/yelp_dataset_challenge/yelp.csv"
lines=yelpreader.readYelpFromCsv(source)
# tre colums expected text,labels,sentiment label
[text_data_train,text_data_target,text_sent]=getTextAndLabel(lines)
text_data_target=labelsConverter(text_data_target,'FOOD')

# Create placeholders
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])
dropout_keep_prob = tf.placeholder(tf.float32)

#process vocabulary
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                     min_frequency=min_word_frequency)
text_processed=mapTextToInt(vocab_processor,text_data_train)
vocab_size = len(vocab_processor.vocabulary_)
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

embedding_output=embeddedLayer(x_data,embedding_mat)


weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out=rnnNet(rnn_size,embedding_output,dropout_keep_prob,weight,bias)

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output) # logits=float32, labels=int32
loss = tf.reduce_mean(losses)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)
[x_train,y_train,x_test,y_test]=shuffle(text_processed,text_data_target)

init = tf.global_variables_initializer()
sess.run(init)
saver=tf.train.Saver()
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []



for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train)/batch_size) + 1
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        
        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)
        
    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    
    # Run Eval Step
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
save_path = saver.save(sess, "/home/mario/tensorflow3/yelp_project/model.ckpt")
print("Model saved in file: %s" % save_path)
#output=rnnClassify(x_data,'FOOD')
#xx_text=['down of the death']





