# ================= Import libraries ====================

import gdown
import uvicorn
import numpy as np

from fastapi import FastAPI
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical

from tokenization import *

# ================= Set paths and parameters ====================

output='model.h5'

device_name = '/device:GPU:0'

m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'

url = "https://drive.google.com/uc?id={}".format("1GDYhTNHhat-qa_6fm5OqWTfTUclHFo6Z")

# ================= User Defined Functions====================

def load_model_for_test(max_len,bert_layer):
  with tf.device(device_name):
    model = build_model(bert_layer, max_len=max_len)
    model.load_weights("model.h5")
  return model  


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512):
  
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    clf_output = sequence_output[:, 0, :]
    
    lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(32, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(2, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def do_test(text,tokenizer,bert_layer):

  suicide=0
  max_len=250

  with tf.device(device_name):
    encoded_text =  bert_encode(text, tokenizer, max_len=max_len)

  model = load_model_for_test(max_len,bert_layer)

  with tf.device(device_name):
    outcome = model.predict(encoded_text)

  if outcome[0][1]>0.5:
    suicide = 1

  if suicide==1:
    result="Suicidal."
  else:
    result="no suicidal."

  return result
