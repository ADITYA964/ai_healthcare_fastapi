# ================= Import libraries ====================

import gdown
import uvicorn
import numpy as np
from fastapi import FastAPI

import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical

from model_utils import *
from tokenization import *

# ================= Set paths and parameters ====================

output='model.h5'

device_name = '/device:GPU:0'

m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'

url = "https://drive.google.com/uc?id={}".format("1GDYhTNHhat-qa_6fm5OqWTfTUclHFo6Z")


# ===================== Configure BERT  ======================
with tf.device(device_name):

  bert_layer = hub.KerasLayer(m_url, trainable=False)

  vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

  do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

  tokenizer = FullTokenizer(vocab_file, do_lower_case)

# ========= Download weights of trained BERT  ===========

gdown.download(url, output, quiet=False)

# ================= Execute API  ====================
app = FastAPI()

@app.get('/')
def get_text(text: str):

  with tf.device(device_name):
    result = do_test([text],tokenizer,bert_layer)

  return {'input text':text,
          'suicide or not': "The Text Contains References to " + result}
