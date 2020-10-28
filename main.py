
# Stage 1: Importing dependencies
import tensorflow as tf

import tensorflow_hub as hub

from official.nlp.bert.tokenization import FullTokenizer
from official.nlp.bert.input_pipeline import create_squad_dataset
from official.nlp.data.squad_lib import generate_tf_record_from_json_file

from official.nlp import optimization

from official.nlp.data.squad_lib import read_squad_examples
from official.nlp.data.squad_lib import FeatureWriter
from official.nlp.data.squad_lib import convert_examples_to_features
from official.nlp.data.squad_lib import write_predictions

import numpy as np
import math
import random
import time
import json
import collections
import os

from google.colab import drive


# Data preprocessing
drive.mount("/content/drive")
input_meta_data = generate_tf_record_from_json_file(
    "/content/drive/My Drive/BERT/data/squad/train-v1.1.json",
    "/content/drive/My Drive/BERT/data/squad/vocab.txt",
    "/content/drive/My Drive/BERT/data/squad/train-v1.1.tf_record")
    
input_meta_data = generate_tf_record_from_json_file(
    "/content/drive/My Drive/BERT/data/squad/train-v1.1.json",
    "/content/drive/My Drive/BERT/data/squad/vocab.txt",
    "/content/drive/My Drive/BERT/data/squad/train-v1.1.tf_record")

input_meta_data = generate_tf_record_from_json_file(
    "/content/drive/My Drive/BERT/data/squad/train-v1.1.json",
    "/content/drive/My Drive/BERT/data/squad/vocab.txt",
    "/content/drive/My Drive/BERT/data/squad/train-v1.1.tf_record")

BATCH_SIZE = 4

train_dataset = create_squad_dataset(
    "/content/drive/My Drive/BERT/data/squad/train-v1.1.tf_record",
    input_meta_data['max_seq_length'], # 384
    BATCH_SIZE,
    is_training=True)


# Stage 3: Model building

class BertSquadLayer(tf.keras.layers.Layer):

  def __init__(self):
    super(BertSquadLayer, self).__init__()
    self.final_dense = tf.keras.layers.Dense(
        units=2,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

  def call(self, inputs):
    logits = self.final_dense(inputs) # (batch_size, seq_len, 2)

    logits = tf.transpose(logits, [2, 0, 1]) # (2, batch_size, seq_len)
    unstacked_logits = tf.unstack(logits, axis=0) # [(batch_size, seq_len), (batch_size, seq_len)] 
    return unstacked_logits[0], unstacked_logits[1]


class BERTSquad(tf.keras.Model):
    
    def __init__(self,
                 name="bert_squad"):
        super(BERTSquad, self).__init__(name=name)
        
        self.bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            trainable=True)
        
        self.squad_layer = BertSquadLayer()
    
    def apply_bert(self, inputs):
#        _ , sequence_output = self.bert_layer([inputs["input_ids"],
#                                               inputs["input_mask"],
#                                               inputs["segment_ids"]])
        
        # New names for the 3 different elements of the inputs, since an update
        # in tf_models_officials. Doesn't change anything for any other BERT
        # usage.
        _ , sequence_output = self.bert_layer([inputs["input_word_ids"],
                                               inputs["input_mask"],
                                               inputs["input_type_ids"]])
        return sequence_output

    def call(self, inputs):
        seq_output = self.apply_bert(inputs)

        start_logits, end_logits = self.squad_layer(seq_output)
        
        return start_logits, end_logits


# Stage 4: Training

TRAIN_DATA_SIZE = 88641
NB_BATCHES_TRAIN = 2000
BATCH_SIZE = 4
NB_EPOCHS = 2
INIT_LR = 5e-5
WARMUP_STEPS = int(NB_BATCHES_TRAIN * 0.1)

train_dataset_light = train_dataset.take(NB_BATCHES_TRAIN)

bert_squad = BERTSquad()

optimizer = optimization.create_optimizer(
    init_lr=INIT_LR,
    num_train_steps=NB_BATCHES_TRAIN,
    num_warmup_steps=WARMUP_STEPS)


def squad_loss_fn(labels, model_outputs):
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    start_logits, end_logits = model_outputs

    start_loss = tf.keras.backend.sparse_categorical_crossentropy(
        start_positions, start_logits, from_logits=True)
    end_loss = tf.keras.backend.sparse_categorical_crossentropy(
        end_positions, end_logits, from_logits=True) enviat la sol·licitud
Rebràs un correu electrònic per informar-te si la t
    
    total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2

    return total_loss

train_loss = tf.keras.metrics.Mean(name="train_loss")


bert_squad.compile(optimizer,
                   squad_loss_fn)

checkpoint_path = "./drive/My Drive/BERT/ckpt_bert_squad/"

ckpt = tf.train.Checkpoint(bert_squad=bert_squad)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")