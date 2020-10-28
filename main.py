
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


# Custom training

for epoch in range(NB_EPOCHS):
    print("Start of epoch {}".format(epoch+1))
    start = time.time()
    
    train_loss.reset_states()
    
    for (batch, (inputs, targets)) in enumerate(train_dataset_light):
        with tf.GradientTape() as tape:
            model_outputs = bert_squad(inputs)
            loss = squad_loss_fn(targets, model_outputs)
        
        gradients = tape.gradient(loss, bert_squad.trainable_variables)
        optimizer.apply_gradients(zip(gradients, bert_squad.trainable_variables))
        print("RAS")
        
        train_loss(loss)
        
        if batch % 50 == 0:
            print("Epoch {} Batch {} Loss {:.4f}".format(
                epoch+1, batch, train_loss.result()))
        
        if batch % 500 == 0:
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} at {}".format(epoch+1,
                                                                ckpt_save_path))
    print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))


# Evaluation

eval_examples = read_squad_examples(
    "/content/drive/My Drive/BERT/data/squad/dev-v1.1.json",
    is_training=False,
    version_2_with_negative=False)

eval_writer = FeatureWriter(
    filename=os.path.join("/content/drive/My Drive/BERT/data/squad/",
                          "eval.tf_record"),
    is_training=False)

my_bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    trainable=False)
vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

def _append_feature(feature, is_padding):
    if not is_padding:
        eval_features.append(feature)
    eval_writer.process_feature(feature)


eval_features = []
dataset_size = convert_examples_to_features(
    examples=eval_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    output_fn=_append_feature,
    batch_size=4)

eval_writer.close()

BATCH_SIZE = 4

eval_dataset = create_squad_dataset(
    "/content/drive/My Drive/BERT/data/squad/eval.tf_record",
    384,#input_meta_data['max_seq_length'],
    BATCH_SIZE,
    is_training=False)

# Making the predictions

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def get_raw_results(predictions):
    for unique_ids, start_logits, end_logits in zip(predictions['unique_ids'],
                                                    predictions['start_logits'],
                                                    predictions['end_logits']):
        yield RawResult(
            unique_id=unique_ids.numpy(),
            start_logits=start_logits.numpy().tolist(),
            end_logits=end_logits.numpy().tolist())


all_results = []
for count, inputs in enumerate(eval_dataset):
    x, _ = inputs
    unique_ids = x.pop("unique_ids")
    start_logits, end_logits = bert_squad(x, training=False)
    output_dict = dict(
        unique_ids=unique_ids,
        start_logits=start_logits,
        end_logits=end_logits)
    for result in get_raw_results(output_dict):
        all_results.append(result)
    if count % 100 == 0:
        print("{}/{}".format(count, 2709))

output_prediction_file = "/content/drive/My Drive/BERT/data/squad/predictions.json"
output_nbest_file = "/content/drive/My Drive/BERT/data/squad/nbest_predictions.json"
output_null_log_odds_file = "/content/drive/My Drive/BERT/data/squad/null_odds.json"

write_predictions(
    eval_examples,
    eval_features,
    all_results,
    20,
    30,
    True,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose=False)


# Home-made prediction

''' We will concatenate the question and the context, separated by a `["SEP"]`, after tokenization, as it has been made for the training set.
The important thing is that we want our answer to start with a real word and to end with a real word. 
The word "ecologically" being tokenized as `["ecological", "##ly"]`, if the ending token is `["ecological"]` 
we want to use "ecologically" as the ending word (same thing if the ending token is `["##ly"]`). 
That is why we first split our context into words, and then into tokens, remembering to which word each token 
belongs to (see the `tokenize_context()` function). '''

my_bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    trainable=False)
vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


def is_whitespace(c):
    '''
    Tell if a chain of characters corresponds to a whitespace or not.
    '''
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_split(text):
    '''
    Take a text and return a list of "words" by splitting it according to
    whitespaces.
    '''
    doc_tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens


def tokenize_context(text_words):
    '''
    Take a list of words (returned by whitespace_split()) and tokenize each word
    one by one. Also keep track, for each new token, of its original word in the
    text_words parameter.
    '''
    text_tok = []
    tok_to_word_id = []
    for word_id, word in enumerate(text_words):
        word_tok = tokenizer.tokenize(word)
        text_tok += word_tok
        tok_to_word_id += [word_id]*len(word_tok)
    return text_tok, tok_to_word_id

def get_ids(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)

def get_mask(tokens):
    return np.char.not_equal(tokens, "[PAD]").astype(int)

def get_segments(tokens):
    seg_ids = []
    current_seg_id = 0
    for tok in tokens:
        seg_ids.append(current_seg_id)
        if tok == "[SEP]":
            current_seg_id = 1-current_seg_id # turns 1 into 0 and vice versa
    return seg_ids


def create_input_dict(question, context):
    '''
    Take a question and a context as strings and return a dictionary with the 3
    elements needed for the model. Also return the context_words, the
    context_tok to context_word ids correspondance and the length of
    question_tok that we will need later.
    '''
    question_tok = tokenizer.tokenize(my_question)

    context_words = whitespace_split(context)
    context_tok, context_tok_to_word_id = tokenize_context(context_words)

    input_tok = question_tok + ["[SEP]"] + context_tok + ["[SEP]"]
    input_tok += ["[PAD]"]*(384-len(input_tok)) # in our case the model has been
                                                # trained to have inputs of length max 384
    input_dict = {}
    input_dict["input_word_ids"] = tf.expand_dims(tf.cast(get_ids(input_tok), tf.int32), 0)
    input_dict["input_mask"] = tf.expand_dims(tf.cast(get_mask(input_tok), tf.int32), 0)
    input_dict["input_type_ids"] = tf.expand_dims(tf.cast(get_segments(input_tok), tf.int32), 0)

    return input_dict, context_words, context_tok_to_word_id, len(question_tok)


    # Creation

my_context = '''Neoclassical economics views inequalities in the distribution of income as arising from differences in value added by labor, capital and land. Within labor income distribution is due to differences in value added by different classifications of workers. In this perspective, wages and profits are determined by the marginal value added of each economic actor (worker, capitalist/business owner, landlord). 
    Thus, in a market economy, inequality is a reflection of the productivity gap between highly-paid professions and lower-paid professions.'''

my_input_dict, my_context_words, context_tok_to_word_id, question_tok_len = create_input_dict(my_question, my_context)

    # Interpretation

start_logits_context = start_logits.numpy()[0, question_tok_len+1:]
end_logits_context = end_logits.numpy()[0, question_tok_len+1:]

start_word_id = context_tok_to_word_id[np.argmax(start_logits_context)]
end_word_id = context_tok_to_word_id[np.argmax(end_logits_context)]


pair_scores = np.ones((len(start_logits_context), len(end_logits_context)))*(-1E10)
for i in range(len(start_logits_context-1)):
    for j in range(i, len(end_logits_context)):
        pair_scores[i, j] = start_logits_context[i] + end_logits_context[j]
pair_scores_argmax = np.argmax(pair_scores)

start_word_id = context_tok_to_word_id[pair_scores_argmax // len(start_logits_context)]
end_word_id = context_tok_to_word_id[pair_scores_argmax % len(end_logits_context)]

# Final answer

predicted_answer = ' '.join(my_context_words[start_word_id:end_word_id+1])
print("The answer to:\n" + my_question + "\nis:\n" + predicted_answer)

from IPython.core.display import HTML
display(HTML(f'<h2>{my_question.upper()}</h2>'))
marked_text = str(my_context.replace(predicted_answer, f"<mark>{predicted_answer}</mark>"))
display(HTML(f"""<blockquote> {marked_text} </blockquote>"""))