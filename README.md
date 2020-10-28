# Fine tuned BERT model for question answering system
**BERT** (Bidirectional Encoder Representations from Transformers) is a recent model introduced by researchers at Google AI Language. BERT makes use of Transformer, an attention mechanism that learns contextual relations between words in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task.

The chart below is a high-level description of the Transformer encoder. The input is a sequence of tokens, which are first embedded into vectors and then processed in the neural network. The output is a sequence of vectors of size H, in which each vector corresponds to an input token with the same index.

Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a [MASK] token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence.

<img src="https://user-images.githubusercontent.com/28005338/84592125-87a4fc80-ae43-11ea-9d18-09678cd9e43f.png" width="450" align="center"/>

Generally speaking during training BERT learns to understand a language and the relation between words. Then BERT has to be fine-tuned for a specific task: *the model receives a question regarding a text sequence and is required to mark the answer in the paragraph*.

For this reason we use **SQuAD 2.2 dataset** (Stanford Question and Answers Dataset) is a question answering dataset containing  100,000+ questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage.

The model tokenize every example in QA, then generate multiple instances per example by concatenating a “[CLS]” token, the tokenized question, a “[SEP]” token, tokens from the content of the document, and a final “[SEP]” token, limiting the total size of each instance to 512 tokens. For each document we generate all possible instances, by listing the document content. For each training instance start and end token indices are computed  to represent the target answer span.

The chart below is a high-level description of the fine-tuning BERT:
<img src="https://user-images.githubusercontent.com/28005338/84592410-d5baff80-ae45-11ea-970e-f93d3ac717a8.png" align="center"/>

**Example**

*Phrase: "The children sat around the fire"*

The idea is to extract the answer for the following question:

*Query: "Who is sitting around the fire?"*

Then the model output will be

<img src="https://user-images.githubusercontent.com/28005338/84592625-6c3bf080-ae47-11ea-9cd3-5820fdede06f.png" align="center"/>
