import sys
import os
import re
import string
import numpy as np
import math
import nltk
import io
import csv
import pandas as pd
import trax
import textwrap
wrapper = textwrap.TextWrapper(width=70)


from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import islice
from rouge import Rouge
from trax import layers as tlayer
from trax.fastmath import numpy as trax_np
from trax.supervised import training



# Extractive Summary



def preprocessing(article):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    article_preprocessed = []
    sentences = sent_tokenize(article)
    for sentence in sentences:
        sentence_preprocessed = []
        sentence = re.sub(r"[^a-zA-Z\s]+", "", sentence)
        words = word_tokenize(sentence)
        for word in words:
            if (word not in stopwords_english and word not in string.punctuation):
                word_stemmed = stemmer.stem(word)  
                sentence_preprocessed.append(word_stemmed)
        if sentence_preprocessed:
            article_preprocessed.append(sentence_preprocessed)
    return article_preprocessed

def convert_list_to_string(sentences):  # converts list of lists to list of strings
    sentences_modified = []   # list of strings
    for sentence in sentences:
        sentence_modified = ' '.join(sentence)
        sentences_modified.append(sentence_modified)
    return sentences_modified

### Feature 1

def calculate_TF_IDF(content):
    flat_words = [word for sent in content for word in sent]
    words_set = set(flat_words)
    words_num = len(words_set)
    tf = pd.DataFrame(np.zeros((len(content), words_num)), columns = list(words_set))
    for i in range (len(content)):
        for w in content[i]:
                      tf[w][i] += 1/len(content[i])  
    idf = {}
    for word in words_set:
        num_docs = 0
        for i in range(len(content)):
            if word in content[i]:
                num_docs += 1
        idf[word] = np.log10(len(content) / num_docs)
    tf_idf = np.zeros(len(content))
    for i in range (len(content)):
        for word in content[i]:
            tf_idf[i] += tf[word][i] * idf[word]
    if tf_idf.size == 0:
        return np.zeros(1)
    tf_idf = tf_idf/max(tf_idf)  # might be commented (this normalizes the tf-idf)
    return tf_idf

### Feature 2

def sentence_length(article_preprocessed):
    article_preprocessed = convert_list_to_string(article_preprocessed)
    max_length = 0
    if len(article_preprocessed) == 0:
        return np.zeros(1)
    for sentence in article_preprocessed:
        if len(sentence.split()) > max_length:
            max_length = len(sentence.split())
    sentence_length_feature = []
    for sentence in article_preprocessed:
        sentence_length_feature.append(len(sentence.split()) / max_length)
    return sentence_length_feature

### Feature 3

def proper_nouns_number(sentences):
    number_proper_nouns_in_sentences = []
    proper_nouns_number_feature = []
    stopwords_english = stopwords.words('english')
    for sentence in sentences:
        words_in_sentence = re.sub(r"[^a-zA-Z\s]+", "", sentence)
        words = word_tokenize(words_in_sentence)
        count_true_words = 0
        for word in words:
            if (word not in stopwords_english and word not in string.punctuation):
                count_true_words += 1
        if(count_true_words > 0):
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            proper_nouns = [word for word, pos in tagged if pos == 'NNP']
            number_proper_nouns_in_sentences.append(len(proper_nouns))
    if len(number_proper_nouns_in_sentences) == 0:
        return np.zeros(1)
    if max(number_proper_nouns_in_sentences) > 0:
        for number_proper_nouns_in_sentence in number_proper_nouns_in_sentences:
            proper_nouns_number_feature.append(number_proper_nouns_in_sentence / max(number_proper_nouns_in_sentences))
    else:
        for number_proper_nouns_in_sentence in number_proper_nouns_in_sentences:
            proper_nouns_number_feature.append(0)
    return proper_nouns_number_feature

def generate_X_labels(article_preprocessed, original_sentences):
    # feature 1 (tf_idf)
    tf_idf_score_feature = calculate_TF_IDF(article_preprocessed)
    # feature 2 (sentence_length)
    sentence_length_feature = sentence_length(article_preprocessed)
    # feature 3 (proper nouns number)
    num_proper_nouns_feature = proper_nouns_number(original_sentences)
    matrix = np.column_stack((tf_idf_score_feature, sentence_length_feature, num_proper_nouns_feature))
    return matrix

def generate_Y_labels(original, summarized):
    Y_list = []
    original_sentences = sent_tokenize(original)
    original_sentences[0] = original_sentences[0][1:] # to remove the \n
    summarized_sentences = sent_tokenize(summarized)
    for original_sentence in original_sentences:
        added = 0
        for summarized_sentence in summarized_sentences:
            if original_sentence in summarized_sentence:
                Y_list.append(1)
                added = 1
                break
        if added == 0:
            Y_list.append(0)
    return Y_list, original_sentences
    
    
def sigmoid(x):
    sig = (1/(1+np.exp(-x)))
    return sig

def predict_extractive_model(model, x):
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], + \
                                                             model['W4'], model['b4'], model['W5'], model['b5'], model['W6'], model['b6'], + \
                                                             model['W7'], model['b7']
    a0 = x.T

    z1 = np.dot(W1 , a0)+ b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2 , a1)+ b2
    a2 = np.tanh(z2)
    z3 = np.dot(W3 , a2)+ b3
    a3 = np.tanh(z3)
    z4 = np.dot(W4 , a3)+ b4
    a4 = np.tanh(z4)
    z5 = np.dot(W5 , a4)+ b5
    a5 = np.tanh(z5)
    z6 = np.dot(W6 , a5)+ b6
    a6 = np.tanh(z6)
    z7 = np.dot(W7 , a6) + b7
    a7 = sigmoid(z7)

    prediction = a7    
    return prediction

def extractive_summary(article, compression_ratio):
    print('compressed to: ', compression_ratio)
    model = {'W1': np.array([[ 0.69843821, -0.19993617,  0.36440881],
       [ 0.95903598,  1.42450063,  0.66612429],
       [ 2.1827463 , -0.08856085, -0.19113287]]), 'b1': np.array([[-0.66530508],
       [-0.60750625],
       [-1.03570939]]), 'W2': np.array([[ 0.65695835, -0.449375  ,  1.70663831],
       [ 0.29069578,  0.2822306 ,  0.27918986],
       [ 0.31508882,  1.00816188, -0.34445394]]), 'b2': np.array([[-0.43456939],
       [ 0.00045761],
       [-0.31112202]]), 'W3': np.array([[ 0.11780398, -0.51736596, -1.56001077],
       [ 0.84980301,  0.65826252, -0.46672083],
       [ 1.52883098, -0.7284255 , -0.17640248]]), 'b3': np.array([[ 0.18916633],
       [ 0.06281344],
       [-0.16717293]]), 'W4': np.array([[ 0.0919177 ,  1.14554835,  1.04559518],
       [-0.33379555,  0.21200468, -0.67899995],
       [-1.20565714, -0.17297097,  0.09055869]]), 'b4': np.array([[ 0.02975375],
       [ 0.35624101],
       [-0.16862802]]), 'W5': np.array([[ 1.26675989,  0.59541044, -0.36713659],
       [-0.16337134, -0.59990016, -0.83756652],
       [-1.15320666,  1.3348449 ,  0.0362139 ]]), 'b5': np.array([[0.30658246],
       [0.16403235],
       [0.27393751]]), 'W6': np.array([[-0.34851762, -0.75102519,  0.69490768],
       [-1.33427527, -0.18975495, -0.14022591],
       [ 0.42502826, -0.18174999, -1.10706824]]), 'b6': np.array([[-0.23081771],
       [-0.1860524 ],
       [ 0.05209766]]), 'W7': np.array([[ 0.87815907,  0.80358377, -1.31396283]]), 'b7': np.array([[-0.92787612]])}
    original_sentences = sent_tokenize(article)
    article_preprocessed_entered = preprocessing(article)
    X_test_entered = generate_X_labels(article_preprocessed_entered, original_sentences)
    summary_predicted = predict_extractive_model(model, X_test_entered)
    num_sentences_summarized = math.ceil(compression_ratio * len(original_sentences))
    highest = np.argsort(summary_predicted[0]) [::-1]
    highest = highest[: num_sentences_summarized]
    highest = sorted(highest) # uncomment to arrange the article
    output_sentences = []
    for i in range (0, num_sentences_summarized):
        output_sentences.append(original_sentences[highest[i]])
        
    output_sentences = ' '.join(output_sentences)
    
    return output_sentences



# Abstractive Summary



VOCAB_DIR = 'Summarizer/'
VOCAB_FILE = 'summarize32k.subword.subwords'
MODEL_DIR = 'model'

def tokenize(words):
    tokens =  next(trax.data.tokenize(iter([words]), vocab_dir = VOCAB_DIR, vocab_file = VOCAB_FILE))
    tokens = list(tokens)
    tokens.append(1)
    return tokens

def detokenize(tokens):
    words = trax.data.detokenize(tokens, vocab_dir = VOCAB_DIR, vocab_file = VOCAB_FILE)
    words = wrapper.fill(words)
    return words

def DecoderBlock(model_dim, feed_forward_depth, heads_num, activation_function, mode, dropout):
    features_dim = model_dim
    heads_dim = features_dim // heads_num

    # X is an input of dimension (batch_size, seqlen, heads_num x heads_dim) and will be converted by the following line to 
    # (batch_size x heads_num, seqlen, heads_dim) to allow matrix multiplication 

    compute_wq_wk_wv = tlayer.Fn('AttnHeads', 
                                      lambda x: trax_np.reshape(trax_np.transpose(
                                          trax_np.reshape(x, (x.shape[0], x.shape[1], heads_num, heads_dim)),
                                          (0, 2, 1, 3)),
                                          (-1, x.shape[1], heads_dim)), n_out=1)
    

    # Create feed-forward block (list) with two dense layers with dropout and input normalized
    # Add list of two Residual blocks: the attention with normalization and dropout and feed-forward blocks
    return [
      tlayer.Residual(
          tlayer.LayerNorm(),
          tlayer.Serial(
            tlayer.Branch( # creates three towers for one input, takes activations and creates queries keys and values
                [tlayer.Dense(features_dim), compute_wq_wk_wv], # queries
                [tlayer.Dense(features_dim), compute_wq_wk_wv], # keys
                [tlayer.Dense(features_dim), compute_wq_wk_wv], # values
            ),
            tlayer.Fn('DotProductAttn', lambda query, key, value: trax_np.matmul(trax_np.exp(trax_np.where(
                                trax_np.tril(trax_np.ones((1, query.shape[-2], query.shape[-2]), 
                                dtype=trax_np.bool_), k=0), trax_np.matmul(query, trax_np.swapaxes(key, -1, -2)) / trax_np.sqrt(query.shape[-1]),
                                trax_np.full_like(trax_np.matmul(query, trax_np.swapaxes(key, -1, -2)) / trax_np.sqrt(query.shape[-1]), -1e9)) -                      
                                trax.fastmath.logsumexp(trax_np.where(trax_np.tril(trax_np.ones((1, query.shape[-2], query.shape[-2]), dtype=trax_np.bool_), k=0),
                                trax_np.matmul(query, trax_np.swapaxes(key, -1, -2)) / trax_np.sqrt(query.shape[-1]),
                                trax_np.full_like(trax_np.matmul(query, trax_np.swapaxes(key, -1, -2)) / trax_np.sqrt(query.shape[-1]), -1e9)),
                                axis=-1, keepdims=True)), value), n_out=1), 
            tlayer.Fn('AttnOutput', lambda x: trax_np.reshape(trax_np.transpose(
                                trax_np.reshape(x, ( -1, heads_num, x.shape[1], heads_dim)), 
                                ( 0, 2, 1 , 3)), (-1, x.shape[1], heads_num * heads_dim)), n_out=1), # to allow for parallel


            tlayer.Dense(features_dim) # Final dense layer
            ),
          tlayer.Dropout(rate = dropout, mode = mode)
        ),
      tlayer.Residual(
            [ 
                tlayer.LayerNorm(),
                tlayer.Dense(feed_forward_depth),
                activation_function(), # Generally ReLU
                tlayer.Dropout(rate = dropout, mode = mode),
                tlayer.Dense(model_dim),
                tlayer.Dropout(rate = dropout,mode = mode)
            ]
        )
      ]
    
def TransformerLM(vocab_size=33300, model_dim=512, feed_forward_depth=2048,
                  layers_num=6, heads_num=8, dropout=0.1, max_len=4096,
                  mode='train', activation_function=tlayer.Relu):

    positional_encoder = [ 
        tlayer.Embedding(vocab_size, model_dim),
        tlayer.Dropout(rate=dropout, mode=mode),
        tlayer.PositionalEncoding(max_len=max_len, mode=mode)]

    decoder_blocks = [ 
        DecoderBlock(model_dim, feed_forward_depth, heads_num, activation_function, mode, dropout) 
        for _ in range(layers_num)]

    return tlayer.Serial(
        tlayer.ShiftRight(mode=mode),
        positional_encoder,
        decoder_blocks,
        tlayer.LayerNorm(),
        tlayer.Dense(vocab_size),
        tlayer.LogSoftmax()
    )

def abstractive_summary(article):
    model = TransformerLM(mode='eval')
    model.init_from_file('Summarizer/abstractive_summary_data/fine_tuning_weights/model.pkl.gz', weights_only=True)
    article_summary = tokenize(article)
    article_summary.append(0)
    summary = []
    generated_word = 0
    start_summary = len(article_summary) 
    while generated_word != 1:
        length_padding = np.power(2, int(np.ceil(np.log2(len(article_summary) + 1)))) - len(article_summary)
        article_summary_padded = article_summary.copy()
        for _ in range(length_padding):
            article_summary_padded.append(0)
        article_summary_padded = np.array(article_summary_padded)[None, :]
        output, _ = model((article_summary_padded, article_summary_padded))  
        log_probs = output[0, len(article_summary), :]
        generated_word = int(np.argmax(log_probs))
        article_summary.append(generated_word)
        
    for i in range(start_summary, len(article_summary)):
        summary.append(article_summary[i])
        
    summary = detokenize(summary[: -1])
    return summary
