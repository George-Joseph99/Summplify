import re
import string
import numpy as np
import math
import nltk
import io
import csv
import pandas as pd


from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import islice
from rouge import Rouge



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
#         sentence_preprocessed = " ".join(sentence_preprocessed)
        if sentence_preprocessed:
            article_preprocessed.append(sentence_preprocessed)
            
        
        
#     words = [word_tokenize(sent) for sent in sentences]
#     words_without_stopwords = [[word for word in sent if word not in stopwords.words('english')] for sent in words]
    return article_preprocessed



def convert_list_to_string(sentences):  # converts list of lists to list of strings
    sentences_modified = []   # list of strings
    for sentence in sentences:
        sentence_modified = ' '.join(sentence)
        sentences_modified.append(sentence_modified)
    return sentences_modified



def calculate_TF_IDF_Ours(content):
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
            
#     print(tf_idf/max(tf_idf))
    tf_idf = tf_idf/max(tf_idf)  # might be commented (this normalizes the tf-idf)
            
    return tf_idf



### Feature 2

def sentence_length(article_preprocessed):
    article_preprocessed = convert_list_to_string(article_preprocessed)
    max_length = 0
    for sentence in article_preprocessed:
        # print(sentence)
        if len(sentence.split()) > max_length:
            max_length = len(sentence.split())
            
    sentence_length_feature = []
    for sentence in article_preprocessed:
        sentence_length_feature.append(len(sentence.split()) / max_length)

#     sentence_length_feature = []
#     for sentence in article_preprocessed:
#         sentence_length_feature.append(1 / len(sentence.split()))

    return sentence_length_feature



def generate_X_labels(article_preprocessed):
    # feature 1 (tf_idf)
#     word_scores = calculate_TF_IDF(article_preprocessed)
#     tf_idf_score_feature = calculate_each_sentence_score(article_preprocessed, word_scores)
    tf_idf_score_feature = calculate_TF_IDF_Ours(article_preprocessed)
    
    
    # feature 2 (sentence_length)
    sentence_length_feature = sentence_length(article_preprocessed)
    
    matrix = np.column_stack((tf_idf_score_feature, sentence_length_feature))
#     matrix = np.array(tf_idf_score_feature).reshape(len(tf_idf_score_feature), 1)

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



def initializing_model():
    X_matrix = []
    X = []
    Y = []
    sentences = []

    article_types = ['business', 'entertainment', 'politics', 'sport', 'tech']

    for article_type in article_types:
        for i in range (1, 51):   # loading business articles
            article_file = io.open("train_original/" + article_type + "/article (" + str(i) +").txt", "r", encoding='utf-8-sig')
            article_file.readline()
            article = article_file.read()
            article_file.close()

            summarized_file = io.open("train_summary/" + article_type + "/summary (" + str(i) +").txt", "r", encoding='utf-8-sig')
            summarized = summarized_file.read()
            summarized_file.close()

            article_preprocessed = preprocessing(article)
        #     article_preprocessed = convert_list_to_string(article_preprocessed)
            X_i = generate_X_labels(article_preprocessed)
            Y_i, original_list_no_first_space = generate_Y_labels(article, summarized)

            if(len(X_i) != len(Y_i)):
                print('Error! features and labels are not equal in length')

            Y.extend(Y_i)
            X_matrix.extend(X_i)
            sentences.extend(original_list_no_first_space)
        

    for x in X_matrix:
        X.append(x.tolist())
        
    X = np.matrix(X)

    m = len(X)

    print(len(X))
    print(len(Y))
    
    
def sigmoid(x):
    # TODO 1: Compute the sigmoid function at the given x (~1 line)
    # For example: sigmoid(2) should compute the value of sigmoid function at x = 2.
    # Hint: Use np.exp instead of math.exp to allow for vectorization.
    #----------------------------------------------------------------------------------------------
    sig = (1/(1+np.exp(-x)))
    #----------------------------------------------------------------------------------------------
    
    return sig



def build_model(nn_hdim, nn_input_dim, nn_output_dim, alpha, X, Y, m, num_passes=20000, print_loss=False):
    
    np.random.seed(0)
    W1 = np.random.randn(nn_hdim, nn_input_dim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((nn_hdim, 1))
    W2 = np.random.randn(nn_output_dim, nn_hdim) / np.sqrt(nn_hdim)
    b2 = np.zeros((nn_output_dim, 1))

    model = {}

    for i in range(0, num_passes):
        DW1 = 0
        DW2 = 0
        Db1 = 0
        Db2 = 0
        cost = 0

        for j in range(0, m):
            a0 = X[j, :].reshape(-1, 1)  # Every training example is a column vector.
            y = Y[j]

            z1 = np.dot(W1 , a0 )+ b1
            a1 = np.tanh(z1)
            z2 = np.dot(W2 , a1) + b2
            a2 = sigmoid(z2)
            
#             if (i == num_passes -1 ):
#                 print('True value: %f, got: %f'% (y, a2))

            cost_j = -1 * ((np.log(a2) * y + (1-y)* np.log(1-a2)))

            da2 =  ( -y/a2  + (1-y)/(1-a2) )
            dz2 =  da2 * a2 * ( 1 - a2)
            dW2 = np.dot(dz2 , a1.T)
            db2 = dz2

            da1 =  np.dot(dz2,W2).T
            dz1 = np.multiply(da1 , 1 - np.square(a1) )
            dW1 = np.dot(dz1 , a0.T )
            db1 = dz1

            DW1 += dW1
            DW2 += dW2
            Db2 += db2
            Db1 += db1
            cost += cost_j
        
        DW1 /= m
        DW2 /= m
        Db1 /= m
        Db2 /= m
        cost /= m

        W1 -= alpha * DW1
        b1 -= alpha * Db1
        W2 -= alpha * DW2
        b2 -= alpha * Db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, cost))

    return model



# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    a0 = x.T
    
    # TODO 6 (aka TODO 2): Apply forward propagation on every test example a0 (a column vector 2x1) with its
    #  corresponding label y. It is required to compute z1, a1, z2, and a2  (SAME AS TODO2).
    # -----------------------------------------------------------------------------------------------
    z1 = np.dot(W1 , a0) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2 , a1) + b2
    a2 = sigmoid(z2)
    # ------------------------------------------------------------------------------------------------
    # Applying a threshold of 0.5 (i.e. predictions greater than 0.5 are mapped to 1, and 0 otherwise)
#     prediction = np.round(a2)
    prediction = a2
    
    return prediction



def summarize(article, compression_ratio = 0.35):
    model = model = {'W1': np.array([[ 1.29643591,  0.45074824],
       [ 0.70773118,  1.58460852],
       [ 2.67738766, -1.74837318],
       [ 1.33109077,  0.10529432],
       [ 0.69949236,  0.23108707],
       [-0.35599099,  2.07436754],
       [ 0.47995731,  0.24472955],
       [ 0.63720221,  0.2369889 ]]), 'b1': np.array([[ 0.24555316],
       [-0.02617257],
       [-1.61662627],
       [-1.31429877],
       [-0.17949485],
       [ 0.07467697],
       [ 0.13714999],
       [ 0.03253189]]), 'W2': np.array([[-0.20112114,  0.1343918 , -2.29593635, -1.68775395, -1.66578869,
         1.04315377, -0.56503497, -1.11453418]]), 'b2': np.array([[-0.64778329]])}
    original_sentences = sent_tokenize(article)
    article_preprocessed_entered = preprocessing(article)
    X_test_entered = generate_X_labels(article_preprocessed_entered)
    summary_predicted = predict(model, X_test_entered)
    num_sentences_summarized = math.ceil(compression_ratio * len(original_sentences))
    
    
    highest = np.argsort(summary_predicted[0]) [::-1]
    highest = highest[: num_sentences_summarized]
    highest = sorted(highest) # uncomment to arrange the article
    output_sentences = []
    

    
    for i in range (0, num_sentences_summarized):
        output_sentences.append(original_sentences[highest[i]])
        
    output_sentences = ' '.join(output_sentences)
    
    return output_sentences



def run_cr7():
    article_file = io.open("cr7.txt", "r", encoding='utf-8-sig')
    article = article_file.read()
    article_file.close
    # print(article)
    summary = summarize(article, 0.35)
    return summary

# model = build_model(nn_hdim= 8, num_passes = 10001, print_loss=True)

if __name__ == '__main__':
    run_cr7()