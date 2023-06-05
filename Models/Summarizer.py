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



def extractive_summary(article, compression_ratio):
    print('compressed to: ', compression_ratio)
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



# Abstractive Summary



VOCAB_DIR = 'Summarizer/'
VOCAB_FILE = 'summarize32k.subword.subwords'
MODEL_DIR = 'model'

def tokenize(input_str):
    integers =  next(trax.data.tokenize(iter([input_str]), vocab_dir = VOCAB_DIR, vocab_file = VOCAB_FILE))
    return list(integers) + [1]



def detokenize(integers):
    sentence = trax.data.detokenize(integers, vocab_dir = VOCAB_DIR, vocab_file = VOCAB_FILE)
    return wrapper.fill(sentence)



def DotProductAttention(query, key, value, mask):
    """Dot product self-attention.
    Args:
        query (jax.interpreters.xla.DeviceArray): array of query representations with shape (L_q by d)
        key (jax.interpreters.xla.DeviceArray): array of key representations with shape (L_k by d)
        value (jax.interpreters.xla.DeviceArray): array of value representations with shape (L_k by d) where L_v = L_k
        mask (jax.interpreters.xla.DeviceArray): attention-mask, gates attention with shape (L_q by L_k)

    Returns:
        jax.interpreters.xla.DeviceArray: Self-attention array for q, k, v arrays. (L_q by L_k)
    """

    assert query.shape[-1] == key.shape[-1] == value.shape[-1], "Embedding dimensions of q, k, v aren't all the same"

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # Save depth/dimension of the query embedding for scaling down the dot product

    # Calculate scaled query key dot product according to formula above
    dots = trax_np.matmul(query, trax_np.swapaxes(key, -1, -2)) / trax_np.sqrt(query.shape[-1])
    
    # Apply the mask
    if mask is not None: # The 'None' in this line does not need to be replaced
        dots = trax_np.where(mask, dots, trax_np.full_like(dots, -1e9))
    
    # Softmax formula implementation
    # Use trax.fastmath.logsumexp of dots to avoid underflow by division by large numbers
    # Hint: Last axis should be used and keepdims should be True
    # Note: softmax = e^(dots - logsumexp(dots)) = E^dots / sumexp(dots)
    logsumexp = trax.fastmath.logsumexp(dots, axis=-1, keepdims=True)

    # Take exponential of dots minus logsumexp to get softmax
    # Use jnp.exp()
    dots = trax_np.exp(dots - logsumexp)

    # Multiply dots by value to get self-attention
    # Use jnp.matmul()
    attention = trax_np.matmul(dots, value)

    ## END CODE HERE ###
    
    return attention



def compute_attention_heads_closure(n_heads, d_head):
    """ Function that simulates environment inside CausalAttention function.
    Args:
        d_head (int):  dimensionality of heads.
        n_heads (int): number of attention heads.
    Returns:
        function: compute_attention_heads function
    """

    def compute_attention_heads(x):
        """ Compute the attention heads.
        Args:
            x (jax.interpreters.xla.DeviceArray): tensor with shape (batch_size, seqlen, n_heads X d_head).
        Returns:
            jax.interpreters.xla.DeviceArray: reshaped tensor with shape (batch_size X n_heads, seqlen, d_head).
        """
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        # Size of the x's batch dimension
        batch_size = x.shape[0]
        # Length of the sequence
        # Should be size of x's first dimension without counting the batch dim
        seqlen = x.shape[1]
        # Reshape x using jnp.reshape()
        # batch_size, seqlen, n_heads*d_head -> batch_size, seqlen, n_heads, d_head
        x = trax_np.reshape(x, (batch_size, seqlen, n_heads, d_head))
        # Transpose x using jnp.transpose()
        # batch_size, seqlen, n_heads, d_head -> batch_size, n_heads, seqlen, d_head
        # Note that the values within the tuple are the indexes of the dimensions of x and you must rearrange them
        x = trax_np.transpose(x, (0, 2, 1, 3))
        # Reshape x using jnp.reshape()
        # batch_size, n_heads, seqlen, d_head -> batch_size*n_heads, seqlen, d_head
        x = trax_np.reshape(x, (-1, seqlen, d_head))
        
        ### END CODE HERE ###
        
        return x
    
    return compute_attention_heads



def dot_product_self_attention(q, k, v):
    """ Masked dot product self attention.
    Args:
        q (jax.interpreters.xla.DeviceArray): queries.
        k (jax.interpreters.xla.DeviceArray): keys.
        v (jax.interpreters.xla.DeviceArray): values.
    Returns:
        jax.interpreters.xla.DeviceArray: masked dot product self attention tensor.
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # Hint: mask size should be equal to L_q. Remember that q has shape (batch_size, L_q, d)
    # NOTE: there is a revision underway with the autograder to tolerate better indexing. 
    # Until then, please index q.shape using negative values (this is equivalent to counting from right to left)
    mask_size = q.shape[-2]

    # Creates a matrix with ones below the diagonal and 0s above. It should have shape (1, mask_size, mask_size)
    # Notice that 1's and 0's get casted to True/False by setting dtype to jnp.bool_
    # Use jnp.tril() - Lower triangle of an array and jnp.ones()
    mask = trax_np.tril(trax_np.ones((1, mask_size, mask_size), dtype=trax_np.bool_), k=0)
    
    ### END CODE HERE ###
    
    return DotProductAttention(q, k, v, mask)



def compute_attention_output_closure(n_heads, d_head):
    """ Function that simulates environment inside CausalAttention function.
    Args:
        d_head (int):  dimensionality of heads.
        n_heads (int): number of attention heads.
    Returns:
        function: compute_attention_output function
    """
    
    def compute_attention_output(x):
        """ Compute the attention output.
        Args:
            x (jax.interpreters.xla.DeviceArray): tensor with shape (batch_size X n_heads, seqlen, d_head).
        Returns:
            jax.interpreters.xla.DeviceArray: reshaped tensor with shape (batch_size, seqlen, n_heads X d_head).
        """
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        # Length of the sequence
        # Should be size of x's first dimension without counting the batch dim
        seqlen = x.shape[1]
        # Reshape x using jnp.reshape() to shape (batch_size, n_heads, seqlen, d_head)
        x = trax_np.reshape(x, ( -1, n_heads, seqlen, d_head))
        # Transpose x using trax_np.transpose() to shape (batch_size, seqlen, n_heads, d_head)
        x = trax_np.transpose(x, ( 0, 2, 1 , 3))
        
        ### END CODE HERE ###
        
        # Reshape to allow to concatenate the heads
        return trax_np.reshape(x, (-1, seqlen, n_heads * d_head))
    
    return compute_attention_output



def CausalAttention(d_feature, 
                    n_heads, 
                    compute_attention_heads_closure=compute_attention_heads_closure,
                    dot_product_self_attention=dot_product_self_attention,
                    compute_attention_output_closure=compute_attention_output_closure,
                    mode='train'):
    """Transformer-style multi-headed causal attention.

    Args:
        d_feature (int):  dimensionality of feature embedding.
        n_heads (int): number of attention heads.
        compute_attention_heads_closure (function): Closure around compute_attention heads.
        dot_product_self_attention (function): dot_product_self_attention function. 
        compute_attention_output_closure (function): Closure around compute_attention_output. 
        mode (str): 'train' or 'eval'.

    Returns:
        trax.layers.combinators.Serial: Multi-headed self-attention model.
    """
    
    assert d_feature % n_heads == 0
    d_head = d_feature // n_heads

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # HINT: The second argument to tl.Fn() is an uncalled function (without the parentheses)
    # Since you are dealing with closures you might need to call the outer 
    # function with the correct parameters to get the actual uncalled function.
    ComputeAttentionHeads = tlayer.Fn('AttnHeads', compute_attention_heads_closure(n_heads, d_head), n_out=1)
        

    return tlayer.Serial(
        tlayer.Branch( # creates three towers for one input, takes activations and creates queries keys and values
            [tlayer.Dense(d_feature), ComputeAttentionHeads], # queries
            [tlayer.Dense(d_feature), ComputeAttentionHeads], # keys
            [tlayer.Dense(d_feature), ComputeAttentionHeads], # values
        ),
        
        tlayer.Fn('DotProductAttn', dot_product_self_attention, n_out=1), # takes QKV
        # HINT: The second argument to tl.Fn() is an uncalled function
        # Since you are dealing with closures you might need to call the outer 
        # function with the correct parameters to get the actual uncalled function.
        tlayer.Fn('AttnOutput', compute_attention_output_closure(n_heads, d_head), n_out=1), # to allow for parallel
        tlayer.Dense(d_feature) # Final dense layer
    )
    
    
    
def DecoderBlock(d_model, d_ff, n_heads,
                 dropout, mode, ff_activation):
    """Returns a list of layers that implements a Transformer decoder block.

    The input is an activation tensor.

    Args:
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        list: list of trax.layers.combinators.Serial that maps an activation tensor to an activation tensor.
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # Create masked multi-head attention block using CausalAttention function
    causal_attention = CausalAttention( 
                        d_model,
                        n_heads=n_heads,
                        mode=mode
                        )

    # Create feed-forward block (list) with two dense layers with dropout and input normalized
    feed_forward = [ 
        # Normalize layer inputs
        tlayer.LayerNorm(),
        # Add first feed forward (dense) layer (don't forget to set the correct value for n_units)
        tlayer.Dense(d_ff),
        # Add activation function passed in as a parameter (you need to call it!)
        ff_activation(), # Generally ReLU
        # Add dropout with rate and mode specified (i.e., don't use dropout during evaluation)
        tlayer.Dropout(rate=dropout, mode=mode),
        # Add second feed forward layer (don't forget to set the correct value for n_units)
        tlayer.Dense(d_model),
        # Add dropout with rate and mode specified (i.e., don't use dropout during evaluation)
        tlayer.Dropout(rate=dropout,mode=mode)
    ]

    # Add list of two Residual blocks: the attention with normalization and dropout and feed-forward blocks
    return [
      tlayer.Residual(
          # Normalize layer input
          tlayer.LayerNorm(),
          # Add causal attention block previously defined (without parentheses)
          causal_attention,
          # Add dropout with rate and mode specified
          tlayer.Dropout(rate=dropout, mode=mode)
        ),
      tlayer.Residual(
          # Add feed forward block (without parentheses)
          feed_forward
        ),
      ]
    
    
    
def TransformerLM(vocab_size=33300,
                  d_model=512,
                  d_ff=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  max_len=4096,
                  mode='train',
                  ff_activation=tlayer.Relu):
    """Returns a Transformer language model.

    The input to the model is a tensor of tokens. (This model uses only the
    decoder part of the overall Transformer.)

    Args:
        vocab_size (int): vocab size.
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_layers (int): number of decoder layers.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        max_len (int): maximum symbol length for positional encoding.
        mode (str): 'train', 'eval' or 'predict', predict mode is for fast inference.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        trax.layers.combinators.Serial: A Transformer language model as a layer that maps from a tensor of tokens
        to activations over a vocab set.
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # Embedding inputs and positional encoder
    positional_encoder = [ 
        # Add embedding layer of dimension (vocab_size, d_model)
        tlayer.Embedding(vocab_size, d_model),
        # Use dropout with rate and mode specified
        tlayer.Dropout(rate=dropout, mode=mode),
        # Add positional encoding layer with maximum input length and mode specified
        tlayer.PositionalEncoding(max_len=max_len, mode=mode)]

    # Create stack (list) of decoder blocks with n_layers with necessary parameters
    decoder_blocks = [ 
        DecoderBlock(d_model, d_ff, n_heads,
                    dropout, mode, ff_activation) for _ in range(n_layers)]

    # Create the complete model as written in the figure
    return tlayer.Serial(
        # Use teacher forcing (feed output of previous step to current step)
        tlayer.ShiftRight(mode=mode), # Specify the mode!
        # Add positional encoder
        positional_encoder,
        # Add decoder blocks
        decoder_blocks,
        # Normalize layer
        tlayer.LayerNorm(),

        # Add dense layer of vocab_size (since need to select a word to translate to)
        # (a.k.a., logits layer. Note: activation already set by ff_activation)
        tlayer.Dense(vocab_size),
        # Get probabilities with Logsoftmax
        tlayer.LogSoftmax()
    )
    
    
    
def next_symbol(cur_output_tokens, model):
    token_length = len(cur_output_tokens)
    padded_length = 2**int(np.ceil(np.log2(token_length + 1)))
    padded = cur_output_tokens + [0] * (padded_length - token_length)
    padded_with_batch = np.array(padded)[None, :] 
    output, _ = model((padded_with_batch, padded_with_batch))  
    log_probs = output[0, token_length, :]
    return int(np.argmax(log_probs))



def greedy_decode(input_sentence, model):
    """Greedy decode function.

    Args:
        input_sentence (string): a sentence or article.
        model (trax.layers.combinators.Serial): Transformer model.

    Returns:
        string: summary of the input.
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # Use tokenize()
    cur_output_tokens = tokenize(input_sentence) + [0]
    generated_output = [] 
    cur_output = 0 
    EOS = 1 
    
    while cur_output != EOS:
        # Get next symbol
        cur_output = next_symbol(cur_output_tokens, model)
        # Append next symbol to original sentence
        cur_output_tokens.append(cur_output)
        # Append next symbol to generated sentence
        generated_output.append(cur_output)
#         print(detokenize(generated_output))
    
    ### END CODE HERE ###
    
    return detokenize(generated_output[:-1])

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

# def abstrctive_summary(article):
#     model = TransformerLM(mode='eval')
#     model.init_from_file('Summarizer/abstractive_summary_data/fine_tuning_weights/model.pkl.gz', weights_only=True)
#     return(greedy_decode(article, model))