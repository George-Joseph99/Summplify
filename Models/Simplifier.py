import re
import numpy as np
import nltk
import csv
import pandas as pd
import nltk
import re
# import requests
import syllables
import enchant
import gensim.downloader as api

from nltk.corpus import brown
from difflib import SequenceMatcher
from array import array
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import zipf_frequency
from nltk.corpus import wordnet
from py_thesaurus import Thesaurus
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from pattern.text.en import pluralize, singularize, comparative, superlative, conjugate
from pattern.text.en import tenses, INFINITIVE, PRESENT, PAST, FUTURE

wiki_freq_dict = {}
lexicon_dict = {}
complex_words = []
BIGHUGE_KEY = "105a58f9d880af14af1ca1abf6b1f996"
word_vectors = api.load("glove-wiki-gigaword-300")
ppdb_filepath = "Simplifier/ppdb-2.0-m-lexical"
lemmatizer = WordNetLemmatizer()


def is_similar(word1, word2):
    similarity_ratio = SequenceMatcher(None, word1, word2).ratio()
    return similarity_ratio >= 0.7

# Function used for flattening list of lists to a single list
def flatten(l):
    return [item for sublist in l for item in sublist]

def generateWikiFreqDict(text_file):
    wiki_freq_dict = {}
    with open(text_file, encoding='utf8') as f:
        for line in f.readlines():
            (word, freq) = line.split()
            wiki_freq_dict[word.lower()] = freq
    return wiki_freq_dict

wiki_freq_dict = generateWikiFreqDict("Simplifier/wiki_frequencies.txt")

def generateLexiconDict(tsv_file):
    lexicon_dict = {}
    with open(tsv_file) as f:
        file = csv.reader(f, delimiter="\t")
        for line in file:
            word = line[0]
            score = line[1]
            lexicon_dict[word.lower()] = score
    return lexicon_dict

lexicon_dict = generateLexiconDict('Simplifier/lexicon.tsv')

# LS step1: Complex Word Identification

def word_preceding(text, word):
    sentences = sent_tokenize(text)
    words = []
    for sentence in sentences:
        words_from_sentence = word_tokenize(sentence)
        words.extend(words_from_sentence)
    index = words.index(word)
    return words[index-1] if index > 0 else None

def word_following(text, word):
    sentences = sent_tokenize(text)
    words = []
    for sentence in sentences:
        words_from_sentence = word_tokenize(sentence)
        words.extend(words_from_sentence)
    index = words.index(word)
    return words[index+1] if index+1 < len(words) else None

def complexWordIdentif(article):
    threshold_scores_dict = {}
    word_sentence_dict = {}
    for sentence in sent_tokenize(article):
        for word_ in word_tokenize(sentence):
            word = word_.lower()
            word_split_hyphen = word_.split("-")
            word_split_underscore = word_.split("_")
            if(len(word_split_hyphen)>1):
                total_freq = 0
                total_lexicon = 0
                for word_hyph in word_split_hyphen:
                    total_freq+=int(wiki_freq_dict.get(word_hyph,0))
                    total_lexicon+=float(lexicon_dict.get(word_hyph,0))
                wiki_freq = int(total_freq/len(word_split_hyphen))
                lexicon_score = float(total_lexicon/len(word_split_hyphen))
                if(wiki_freq<12000 or lexicon_score>3.0):
                    threshold_scores_dict[word_]=1
                    word_sentence_dict[word_] = sentence
                    
            elif(len(word_split_underscore)>1):
                total_freq = 0
                total_lexicon = 0
                for word_un in word_split_underscore:
                    total_freq+=int(wiki_freq_dict.get(word_un,0))
                    total_lexicon+=float(lexicon_dict.get(word_un,0))
                wiki_freq = int(total_freq/len(word_split_underscore))
                lexicon_score = float(total_lexicon/len(word_split_underscore))
                if(wiki_freq<12000 or lexicon_score>3.0):
                    threshold_scores_dict[word_]=1
                    word_sentence_dict[word_] = sentence

            else:
                threshold_scores_dict[word]=0
                if(word in wiki_freq_dict and word in lexicon_dict):
                    wiki_freq = int(wiki_freq_dict[word])
                    lexicon_score = float(lexicon_dict[word])
    #                 print(word, wiki_freq, lexicon_score)
                    if(wiki_freq<12000 or lexicon_score>3.0):
                        threshold_scores_dict[word_]=1
                        word_sentence_dict[word_] = sentence

                elif(word in wiki_freq_dict):
                    wiki_freq = int(wiki_freq_dict[word])
    #                 print(word, wiki_freq)
                    if(wiki_freq<12000):
                        threshold_scores_dict[word_]=1
                        word_sentence_dict[word_] = sentence
                elif(word in lexicon_dict):
                    lexicon_score = float(lexicon_dict[word])
    #                 print(word, lexicon_score)
                    if(lexicon_score>3.0):
                        threshold_scores_dict[word_]=1
                        word_sentence_dict[word_] = sentence
    return threshold_scores_dict, word_sentence_dict

# Get the pos tag of a certain word 

def getPosTag(word):
    tag = nltk.pos_tag([word])
    return tag[0][1]

def getPosTagFromSentence(string, target_word):
    pos_tag=""
    tokens = nltk.word_tokenize(string.lower())
    tag = nltk.pos_tag(tokens)
    for pair in tag:
        if(pair[0]==target_word.lower()):
            pos_tag = pair[1]
    return pos_tag

# Used with word net dictionary

def getTypeFromTag(tag):
    # Convert all forms of noun tags to "n" noun type
    if(tag=="NN" or tag=="NNS" or tag=="NNP" or tag=="NNPS"):
        return 'n'
    # Convert all forms of adjective tags to "a" adjective type
    elif(tag=="JJ" or tag=="JJR" or tag=="JJS"):
        return 'a'
    # Convert all forms of verb tags to "v" verb type
    elif(tag=="VBZ" or tag=="VB" or tag=="VBP" or tag=="VBN" or tag=="VBG" or tag=="VBD"):
        return 'v'
    # Convert all forms of adverb tags to "r" adverb type
    elif(tag=="RBS" or tag=="RB" or tag=="RBR"):
        return 'r'
    else:
        return tag
    
# Word Net Synonyms

def getSynWordNet(complex_word):
    synonyms = []
    for synset in wordnet.synsets(complex_word):
            for l in synset.lemmas():
                synonyms.append(l.name())
    return list(set(synonyms))

# Word Net Synonyms with specific pos tag

def getSynWordNetSpec(complex_word, pos_tag):
    synonyms = []
    for synset in wordnet.synsets(complex_word):
        if(synset.pos()==pos_tag):
            for l in synset.lemmas():
                synonyms.append(l.name())
    return list(set(synonyms))

# BigHuge thesaurus

def getSynBigHuge(complex_word):
    bighuge_synonyms = []
    r = requests.get(url='http://words.bighugelabs.com/api/2/'+BIGHUGE_KEY+'/'+complex_word+'/json')  
    if(r.status_code!=404 and r.status_code!=500):
#         print(type(r.json()),"\n",r.json())
        if(type(r.json()) is dict):
            synonym_dict = r.json()
            for key in synonym_dict: # key may be: noun/verb/adjective/adverb
                synonym_list = synonym_dict[key].get("syn")
                if(synonym_list):
                    bighuge_synonyms.append(synonym_list)
            flatList = [element for innerList in bighuge_synonyms for element in innerList] # Convert it to a single list
            return flatList,synonym_dict 
        else:
            return r.json(),{}
    else:
        return [],{}
            

# May not use this function to avoid doing multiple requests for the same word
# Instead, use the aboive one and make one request per word, then search for the specific pos tag inside the returned dict
def getSynBigHugeSpec(complex_word, pos_tag):
    bighuge_synonyms = []
    r = requests.get(url='http://words.bighugelabs.com/api/2/'+BIGHUGE_KEY+'/'+complex_word+'/json') 
    if(r.status_code!=404):
        if(type(r.json()) is dict):
            synonym_dict = r.json()
            if(pos_tag in synonym_dict):
                bighuge_synonyms = synonym_dict[pos_tag].get("syn")
        else:
            bighuge_synonyms = r.json()
    return bighuge_synonyms

# OpenOffice thesaurus

# This generates a dictionary "thesaurus" with (word,pos) as the key and a set of synonyms as the value
# For example:
# thesaurus[("happy","adj")] = {'glad', 'pleased', 'prosperous', 'cheerful', ... }
# possible pos: noun,verb,adj,adv
def generateTheSaurusDict():
    thesaurus = {}
    with open("Simplifier/th_en_US_new.dat") as f:
        code = f.readline()    # Skip the file encoding
        while(True):
            word_count_line = f.readline()
            if(word_count_line == ""):
                break
            (word,count) = word_count_line.split('|')
            for i in range(0,int(count)):
                pos_synonyms = f.readline().split("|")
                synonyms_list = pos_synonyms[1:]
                pos = re.sub(r"[\([{})\]]", "",pos_synonyms[0]) # Remove te brackest surronding the pos (noun - verb - adv - adj)
                synonyms_list = [synonym.strip() for synonym in synonyms_list] # Remove unnecessary spaces
                if((word,pos) not in thesaurus):
                    thesaurus[(word,pos)]=set()
                for synonym in synonyms_list:
                    if(synonym not in thesaurus[(word,pos)] and synonym!=word):
                        thesaurus[(word,pos)].add(synonym)
    return thesaurus
# PPDB dictionary

def genPPDBdict(file_path):
    ppdb_dict = {}
    with open(file_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            line = row[0]
            word_list = line.split("|||")
            source = word_list[1].strip()
            target = word_list[2].strip()
            if source in ppdb_dict:
                ppdb_dict[source].append(target)
            else:
                ppdb_dict[source] = [target]
        return ppdb_dict
    
thesaurus = generateTheSaurusDict()
ppdb = genPPDBdict(ppdb_filepath)

def genSubstitution(complex_word):
    thesaurus_candidates = []
    wordnet_candidates = []
    wordnet_candidates = getSynWordNet(complex_word)
    for key in thesaurus:
        if(key[0] == complex_word):
            thesaurus_candidates.append(thesaurus.get((key[0], key[1])))
    return wordnet_candidates, list(thesaurus_candidates)
    
def genSubstitutionSpec(complex_word, sentence):
    thesaurus_candidates = []
    bighuge_candidates = []
    wordnet_candidates = []
    ppdb_candidates = []
    pos_tag = getPosTagFromSentence(sentence, complex_word)
    word_type = getTypeFromTag(pos_tag)
    wordnet_candidates = getSynWordNetSpec(complex_word.lower(), word_type)
    flat_syn_list, bighuge_dict = [],{}
#     flat_syn_list, bighuge_dict = getSynBigHuge(complex_word)
    if(complex_word.lower() in ppdb):
        ppdb_candidates = ppdb[complex_word.lower()]
    if(flat_syn_list and not bighuge_dict):
        bighuge_candidates = flat_syn_list
    if(word_type == 'n'):
        thesaurus_candidates = thesaurus.get((complex_word.lower(),"noun"))
        if("noun" in bighuge_candidates):
            bighuge_candidates = bighuge_dict["noun"].get("syn")
    elif(word_type == 'r'):
        thesaurus_candidates = thesaurus.get((complex_word.lower(),"adv"))
        if("adverb" in bighuge_candidates):
            bighuge_candidates = bighuge_dict["adverb"].get("syn")
    elif(word_type == 'v'):
        thesaurus_candidates = thesaurus.get((complex_word.lower(),"verb"))
        if("verb" in bighuge_candidates):
            bighuge_candidates = bighuge_dict["verb"].get("syn") 
    elif(word_type == 'a'):
        thesaurus_candidates = thesaurus.get((complex_word.lower(),"adj"))
        if("adjective" in bighuge_candidates):
            bighuge_candidates = bighuge_dict["adjective"].get("syn")

    return wordnet_candidates, thesaurus_candidates, bighuge_candidates, ppdb_candidates


def filterSubstitutions(word_subs_dict):
    word_subs_dict_filtered = {}
    wnl = WordNetLemmatizer()
    ps = PorterStemmer()
    for word in word_subs_dict:
        word_lemm = wnl.lemmatize(word.lower())
        filtered_subs_list = []
        subs_set = word_subs_dict[word]
        if(len(subs_set)>0):
            for subs in subs_set:
                word_list = nltk.word_tokenize(subs)
                word_split_hyphen = subs.split("-")
                word_split_underscore = subs.split("_")
                if(len(word_list) > 1):
                    add_word = True
                    for word_ in word_list:
                        word_lemm = wnl.lemmatize(word.lower())
                        subs_lemm = wnl.lemmatize(word_.lower())
                        if(ps.stem(word_.lower())==ps.stem(word.lower()) or subs_lemm==word_lemm or is_similar(word_.lower(), word.lower()) or is_similar(word_.lower(), word_lemm) 
                          or is_similar(subs_lemm, word_lemm) or is_similar(subs_lemm, word.lower())):
                            add_word = False
                            break
                    if(add_word):
                        filtered_subs_list.append(subs)
                elif(len(word_split_hyphen)>1):
                    add_word = True
                    for word_ in word_split_hyphen:
                        word_lemm = wnl.lemmatize(word.lower())
                        subs_lemm = wnl.lemmatize(word_.lower())
                        if(ps.stem(word_.lower())==ps.stem(word.lower()) or subs_lemm==word_lemm or is_similar(word_.lower(), word.lower()) or is_similar(word_.lower(), word_lemm) 
                          or is_similar(subs_lemm, word_lemm) or is_similar(subs_lemm, word.lower())):
                            add_word = False
                            break
                    if(add_word):
                        filtered_subs_list.append(subs)
                elif(len(word_split_underscore)>1):
                    add_word = True
                    for word_ in word_split_underscore:
                        word_lemm = wnl.lemmatize(word.lower())
                        subs_lemm = wnl.lemmatize(word_.lower())
                        if(ps.stem(word_.lower())==ps.stem(word.lower()) or subs_lemm==word_lemm or is_similar(word_.lower(), word.lower()) or is_similar(word_.lower(), word_lemm)
                          or is_similar(subs_lemm, word_lemm) or is_similar(subs_lemm, word.lower())):
                            add_word = False
                            break
                    if(add_word):
                        filtered_subs_list.append(subs)
                else: 
                    subs_lemm = wnl.lemmatize(subs.lower())
                    if(ps.stem(subs.lower())!=ps.stem(word.lower()) and subs_lemm!=word_lemm and not is_similar(subs.lower(), word.lower()) and not is_similar(subs.lower(), word_lemm)
                      and not is_similar(subs_lemm, word_lemm) and not is_similar(subs_lemm, word.lower())):
                        filtered_subs_list.append(subs)
        word_subs_dict_filtered[word] = filtered_subs_list
    return word_subs_dict_filtered

def convertGrammStructure(word_subs_dict, word_sentence_dict):
    modified_word_subs_dict = word_subs_dict.copy()
    for word in word_subs_dict:
        subs_list = word_subs_dict[word]
        word_tag = getPosTagFromSentence(word_sentence_dict[word],word)
        word_type = getTypeFromTag(word_tag)
        if(word_type == 'n'): #NOUNS
            modified_subs = []
            for subs in subs_list:
                subs_tag = getPosTag(subs)
                subs_type = getTypeFromTag(subs_tag)
#                 print(subs, subs_type)
                if(word_tag=="NN" and subs_tag=="NNS"):
                    new_subs = singularize(subs)
                    modified_subs.append(new_subs)
                elif(word_tag=="NNS" and subs_tag=="NN"):
                    new_subs = pluralize(subs)
                    modified_subs.append(new_subs)
                elif((word_tag=="NNS" and subs_tag=="NNS") or (word_tag=="NN" and subs_tag=="NN")):
                    modified_subs.append(subs)
            modified_word_subs_dict[word] = modified_subs
        elif(word_type == 'a'): #ADJECTIVES
            modified_subs = []
            for subs in subs_list:
                subs_tag = getPosTag(subs)
                subs_type = getTypeFromTag(subs_tag)
                if(word_tag=="JJR" and (subs_tag=="JJS" or subs_tag=="JJ")):
                    new_subs = comparative(subs)
                    modified_subs.append(new_subs)
                elif(word_tag=="JJS" and (subs_tag=="JJR" or subs_tag=="JJ")):
                    new_subs = superlative(subs)
                    modified_subs.append(new_subs)
                elif((word_tag=="JJR" and subs_tag=="JJR") or (word_tag=="JJS" and subs_tag=="JJS") or (word_tag=="JJ" and subs_tag=="JJ")):
                    modified_subs.append(subs)
            modified_word_subs_dict[word] = modified_subs
        elif(word_type == 'v'): #VERBS
            modified_subs = []
            try:
                for subs in subs_list:
                    subs_tokens = nltk.word_tokenize(subs)
    #                 subs_split = subs.split(" ")
                    subs_tag = getPosTag(subs)
                    subs_type = getTypeFromTag(subs_tag)
                    if(word_tag=="VBD" or word_tag=="VBN"): #-->PAST
                        if(len(subs_tokens)>1):
                            first_half_tense = tenses(subs)
                            if(subs_type=='v'):
                                new_word = conjugate(subs_tokens[0], tense=PAST)
                                subs_tokens[0] = new_word
                                new_subs = ' '.join(subs_tokens)
                                modified_subs.append(new_subs)
                        else:
                            new_subs = conjugate(subs, tense=PAST)
                            modified_subs.append(new_subs)
                    else:
                        complex_tense = tenses(word)
                        if(word_tag=="VBP" or word_tag=="VBZ"): #-->PRESENT
                            new_subs = conjugate(subs, tense=PRESENT)
                            modified_subs.append(new_subs)
                        elif((len(complex_tense)>0 and complex_tense[0][0]=="infinitive") or word_tag=="VBG"): #-->INFINITIVE
                            new_subs = conjugate(subs, tense=INFINITIVE)
                            modified_subs.append(new_subs)
                        elif(len(complex_tense)>0 and complex_tense[0][0]=="future"): #-->FUTURE
        #                     print(word_tag, complex_tense, complex_tense[0][0], word, "\n")
                            new_subs = conjugate(subs, tense=FUTURE)
                            modified_subs.append(new_subs)   
                        else:
                            modified_subs.append(subs)  
                modified_word_subs_dict[word] = modified_subs
            except StopIteration as e:
                print("exceptionn") 
    return modified_word_subs_dict

# LS Step3: Substitution Ranking

def getNgramScore(phrase, start_year=2000, end_year=2019, corpus=26, smoothing=0):
    avg_score = 0
    google_ngram_url = "https://books.google.com/ngrams/json?content="+phrase+'&year_start=' + str(start_year) + '&year_end=' + str(end_year) + '&corpus=' + str(corpus) + '&smoothing=' + str(smoothing)
    # response = requests.get(google_ngram_url)
    response = ''
    if(response):
        output = response.json()
        print(output)
        return_data = []
        scores_list = []
        total_score = 0
        if(len(output) > 0):
            for num in range(len(output)):
                scores_list.append(output[num]['timeseries'])
            list_flatten = flatten(scores_list)
            for score in list_flatten:
                total_score+=(score)
            avg_score = total_score/len(list_flatten)
    return avg_score
            
def extractFeaturesFromWord(target_word, three_gram_dict):
    
    # Features we have are:
    # lex_exist_flag, complexity_score, word_length, syllable_count, wiki_freq, ngram_score
    lex_exist_flag = -1
    complexity_score = -1
    word_length = -1
    syllable_count = -1
    wiki_freq = -1
    ngram_score = -1

    
    # Before extracting features, check if it's a multi-word phrase, if so, we work on the longest word
    longest_word = ''
    word_list = nltk.word_tokenize(target_word)
    word_split_hyphen = target_word.split("-")
    word_split_underscore = target_word.split("_")
    if(len(word_list) > 1):
        longest_word = ''
        max_length = 0
        for word in word_list:
            if(len(word)>max_length):
                max_length=len(word)
                longest_word = word
    elif(len(word_split_hyphen) > 1):
        longest_word = ''
        max_length = 0
        for word in word_split_hyphen:
            if(len(word)>max_length):
                max_length=len(word)
                longest_word = word
    elif(len(word_split_underscore) > 1):
        longest_word = ''
        max_length = 0
        for word in word_split_underscore:
            if(len(word)>max_length):
                max_length=len(word)
                longest_word = word
        
    # EXTRACTING FEATURES
    
    # Feature 1: Binary number representing word's presence in lexicon (1:existent 0:non existent)
#     if(target_word in lexicon_dict):
#         lex_exist_flag = 1
#     else:
#         lex_exist_flag = 0
            
    # Feature 2: Complexity score of the word in the lexicon
    if(target_word in lexicon_dict):
        complexity_score = float(lexicon_dict[target_word])
    else:
        complexity_score = 0 # If word is not found in lexicon, set its complexity score with 0 also
      
    # Feature 3: word length (character count)
    word_length = len(target_word)
    
    # Feature 4: Syllable count
    syllable_count = syllables.estimate(target_word)     
    
    # Feature 5: Frequency with respect to Wiki-Frequency
    if(target_word in wiki_freq_dict):
        wiki_freq = int(wiki_freq_dict[target_word])
    else:
        wiki_freq = 0
        
    # Feature 6: Google Ngram average score
    if(target_word in three_gram_dict):
        three_words = three_gram_dict[target_word]
#         for word in three_words:
#             phrase+=word
        phrase = three_words[0] + " " + three_words[1] + " " + three_words[2]
        # print("p ",phrase)
        ngram_score = getNgramScore(phrase)
    else:
        ngram_score = 0
    
    return [complexity_score, word_length, syllable_count, wiki_freq, ngram_score] 

# Pair wise features
def getCosSim(vec_1, vec_2):
    cos_sim = cosine_similarity([vec_1], [vec_2])
    return cos_sim[0][0]

def similarityRatio(word1, word2):
    similarity_ratio = SequenceMatcher(None, word1, word2).ratio()
    return similarity_ratio 
# -------------------------------
def sigmoid(x):
    sig = (1/(1+np.exp(-x)))   
    return sig

def batch_normalize(X, eps=1e-5):
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    X_norm = (X - mean) / np.sqrt(var + eps)
    return X_norm

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    a0 = x.T

    # Forward propagation
    z1 = np.dot(W1 , a0) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2 , a1) + b2
    a2 = z2
    prediction = a2  
    return prediction[0]

def simplify(text):
    
    model = {'W1': [[ 0.66145244,  0.16270804,  0.37252965,  0.84023676,  0.7127424 ,
        -0.3567563 ,  0.37010674],
       [-0.04221344,  0.04851904,  0.12056598,  0.15521051,  0.60280313,
         0.31525486,  0.1044011 ],
       [ 0.16965626,  0.17308712,  0.52687586, -0.0186932 ,  0.26320379,
        -0.29983551, -0.92539477],
       [ 0.15125279,  0.27851575, -0.37832276,  0.86823293, -0.4259034 ,
         0.03437329, -0.06773697],
       [ 0.57883909,  0.56154123,  0.04959171,  0.09650677, -0.27608262,
        -0.73508525, -0.11209866],
       [ 0.03722021,  0.41981163,  0.43454857, -0.15009002, -0.14776318,
        -0.40222456, -0.55945027],
       [-0.66710857,  0.69112549, -0.24370165, -0.13958974, -0.43323821,
         0.30040376, -0.62147804],
       [-0.16938653, -0.51091617, -0.0947284 , -0.20217333, -0.25982684,
         0.01728051,  0.153387  ]], 'b1': [[ 0.01014841],
       [-0.04420122],
       [-0.1130388 ],
       [-0.12162635],
       [-0.05295276],
       [-0.00409472],
       [-0.08975829],
       [-0.39391215]], 'W2': [[-0.02744851, -0.39425468, -0.16648875, -0.16939112, -0.13249023,
         0.03129614, -0.05804739, -0.40915087]], 'b2': [[0.34160712]]}
    
    num_features = 7
    vowels = "aeiouAEIOU"
    prediction = []
    word_replace_dict = {}
    thresh_scores = {} 
    thresh_scores, word_sentence_dict = complexWordIdentif(text)
#     print(word_sentence_dict)
    word_subst_dict = {}
    new_text = text # initialize the new string with the original one
    for word in word_sentence_dict:
        word_subst_dict[word] = set()
        sentence = word_sentence_dict[word]
        word_net_cand, thesaurus_cand, bighuge_cand, ppdb_candidates = genSubstitutionSpec(word, sentence)
        if(word_net_cand):
            word_subst_dict[word].update(word_net_cand)
        if(thesaurus_cand):
            word_subst_dict[word].update(thesaurus_cand)
        if(bighuge_cand):
            word_subst_dict[word].update(bighuge_cand)
        if(ppdb_candidates):
            word_subst_dict[word].update(ppdb_candidates)
    filtered_subs_dict = filterSubstitutions(word_subst_dict)
#     print(word_subst_dict)
#     print(filtered_subs_dict)
    modified_word_subs_dict = convertGrammStructure(filtered_subs_dict, word_sentence_dict)    
#     print(modified_word_subs_dict)
    
    # EXTRACT FEATURES
    for target_word in modified_word_subs_dict:
        three_gram_dict = {}
        feature_matrix = []
        cosine_similarities = []
        similarity_ratios = []
        candidates = []
        candidates = modified_word_subs_dict[target_word]
#         print(candidates)
#         candidates = [target_word] + modified_word_subs_dict[target_word]
        if(len(candidates)>0):
            for candidate in candidates:
                three_gram_phrase = ''
                prev_word = word_preceding(sentence, target_word)
                next_word = word_following(sentence, target_word)
                if(prev_word and next_word):
                    three_gram_phrase = prev_word + " " + candidate + " " + next_word
                    three_gram_dict[candidate] = [prev_word, candidate, next_word]
                features_list = extractFeaturesFromWord(candidate, three_gram_dict)
                if(target_word.lower() in word_vectors and candidate.lower() in word_vectors):     
                    target_word_vector = word_vectors[target_word.lower()]
                    substitution_vector = word_vectors[candidate.lower()]
                    cos_similarity = getCosSim(target_word_vector, substitution_vector)
                else:
                    cos_similarity = 0
                similarity_ratios.append(similarityRatio(target_word.lower(), candidate.lower()))
                cosine_similarities.append(cos_similarity)
                feature_matrix.append(features_list)
            cosine_similarities = np.array(cosine_similarities).reshape(len(feature_matrix),1)
            similarity_ratios = np.array(similarity_ratios).reshape(len(feature_matrix),1)
            X = np.hstack((feature_matrix,cosine_similarities, similarity_ratios))
            max_in_column = np.max(X,axis=0)
            for i in range(num_features):
                if(max_in_column[i] != 0):
                    X[:, i] /= max_in_column[i]
#             print(candidates)
#             print(X)
            prediction = predict(model, X)
#             print(candidates)
#             print(prediction)
            min_value = min(prediction)
            prediction_list = prediction.tolist()
            min_index=prediction_list.index(min_value)
#             print(prediction_list,"\n",candidates)
            chosen_candidate = candidates[min_index]
            if(target_word[0].isupper()):
                chosen_candidate = chosen_candidate[0].upper() + chosen_candidate[1:]
            if(target_word.isupper()):
                chosen_candidate = chosen_candidate.upper()
            if(prev_word == 'a' and chosen_candidate[0] in vowels):
                new_text = new_text.replace('a '+ target_word, 'an ' + chosen_candidate)
            elif(prev_word == 'an' and chosen_candidate[0] not in vowels):
                new_text = new_text.replace('an '+ target_word, 'a ' + chosen_candidate)
            else:
                new_text = new_text.replace(target_word, chosen_candidate)
            word_replace_dict[target_word] = chosen_candidate
#             print(candidates, "\n", prediction)
    return new_text

