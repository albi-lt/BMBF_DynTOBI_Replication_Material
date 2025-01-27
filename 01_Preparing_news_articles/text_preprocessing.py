def convert_umlauts(dataframe,textcolumn):
    """
    This function must be passed a data frame and a text column.
    The German umlauts are then converted to a dot-free notation.
    """
    dataframe[textcolumn].replace('Ä','AE',regex=True, inplace = True)
    dataframe[textcolumn].replace('ä','ae',regex=True, inplace = True)
    dataframe[textcolumn].replace('Ü','UE',regex=True, inplace = True)
    dataframe[textcolumn].replace('ü','ue',regex=True, inplace = True)
    dataframe[textcolumn].replace('Ö','OE',regex=True, inplace = True)
    dataframe[textcolumn].replace('ö','oe',regex=True, inplace = True)
    dataframe[textcolumn].replace('ß','ss',regex=True, inplace = True)
    
    return dataframe
############
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True,max_len = 30))  #exclude tokens with one character and those with more than 30 characters


############
def count_tokens(dataframe, textcolumn, returncolumn):
    no_of_tokens = []
    for i in range(0,len(dataframe)):
        no_of_tokens.append(len(dataframe[textcolumn][i]))

    dataframe[returncolumn] = no_of_tokens
    return dataframe
############

def get_space_token(dataframe, tokenizedcolumn, new_columnname):
    space_tokens = []
    for i in dataframe[tokenizedcolumn]:
            a = ','.join(i)
            b= a.replace(',',' ')
            space_tokens.append(b)
    
    dataframe[new_columnname] = space_tokens
    return dataframe
###########

import collections
from collections import Counter 
import itertools
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

def get_token_statistics(dataframe,tokencolumn,spacetokencolumn,savepath):
    #Top tokens
    flat = itertools.chain.from_iterable(dataframe[tokencolumn])

    corpus = list(flat)
    mc = collections.Counter(corpus).most_common() #if integer is passed here, then e.g. top 1000 tokens are calculated

    list_mc_tokens = [list(ele) for ele in mc]
    list_mc_tokens_correct = list(itertools.chain.from_iterable(list_mc_tokens))
    del list_mc_tokens_correct[1::2] #delete every second element strating from index 1
    top_tokens = list_mc_tokens_correct
    table = list_mc_tokens
    df_top_tokens = pd.DataFrame(table)
    df_top_tokens.columns= ['Vocabulary','Frequency']
    
    #idf-scores
    v = TfidfVectorizer()
    x = v.fit_transform(dataframe[spacetokencolumn]) #tokens must be separated by a space and not by a comma
    
    v.vocabulary_ 
    
    feature_names = v.get_feature_names()
    idfwert = v.idf_

    df_idf = pd.DataFrame()
    df_idf['Vocabulary'] = feature_names
    df_idf['idf_score'] = idfwert
    
    df_token_statistics = pd.merge(df_top_tokens,df_idf, how = 'inner', on = 'Vocabulary')
    df_token_statistics.to_csv(savepath)
    return df_token_statistics  

############
from nltk.util import ngrams

#n = 2 bigram, n=3 trigram
def ngramconvert(dataframe,n,space_token, outputtoken):
    docs_ngram_tuples = dataframe[space_token].apply(lambda sentence: list(ngrams(sentence.split(), n)))
    preprocessed_bigram_list = []
    for i in range(0,len(dataframe)):
        preprocessed_bigram_list.append(list(map('_'.join, docs_ngram_tuples[i])))  
    
    dataframe[outputtoken] = preprocessed_bigram_list
    return dataframe
###########

from nltk.stem.snowball import SnowballStemmer
def snowballstem_tokens(dataframe, text_token_column):
    Stemmer=SnowballStemmer("german")

    stemmed_docs = []
    for i in dataframe[text_token_column]:
        stemmed_doc = []
        for j in i:
            stemmed_doc.append(Stemmer.stem(j))
        stemmed_docs.append(stemmed_doc)
    
    dataframe['stemmed_token'] = stemmed_docs
    return dataframe

###########

def create_custombigram(custom_bigramlist,dataframe,tokencolumn,outputcolumn):
    bigrams_set = set(custom_bigramlist)
    bigram_corpus = []
    for doc in dataframe[tokencolumn]:
        bigram_doc = []
        for j in range(len(doc)):
            if (j>0 and doc[j-1]+"_"+doc[j] in bigrams_set):
                bigram_doc.pop()
                bigram_doc.append(doc[j-1]+"_"+doc[j])
            else:
                bigram_doc.append(doc[j])

        bigram_corpus.append(bigram_doc)
        
    dataframe[outputcolumn] = bigram_corpus
    return dataframe
############

def remove_stopwords(texttoken,stopwordslist):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwordslist] for doc in texttoken]


def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]