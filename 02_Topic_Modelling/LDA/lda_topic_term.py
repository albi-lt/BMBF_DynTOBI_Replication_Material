#search topics by term and create topic-term matrix

import pandas as pd
from gensim.models import LdaModel
from gensim.test.utils import common_dictionary, common_corpus
import gensim.corpora as corpora
import os

##functions
def get_topic_term_df(ldamodel):
    """
    load gensim ldamodel with models.ldamodel.LdaModel.load('path') and pass it to the function
    function returns dataframe with topic number (row) and term (column) and corresponding topic-term-weight
    """
    df = pd.DataFrame(ldamodel.get_topics(), columns=ldamodel.id2word.values(), index=[f'topic {i}' for i in range(ldamodel.num_topics)])
    return df

def get_dominant_topic(ldamodel, corpus):
    """
    Dieser Funktion wird das gespeicherte LDA-Modell und der Korpus (Format wie Input für zu traininerende LDA-Modelle)
    übergeben. Zurückgegeben wird eine Liste der dominanten Topics der jeweiligen Dokumente des Korpus.
    """
    topic_dist_lda = ldamodel.get_document_topics(corpus)
    
    dominant_topic = []
    for doc in range(0,len(topic_dist_lda)):
        topic_id_anteil = topic_dist_lda[doc]
        topic_id_anteil.sort(key=lambda tup: tup[1], reverse = True)
        dominant_topic.append(topic_id_anteil[0][0])
    return dominant_topic


##

#load best lda_model after additional human judging
topics = 120

ldamodel = models.ldamodel.LdaModel.load(f"lda_{topics}t_cbt_no_below_0005_no_above_065/lda_{topics}t_cbt_no_below_0005_no_above_065.model") 

pd.DataFrame(ldamodel.get_topics(), columns=ldamodel.id2word.values(), index=[f'topic {i}' for i in range(ldamodel.num_topics)])['wikipedia'].describe()

term = 'wikipedia'
pd.DataFrame(ldamodel.get_topics(), columns=ldamodel.id2word.values(), index=[f'topic {i}' for i in range(ldamodel.num_topics)])[term][pd.DataFrame(ldamodel.get_topics(), columns=ldamodel.id2word.values(), index=[f'topic {i}' for i in range(ldamodel.num_topics)])[term]==pd.DataFrame(ldamodel.get_topics(), columns=ldamodel.id2word.values(), index=[f'topic {i}' for i in range(ldamodel.num_topics)])[term].max()]
#analyze output for different terms

#load
id2word = corpora.Dictionary.load('../02_Topic_Modelling/id2word-custom_bigram_token.dict')
corpus = corpora.MmCorpus('../02_Topic_Modelling/corpus-custom_bigram_token.mm')


df_topic_term_weights = get_topic_term_df(ldamodel)
if not os.path.exists("LDA_120_Topic_Term_Matrix/"):
        os.makedirs("LDA_120_Topic_Term_Matrix/")
        
df_topic_term_weights.to_csv("LDA_120_Topic_Term_Matrix/lda_topic_term_weights.csv") 
df_topic_term_weights.to_pickle("LDA_120_Topic_Term_Matrix/lda_topic_term_weights.pkl")
df_topic_term_weights

#dominant topics
dominante_topics = get_dominant_topic(ldamodel=lda_model, corpus=corpus) #go rather for topic-term-weights than for dominant topic per document