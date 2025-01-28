####### packages
import pandas as pd
import gensim
import gensim.corpora as corpora
import logging
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric
from tqdm import tqdm
from gensim import models
import re
import matplotlib.pyplot as plt

####### functions
#https://tedboy.github.io/nlps/_modules/gensim/corpora/dictionary.html#Dictionary.filter_extremes
# change gensim function in order to use also fraction of total corpus size in no_below


def gensim_filter_extremes(no_below, no_above, keep_n, id2word):

    import numpy as np
    """
    adjusted gensim function to make it similar to sklearn
    
    id2word: dict
    no_below: fraction of total corpus
    no_above: "-"
    
    Filter out tokens that appear in

    1. less than `no_below` documents (fraction of total corpus)
    2. more than `no_above` documents (fraction of total corpus size 
    3. after (1) and (2), keep only the first `keep_n` most frequent tokens (or
       keep all if `None`).

    After the pruning, shrink resulting gaps in word ids.

    **Note**: Due to the gap shrinking, the same word may have a different
    word id before and after the call to this function!
    """
    no_above_abs = int(no_above * id2word.num_docs)  # convert fractional threshold to absolute threshold #sklearn (?) schauen, ob ich hier auch aufrunden sollte
    no_below_abs = int(np.ceil(no_below * id2word.num_docs)) #da in sklearn aufgerundet wird np.ceil nutzen

    # determine which tokens to keep
    good_ids = (
        v for v in iter(id2word.token2id.values())
        if no_below_abs <= id2word.dfs.get(v, 0) <= no_above_abs)
    good_ids = sorted(good_ids, key=id2word.dfs.get, reverse=True)
    if keep_n is not None:
        good_ids = good_ids[:keep_n]
    bad_words = [(id2word[id], id2word.dfs.get(id, 0)) for id in set(id2word).difference(good_ids)]

    # do the actual filtering, then rebuild dictionary to remove gaps in ids
    id2word.filter_tokens(good_ids=good_ids)
    return id2word




df_heise_article_preprocessed=pd.read_pickle('.../data/data_heise_preprocessed.pkl')
id2word = corpora.Dictionary(df_heise_article_preprocessed['custom_bigram_token'])
id2word_filtered = gensim_filter_extremes(no_below = 0.005, no_above = 0.65, keep_n = None, id2word = id2word)

texts = df_heise_article_preprocessed['custom_bigram_token']

corpus = [id2word_filtered.doc2bow(text) for text in texts]

id2word_filtered.save('id2word-custom_bigram_token.dict')  # save dict 
corpora.MmCorpus.serialize('corpus-custom_bigram_token.mm', corpus)  # save corpus

#load corpus and dict
id2word = corpora.Dictionary.load('id2word-custom_bigram_token.dict') #filtered corpus
corpus = corpora.MmCorpus('corpus-custom_bigram_token.mm')
texts = df_heise_article_preprocessed['custom_bigram_token']

#https://www.meganstodel.com/posts/callbacks/

logging.basicConfig(filename='model_callback_cbt_no_below_0005_no_above_065_15epochs_part2.log',
                    format="%(asctime)s:%(levelname)s:%(message)s",
            gi        level=logging.NOTSET)

perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
convergence_logger = ConvergenceMetric(logger='shell')
coherence_cv_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence = 'c_v', texts = texts)

print(corpus) #Sparsität: 16809930/(181402*4681) =  0,98020365 also zu 98% leere Dokument-Term-Matrix 

# List of the different topics to try
topics = [80,90,100,110,120,130,140,150,200,250] #50,60,70

passes = 15

for topic in topics:
   
    # indicate new model
    logging.debug(f'Start of model: {topic} topics')

    # lda model
    model = gensim.models.ldamodel.LdaModel(corpus=corpus,
             id2word=id2word,
             num_topics=topic,
             update_every=10,
             chunksize = 10000,                        
             passes=passes,
            alpha ='auto',
             per_word_topics = True,                        
            random_state=100,
            callbacks=[convergence_logger, perplexity_logger, coherence_cv_logger])

    # indicate end in logtext
    logging.debug(f'End of model: {topic} topics')

    # Save
    if not os.path.exists(f"lda_{topic}t_cbt_no_below_0005_no_above_065/"):
        os.makedirs(f"lda_{topic}t_cbt_no_below_0005_no_above_065/")#ändern in t_cbt

    model.save(f"lda_{topic}t_cbt_no_below_0005_no_above_065/lda_{topic}t_cbt_no_below_0005_no_above_065.model") 

topics = [50, 60, 70,80, 90, 100, 110, 120, 130, 140, 150, 200, 250]


all_metrics = pd.DataFrame()

for topic in tqdm(topics):
    model = models.ldamodel.LdaModel.load(f"lda_{topic}t_cbt_no_below_0005_no_above_065/lda_{topic}t_cbt_no_below_0005_no_above_065.model")
    df = pd.DataFrame.from_dict(model.metrics)
    df['topic'] = topic
    all_metrics = pd.concat([all_metrics, df])

all_metrics.reset_index(inplace = True)
all_metrics.rename(columns = {'index':'epoch'}, inplace = True)

#visualize cv-scores
for topic in topics:
    plt.plot(all_metrics[all_metrics['topic']==topic][['epoch']], all_metrics[all_metrics['topic']==topic][['Coherence']], color = 'red', marker ='o')
    plt.title(f'Coherence Scores of LDA Topic Modells with {topic} topics and various epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Coherence c_v', fontsize=14)
    plt.grid(True)
    plt.show()

all_metrics[all_metrics['epoch']==14].groupby('topic').mean() #Coherence Maximal bei 60 Topics (cv)

plt.plot(all_metrics[all_metrics['epoch']==14].groupby('topic').mean().index, all_metrics[all_metrics['epoch']==14].groupby('topic').mean()['Coherence'], color='red', marker='o')
plt.title('Coherence Scores of LDA Topic Model with various topic numbers', fontsize=14)
plt.xlabel('Topic', fontsize=14)
plt.ylabel('Coherence c_v', fontsize=14)
plt.grid(True)
plt.show()
# take second highest peak k=120 topics

#check final model's (with k=120 topics) parameters
topics = 120
ldamodel = models.ldamodel.LdaModel.load(f"lda_{topics}t_cbt_no_below_0005_no_above_065/lda_{topics}t_cbt_no_below_0005_no_above_065.model")

print(ldamodel)
print(f"Number of topics: {ldamodel.num_topics}")
print(f"Alpha (document-topic density): {ldamodel.alpha}")
print(f"Eta (topic-word density): {ldamodel.eta}")
print(f"Decay: {ldamodel.decay}")
print(f"Offset: {ldamodel.offset}")
print(f"Iterations: {ldamodel.iterations}")
print(f"Random state: {ldamodel.random_state}")