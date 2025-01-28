import pandas as pd
import joblib
import gensim
import gensim.corpora as corpora

pvtmmodel = joblib.load('120_15epochs\pvtm_model_120')
df_topic_doc = pd.read_pickle("PVTM_120_Topics_topicweights/pvtm_original_topicweights.pkl")
df_topic_doc.drop(columns = 'date_preprocessed', inplace = True)
df_topic_doc['document'] = pvtmmodel.documents #documents from trained pvtm-model (convert to list)
df_topic_doc['document'] = df_topic_doc.document.str.split(' ')


corpus = [id2word.doc2bow(text) for text in df_topic_doc['document']]
id2word = corpora.Dictionary(df_topic_doc['document'])

#create empty term-document matrix
df_term_document = pd.DataFrame(columns = id2word.values(), index=range(0,len(df_topic_doc['document'])))

dokument_term_freq = [[(id2word[id], freq) for id, freq in cp] for cp in corpus]

#k is document-id
for k, row in df_term_document.iterrows():
    words= [x[0] for x in dokument_term_freq[k]]
    values= [x[1] for x in dokument_term_freq[k]]
    df_term_document.loc[row.name, words] = values

df_term_document = df_term_document.fillna(0) #dataframe of term document freqs

df_topic_term = df_topic_doc.drop(columns = 'document').transpose().dot(df_term_document)

#df_topic_term #normalize rows, so the elements sum up to 1

for k, row in df_topic_term.iterrows():
    normalizer = 1 / float( sum(row) )
    normalized = [x * normalizer for x in row]
    df_topic_term.loc[row.name] = normalized

if not os.path.exists("PVTM_Topic_Term_Matrix/"):
        os.makedirs("PVTM_Topic_Term_Matrix/")
        
df_topic_term.to_csv("PVTM_Topic_Term_Matrix/method_1_topic_term_weights.csv") #method1 is Topic-Dokument Matrix * Dokument-Term Matrix, result is normalized
df_topic_term.to_pickle("PVTM_Topic_Term_Matrix/method_1_topic_term_weights.pkl")


#new word clouds with method1

# in einer pdf-Datei
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import numpy as np
import os

#Für kreisrunde Wordclouds (default Einstellungen erzeugen unschöne WordClouds)
x,y = np.ogrid[:300, :300]
shape = (x-150)**2 + (y-150)**2 > 130**2
shape = 255*shape.astype(int)

pdf = matplotlib.backends.backend_pdf.PdfPages("PVTM_Topic_Term_Matrix/method_1_allpvtmTopics.pdf")


for t in range(0,pvtmmodel.gmm.n_components):
    plt.figure()
    plt.imshow(WordCloud(background_color='white',
                  width=2500,
                  height=2500,
                  scale=50,
                  max_words=200,
                  #colormap='tab10',
                  mask = shape).fit_words(dict(df_topic_term.iloc[t].sort_values(ascending=False)[0:50])))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    pdf.savefig(orientation = 'landscape', dpi = 450)

pdf.close()


#m2 dokument freq noch durch Anzahl der Token im jeweiligen Dokument teilen
#m2 (DTM is normalized by doclength)

numberoftokens = []
for i in df_topic_doc['document']:
    numberoftokens.append(len(i))

df_term_document_normalized_by_doclength = df_term_document.divide(numberoftokens,axis =0)

df_topic_term_normalized_by_doclength = df_topic_doc.drop(columns = 'document').transpose().dot(df_term_document_normalized_by_doclength)

#rowsums should sum up to 1
for k, row in df_topic_term_normalized_by_doclength.iterrows():
    normalizer = 1 / float( sum(row) )
    normalized = [x * normalizer for x in row]
    df_topic_term_normalized_by_doclength.loc[row.name] = normalized

df_topic_term_normalized_by_doclength.to_csv("PVTM_Topic_Term_Matrix/method_2_topic_term_weights.csv") #method2 is Topic-Dokument Matrix * Dokument-Term Matrix normalized by doclength, result is normalized so that rowsums sum up to 1
df_topic_term_normalized_by_doclength.to_pickle("PVTM_Topic_Term_Matrix/method_2_topic_term_weights.pkl")

#new word clouds with method2

# in einer pdf-Datei
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import numpy as np
import os

#Für kreisrunde Wordclouds (default Einstellungen erzeugen unschöne WordClouds)
x,y = np.ogrid[:300, :300]
shape = (x-150)**2 + (y-150)**2 > 130**2
shape = 255*shape.astype(int)

pdf = matplotlib.backends.backend_pdf.PdfPages("PVTM_Topic_Term_Matrix/method_2_allpvtmTopics.pdf")


for t in range(0,pvtmmodel.gmm.n_components):
    plt.figure()
    plt.imshow(WordCloud(background_color='white',
                  width=2500,
                  height=2500,
                  scale=50,
                  max_words=200,
                  #colormap='tab10',
                  mask = shape).fit_words(dict(df_topic_term_normalized_by_doclength.iloc[t].sort_values(ascending=False)[0:50])))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    pdf.savefig(orientation = 'landscape', dpi = 450)

pdf.close()

