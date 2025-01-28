#pvtm
#python version: 3.7.10

import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df_data = pd.read_csv('../data/data_heise_preprocessed.csv',index_col = 'Unnamed: 0')

#train pvtm analogous to LDA
from tqdm import tqdm
from pvtm import pvtm
p = pvtm.PVTM(df_data.space_custom_bigram_token)
_ = p.preprocess(min_df = 0.005, max_df =0.65)

component = 120

p.fit(vector_size = 100, # dimensionality of the feature vectors (Doc2Vec)
     n_components = component, # number of Gaussian mixture components, i.e. Topics (GMM)
     epochs=15)
savepath = f'{component}_15epochs'
p.save(f"{savepath}/pvtm_model_{p.gmm.n_components}")

#extract topic weights
df_data['pvtm_gmm_probas'] = [p.gmm.predict_proba(p.doc_vectors[i].reshape(1,-1))[0] for i in range(df_data.shape[0])]
probas = df_data.pvtm_gmm_probas.apply(pd.Series)# Dokument-ID:row, TOPIC-ID:column
probas['date_preprocessed'] = df_data.date_preprocessed

#save dok-topic-propabilities
topics = 120
if not os.path.exists(f"PVTM_{topics}_Topics_topicweights/"):
    os.makedirs(f"PVTM_{topics}_Topics_topicweights/")

probas.to_csv(f"PVTM_{topics}_Topics_topicweights/pvtm_original_topicweights.csv")
probas.to_pickle(f"PVTM_{topics}_Topics_topicweights/pvtm_original_topicweights.pkl")


#pvtm wordclouds (built-in)

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import numpy as np
import os

#alle in eine Datei

if not os.path.exists(f"PVTM_{topics}Topics_Wordclouds/"):
    os.makedirs(f"PVTM_{topics}Topics_Wordclouds/")
pdf = matplotlib.backends.backend_pdf.PdfPages(f"PVTM_{topics}Topics_Wordclouds/allpvtmTopics.pdf")

for t in range(p.gmm.n_components):
    plt.figure()
    text = p.top_topic_center_words.iloc[t,:50]
    text = " ".join(text)
    
    plt.imshow(WordCloud(background_color='white',
                  width=2500,
                  height=2500,
                  scale=50,
                  max_words=200,
                  #colormap='tab10',
                  mask = shape).generate(text))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    pdf.savefig(orientation = 'landscape', dpi = 450)
    plt.show()

pdf.close()





