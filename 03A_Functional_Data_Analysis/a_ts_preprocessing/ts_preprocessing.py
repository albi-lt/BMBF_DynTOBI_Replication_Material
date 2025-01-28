from gensim import models
import gensim.corpora as corpora
import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#load all data

topics = 120

ldamodel = models.ldamodel.LdaModel.load(f"../02_Topic_Modelling/lda_{topics}t_cbt_no_below_0005_no_above_065/lda_{topics}t_cbt_no_below_0005_no_above_065.model")

id2word = corpora.Dictionary.load('../02_Topic_Modelling/id2word-custom_bigram_token.dict')
corpus = corpora.MmCorpus('../02_Topic_Modelling/corpus-custom_bigram_token.mm')

df_heise_article_preprocessed=pd.read_pickle('../01_Webscraper/data/data_heise_preprocessed.pkl')

#extracting topic weights
topic_dist_lda = ldamodel.get_document_topics(corpus, minimum_probability = 0)

topicweights = []
for i in range(0,len(topic_dist_lda)):
    topicweights.append(topic_dist_lda[i])

dct = [dict(topicweights[i]) for i in range(0,len(topicweights))]
df = pd.DataFrame.from_dict(dct)

cols=df.columns.tolist()
cols.sort()

df_topicweights=df[cols]
df_topicweights = df_topicweights.fillna(0)

if len(cols) == len(range(0,topics)):
    print('Everything is fine.')
else:
    difference = set(list(range(0,topics))).difference(set(cols))
    print(f'Extracting Topic Weights was not successful.Topics {list(difference)} not in dataframe.')

df_topicweights['date_preprocessed'] = df_heise_article_preprocessed.date_preprocessed


if not os.path.exists(f"{topics}_Topics_topicweights/"):
    os.makedirs(f"{topics}_Topics_topicweights/")
df_topicweights.to_csv(f"{topics}_Topics_topicweights/original_topicweights.csv")
df_topicweights.to_pickle(f"{topics}_Topics_topicweights/original_topicweights.pkl")

# process topicweights further
df_topicweights = pd.read_pickle(f"{topics}_Topics_topicweights/original_topicweights.pkl")

df_topicweights = df_topicweights.set_index('date_preprocessed')
df_grouped_monthly = df_topicweights.groupby(pd.Grouper(freq ='M')).mean().fillna(0)

df_grouped_monthly.to_pickle(f"{topics}_Topics_topicweights/monthly_grp_topicweights.pkl")
df_grouped_monthly.to_csv(f"{topics}_Topics_topicweights/monthly_grp_topicweights.csv")

# micro deflections are not of interest -> smooth series
#Jede Spalte von 0 bis 119 via HP-Filter glätten
#https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html

index_monthly = pd.period_range(start='1996-04-01', end='2021-10-01', freq='M')
df_grouped_monthly.set_index(index_monthly, inplace = True)#Jede Spalte von 0 bis 119 via HP-Filter glätten

df_cycle_monthly = pd.DataFrame()
df_trend_monthly = pd.DataFrame()

if not os.path.exists(f"{topics}_Topics_topicweights/hp_filtered"):
    os.makedirs(f"{topics}_Topics_topicweights/hp_filtered")

pdf = matplotlib.backends.backend_pdf.PdfPages(f"{topics}_Topics_topicweights/hp_filtered/all_topics_monthly_hp_filter.pdf")

for i in range(0,len(df_grouped_monthly.columns)):
    cycle, trend = sm.tsa.filters.hpfilter(df_grouped_monthly[i], 14400) #columns dtype is object and no string otherwise: str(i) HP-filter with lambda = 14400
    decomp = df_grouped_monthly[[i]]
    decomp["cycle_"+str(i)] = cycle
    decomp["trend_"+str(i)] = trend

    df_cycle_monthly['cycle_'+str(i)] = cycle
    df_trend_monthly['trend_'+str(i)] = trend
    
    fig, ax = plt.subplots(figsize= (30,20))
    decomp[[i, "trend_"+str(i),'cycle_'+str(i)]]["1996-04-01":].plot(ax=ax, fontsize = 16)
    ax.grid()
    ax.set_xlabel('year', fontsize = 25)
    ax.set_ylabel('arithm. mean Topic weight/month and smoothing with HP filter lambda=14400', fontsize = 25)
    ax.set_title('Time trend of Topic weights on a monthly basis by Topic:' + str(i) + ' Smoothing with HP-Filter lambda=14400' ,fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    pdf.savefig(orientation = 'landscape', dpi = 450)

pdf.close()

df_trend_monthly.to_csv(f"{topics}_Topics_topicweights/monthly_grouped_hp_filtered_trend_component_topicweights.csv")
df_trend_monthly.to_pickle(f"{topics}_Topics_topicweights/monthly_grouped_hp_filtered_trend_component_topicweights.pkl")

#z-standardization
topicweights_trend_zscore = df_trend_monthly.apply(stats.zscore)

topicweights_trend_zscore.to_csv(f"{topics}_Topics_topicweights/monthly_grouped_hp_filtered_trend_component_z_std_topicweights.csv")
topicweights_trend_zscore.to_pickle(f"{topics}_Topics_topicweights/monthly_grouped_hp_filtered_trend_component_z_std_topicweights.pkl")

if not os.path.exists(f"{topics}_Topics_topicweights/hp_filtered_z_std"):
    os.makedirs(f"{topics}_Topics_topicweights/hp_filtered_z_std")

pdf = matplotlib.backends.backend_pdf.PdfPages(f"{topics}_Topics_topicweights/hp_filtered_z_std/all_topics_z_std_monthly_hp_filter.pdf")
    
for i in range(0,len(topicweights_trend_zscore.columns)):
    fig, ax = plt.subplots(figsize= (30,20))
    topicweights_trend_zscore[["trend_"+str(i)]]["1996-04-01":].plot(ax=ax, fontsize = 16)
    ax.grid()
    ax.set_xlabel('year', fontsize = 25)
    ax.set_ylabel('arithm. mean Topic weight/month and smoothing with HP filter lambda=14400 and z-standardization', fontsize = 25)
    ax.set_title('Time trend of Topic weights on a monthly basis by Topic:' + str(i) + ' Smoothing with HP-Filter lambda=14400 and z-standardization' ,fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    pdf.savefig(orientation = 'landscape', dpi = 450)

pdf.close()