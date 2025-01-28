import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import scipy.stats as stats


#processing of topic weights
df_topicweights = pd.read_pickle(f"PVTM_{topics}_Topics_topicweights/pvtm_original_topicweights.pkl")
df_topicweights['date_preprocessed'] = pd.to_datetime(df_topicweights.date_preprocessed,format = '%Y-%m-%d')
df_topicweights = df_topicweights.set_index('date_preprocessed')
df_grouped_monthly = df_topicweights.groupby(pd.Grouper(freq = "M")).mean().fillna(0)

df_grouped_monthly.to_pickle(f"PVTM_{topics}_Topics_topicweights/pvtm_monthly_grp_topicweights.pkl")
df_grouped_monthly.to_csv(f"PVTM_{topics}_Topics_topicweights/pvtm_monthly_grp_topicweights.csv")


index_monthly = pd.period_range(start="1996-04-01", end ="2021-10-01", freq = "M")
df_grouped_monthly.set_index(index_monthly, inplace = True)

df_cycle_monthly = pd.DataFrame()
df_trend_monthly = pd.DataFrame()

if not os.path.exists(f"PVTM_{topics}_Topics_topicweights/pvtm_hp_filtered"):
    os.makedirs(f"PVTM_{topics}_Topics_topicweights/pvtm_hp_filtered")

for i in range(0,len(df_grouped_monthly.columns)):
    cycle, trend = sm.tsa.filters.hpfilter(df_grouped_monthly[i], 14400) #columns dtype is object and no string otherwise: str(i)
    decomp = df_grouped_monthly[[i]]
    decomp["cycle_"+str(i)] = cycle
    decomp["trend_"+str(i)] = trend

    df_cycle_monthly['cycle_'+str(i)] = cycle
    df_trend_monthly['trend_'+str(i)] = trend
    
    fig, ax = plt.subplots(figsize= (30,20))
    decomp[[i, "trend_"+str(i),'cycle_'+str(i)]]["1996-04-01":].plot(ax=ax, fontsize = 16)
    ax.grid()
    ax.set_xlabel('year', fontsize = 25)
    ax.set_ylabel('arithm. mean PVTM-Topic weight/month and smoothing with HP filter lambda=14400', fontsize = 25)
    ax.set_title('Time trend of PVTM-Topic weights on a monthly basis by Topic:' + str(i) + ' Smoothing with HP-Filter lambda=14400' ,fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.savefig(f"PVTM_{topics}_Topics_topicweights/pvtm_hp_filtered/pvtm_monthly_hp_filter_topic" + str(i)+".pdf")
    plt.show()

#alle Plots in eine Datei

df_cycle_monthly = pd.DataFrame()
df_trend_monthly = pd.DataFrame()

if not os.path.exists(f"PVTM_{topics}_Topics_topicweights/pvtm_hp_filtered"):
    os.makedirs(f"PVTM_{topics}_Topics_topicweights/pvtm_hp_filtered")

pdf = matplotlib.backends.backend_pdf.PdfPages(f"PVTM_{topics}_Topics_topicweights/pvtm_hp_filtered/all_topics_pvtm_monthly_hp_filter.pdf")

for i in range(0,len(df_grouped_monthly.columns)):
    cycle, trend = sm.tsa.filters.hpfilter(df_grouped_monthly[i], 14400) #columns dtype is object and no string otherwise: str(i)
    decomp = df_grouped_monthly[[i]]
    decomp["cycle_"+str(i)] = cycle
    decomp["trend_"+str(i)] = trend

    df_cycle_monthly['cycle_'+str(i)] = cycle
    df_trend_monthly['trend_'+str(i)] = trend
    
    fig, ax = plt.subplots(figsize= (30,20))
    decomp[[i, "trend_"+str(i),'cycle_'+str(i)]]["1996-04-01":].plot(ax=ax, fontsize = 16)
    ax.grid()
    ax.set_xlabel('year', fontsize = 25)
    ax.set_ylabel('arithm. mean PVTM-Topic weight/month and smoothing with HP filter lambda=14400', fontsize = 25)
    ax.set_title('Time trend of PVTM-Topic weights on a monthly basis by Topic:' + str(i) + ' Smoothing with HP-Filter lambda=14400' ,fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    pdf.savefig(orientation = 'landscape', dpi = 450)

pdf.close()

#save trend component
df_trend_monthly.to_csv(f"PVTM_{topics}_Topics_topicweights/pvtm_monthly_grouped_hp_filtered_trend_component_topicweights.csv")
df_trend_monthly.to_pickle(f"PVTM_{topics}_Topics_topicweights/pvtm_monthly_grouped_hp_filtered_trend_component_topicweights.csv")

topicweights_trend_zscore = df_trend_monthly.apply(stats.zscore)

topicweights_trend_zscore.to_csv(f"PVTM_{topics}_Topics_topicweights/pvtm_monthly_grouped_hp_filtered_trend_component_z_std_topicweights.csv")
topicweights_trend_zscore.to_pickle(f"PVTM_{topics}_Topics_topicweights/pvtm_monthly_grouped_hp_filtered_trend_component_z_std_topicweights.pkl")

#visualise smoothed series
if not os.path.exists(f"PVTM_{topics}_Topics_topicweights/pvtm_hp_filtered_z_std"):
    os.makedirs(f"PVTM_{topics}_Topics_topicweights/pvtm_hp_filtered_z_std")

pdf = matplotlib.backends.backend_pdf.PdfPages(f"PVTM_{topics}_Topics_topicweights/pvtm_hp_filtered_z_std/all_topics_pvtm_z_std_monthly_hp_filter.pdf")
    
for i in range(0,len(topicweights_trend_zscore.columns)):
    fig, ax = plt.subplots(figsize= (30,20))
    topicweights_trend_zscore[["trend_"+str(i)]]["1996-04-01":].plot(ax=ax, fontsize = 16)
    ax.grid()
    ax.set_xlabel('year', fontsize = 25)
    ax.set_ylabel('arithm. mean PVTM Topic weight/month and smoothing with HP filter lambda=14400 and z-standardization', fontsize = 25)
    ax.set_title('Time trend of PVTM Topic weights on a monthly basis by Topic:' + str(i) + ' Smoothing with HP-Filter lambda=14400 and z-standardization' ,fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    pdf.savefig(orientation = 'landscape', dpi = 450)

pdf.close()