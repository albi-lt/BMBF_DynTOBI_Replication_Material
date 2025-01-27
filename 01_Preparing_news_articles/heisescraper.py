import requests 
import bs4 as bs
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urljoin

def get_soup(quote_page):
    page = requests.get(quote_page)
    soup = BeautifulSoup(page.text, 'html.parser') 
    return page, soup

path = 'C:/Users/Albina/JLUbox/DynTOBI/2021_aktuelle_Entwicklungen/Webscraper/' 

## Download all relevant links from all archive pages
years = list(range(1996,2022))
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

dfs = []
for year in years:
    print(year)
    for month in months:
        print(month)
        url = 'https://www.heise.de/newsticker/archiv/{}/{}/'.format(year, month)
        page, soup = get_soup(url)
        
        url_list, title_list, brand_list = [],[],[]
        for archiv in soup.find_all('a',{'class':'a-article-teaser__link archive__link'}, href=True):
            url_list.append(urljoin(url, archiv.attrs['href']))
            title_list.append(archiv.attrs['title'].strip())
    
            brand = archiv.select_one('.a-article-branding')
            if brand:
                brand_list.append(brand.text.strip())
            else:
                brand_list.append('newsticker')  
        
        df = pd.DataFrame([title_list,url_list,brand_list]).T
        df.columns = ["title", "url", "brand"]
        dfs.append(df)
out = pd.concat(dfs)
out.reset_index(inplace=True) #in order to get consecutive index column
out.drop('index',axis=1, inplace = True)

# create folder
os.makedirs(path + 'heise_archiv/')

# save urls
out.to_csv(path + 'heise_archiv/urls.csv',encoding='utf-8-sig')

out_selected_brands = out[out['brand'].isin(["newsticker","c't Magazin","MIT Technology Review"])].reset_index().drop('index',axis=1) #consider only newsticker, 'c't and MIT Technology Review; e.g. category Mac & i would overrepresent Apple
out_selected_brands.to_csv('heise_archiv/urls_selected_brands.csv',encoding='utf-8-sig')

out_selected_brands = pd.read_csv(path + 'heise_archiv/urls_selected_brands.csv',encoding='utf-8-sig', index_col = 'Unnamed: 0')


import time
import justext
import numpy as np
import os

frame = out_selected_brands.copy()
print(out_selected_brands.shape)
errorrate=0

#remove errors.txt file
if os.path.exists(path + 'heise_archiv/errors.txt'):
    os.remove(path + 'heise_archiv/errors.txt')
    print("The file has been deleted successfully")

for i,row in frame.iterrows():
    
    print(i)

    try:
        print(row.url)
        response,soup = get_soup(row.url)
        paragraphs = justext.justext(response.content, justext.get_stoplist("German"))
        b = [paragraph.text.replace('\n',' ') for paragraph in paragraphs if not paragraph.is_boilerplate]
        text = "".join(b)
        if text == '' : 
            print(i, row.url, 'nope')

        author = soup.find_all('meta',{'name':'author'})[0]['content']
        description = soup.find_all('meta',{'property':'og:description'})[0]['content']
        date = soup.find_all('meta',{'name':'date'})[0]['content']

        tmpdf = pd.DataFrame([text, author, description, date, row.title, row.url, row.brand])
        
        mode, header= ('w', True) if i <1 else ("a", False)
        tmpdf.T.to_csv(path + 'heise_archiv/heise_archiv.csv',encoding='utf-8', mode=mode, header=header)
        
    except Exception as e: 
        print('ERROR',e)
        print(i, row.url)
        errorrate+=1
        print('Current Errorrate:' + str(errorrate/i))
        with open(path + 'heise_archiv/' + 'errors.txt', 'a') as f: #stream urls in txt-file which couldn't be scrapped
            f.write(row.url)
            f.write('\n')
            

#rescrape urls from error txt-file 
with open(path + 'heise_archiv/' + 'errors.txt', 'r') as f: #read file
        error_list = []   
        for line in f:
            error_list.append(line.strip())

# errors are mostly caused by missing author names
for i, url in enumerate(error_list):
    print(i)
    
    try:
        print(url)
        response,soup = get_soup(url)
        paragraphs = justext.justext(response.content, justext.get_stoplist("German"))
        b = [paragraph.text.replace('\n',' ') for paragraph in paragraphs if not paragraph.is_boilerplate]
        text = "".join(b)
        if text == '' : 
            print(i, url, 'no text')

        author = eval(soup.find('script',attrs= {'type':'application/ld+json'}).text.strip())['author']['name']
        description = soup.find_all('meta',{'property':'og:description'})[0]['content']
        date = soup.find_all('meta',{'name':'date'})[0]['content']

        tmpdf_errors = pd.DataFrame([text, author, description, date, url])
        mode, header= ('w', True) if i <1 else ("a", False)
        
        tmpdf_errors.T.to_csv(path + 'heise_archiv/heise_archiv_errors.csv',encoding='utf-8', mode=mode, header=header)
        
    except Exception as e: 
        print('ERROR',e)
        with open(path + 'heise_archiv/' + 'errors_errors.txt', 'a') as f: #stream urls in another txt-file which couldn't be scrapped
            f.write(url)
            f.write('\n')
            
#preprocess and concatenate dataframes
#DF1
heise_archiv_errors = pd.read_csv(path + 'heise_archiv\heise_archiv_errors.csv', index_col = 'Unnamed: 0')
heise_archiv_errors.columns = ['text','author','description','date','url']
out_selected_brands = pd.read_csv(path + 'heise_archiv/urls_selected_brands.csv',encoding='utf-8-sig', index_col = 'Unnamed: 0')
heise_archiv_errors_preprocessed = pd.merge(heise_archiv_errors,out_selected_brands, how = 'inner', on = 'url').drop_duplicates()

dates=[]
for i in range(0,len(heise_archiv_errors_preprocessed)):
    dates.append(pd.to_datetime(heise_archiv_errors_preprocessed.iloc[i].date.split('T')[0], format = '%Y-%m-%d'))
    
heise_archiv_errors_preprocessed['date_preprocessed'] = dates    

heise_archiv_errors_preprocessed.sort_values(by = 'date_preprocessed', inplace = True) 

#DF2
heise_archiv = pd.read_csv(path + 'heise_archiv\heise_archiv.csv', index_col = 'Unnamed: 0')
heise_archiv_preprocessed = heise_archiv.drop_duplicates()
heise_archiv_preprocessed.columns = ['text','author','description','date','title','url','brand']
dates=[]
for i in range(0,len(heise_archiv_preprocessed)):
    dates.append(pd.to_datetime(heise_archiv_preprocessed.iloc[i].date.split('T')[0], format = '%Y-%m-%d'))
    
heise_archiv_preprocessed['date_preprocessed'] = dates    

heise_archiv_preprocessed.sort_values(by = 'date_preprocessed', inplace = True)
                                      

df_heise_article = pd.concat([heise_archiv_preprocessed, heise_archiv_errors_preprocessed], ignore_index=True).sort_values(by = 'date_preprocessed').dropna(subset = ['text']) #concatenate dataframes and drop empty articles
df_heise_article.reset_index(drop = 'index', inplace = True)

df_heise_article.to_csv(path + 'heise_archiv/heise_article.csv',encoding='utf-8')
df_heise_article.to_pickle(path + 'heise_archiv/heise_article.pkl')
