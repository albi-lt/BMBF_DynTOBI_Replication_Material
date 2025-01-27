# Branch for web scraping and preprocessing heise articles

heisescraper.py results from al-01-dev_heise_scraper (use heisescraper.py in order to scrap articles!)

al-02-eda_and_preprocessing_heise_articles -> exploratory data analysis and preprocessing,  uses functions from text_preprocessing.py

____
How to generate requirements txt-file:

(dyntobi) C:\Users\Albina\JLUbox\DynTOBI\2021_aktuelle_Entwicklungen\Webscraper>jupyter nbconvert --output-dir="./reqs" --to script al-01-dev_heise_scraper.ipynb
[NbConvertApp] Converting notebook al-01-dev_heise_scraper.ipynb to script
[NbConvertApp] Writing 8734 bytes to reqs\al-01-dev_heise_scraper.py

(dyntobi) C:\Users\Albina\JLUbox\DynTOBI\2021_aktuelle_Entwicklungen\Webscraper>cd reqs

(dyntobi) C:\Users\Albina\JLUbox\DynTOBI\2021_aktuelle_Entwicklungen\Webscraper\reqs>pipreqs
INFO: Successfully saved requirements file in C:\Users\Albina\JLUbox\DynTOBI\2021_aktuelle_Entwicklungen\Webscraper\reqs\requirements.txt
____
