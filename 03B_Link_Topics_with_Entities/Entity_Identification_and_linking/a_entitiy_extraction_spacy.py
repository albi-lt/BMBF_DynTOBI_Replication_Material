import pandas as pd
import spacy
from tqdm import tqdm
import pickle


def extract_ners(text, nlp):
    """
    Extract ners from text using spacy
    :param text: str. some text
    :param nlp: spacy nlp-object. e.g. nlp = spacy.load("de_core_news_lg")
    :return: list of lists. each inner list holds entity text, start_char, end_char and label_
    """
    return [[e.text, e.start_char, e.end_char, e.label_] for e in nlp(text).ents]


if __name__ == "__main__":

    tqdm.pandas()
    nlp = spacy.load("de_core_news_lg")
    #    heise = pd.read_csv(r"H:/2017-10-16_TOBI/scraping/2018-01-20_heise_archiv/input/heise_archiv.csv")

    with open('data/internal/data_heise_preprocessed_small.pkl', 'rb') as pickle_file:
        heise = pickle.load(pickle_file)

    # extract entities
    ents = heise.originaltext.progress_apply(lambda x: extract_ners(x, nlp)).explode()

    # Add metadata and save in results folder
    with open("data/interim/ents.csv", mode="w", encoding="utf-8") as file:
        file.write("ent\tstart\tend\tlabel\tdrop\n")
        for idx, e in zip(ents.index.values, ents.values):
            if isinstance(e, list):
                file.write(f"{idx}\t")
                for ee in e:
                    file.write(f"{ee}\t")
                file.write("\n")

    # Add metadata and save in results folder
    entsdf = pd.read_csv("data/interim/ents.csv", sep='\t')
    result = entsdf.join(heise)
    result.to_parquet("data/results/ents_with_metadata.gz")
