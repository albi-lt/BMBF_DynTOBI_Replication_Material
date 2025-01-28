import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
import time

import awswrangler as wr

if __name__ == "__main__":
    # Load the model
    print("Loading model...")
    model = SentenceTransformer('all-mpnet-base-v2', device="cuda")
    # model = SentenceTransformer('shahrukhx01/paraphrase-mpnet-base-v2-fuzzy-matcher')
    print("Model loaded!")
    print("Loading data...")
    # load named entities data and print how long it took
    start = time.time()
    entsdf = pd.read_parquet("data/internal/ents_with_meta.gz")
    print("Data loaded!", time.time() - start, "seconds")

    print("Preprocessing data...")
    # we only care about organizations
    entsdf = entsdf.dropna(subset=["ent"])[entsdf.label == "ORG"]
    ents = entsdf.ent.unique()
    print(entsdf.shape, ents.shape)

    # define some mappings
    ents2id = {ent: id_ for id_, ent in enumerate(ents)}
    ent2count = entsdf.ent.value_counts().to_dict()

    print("Preprocessing data done!")

    # create embeddings and print how long it took in seconds
    print("Creating embeddings...")
    start = time.time()
    # encode unique ents. # create embeddings for all entities
    vecs_ents = model.encode(ents, batch_size=128, show_progress_bar=True)
    print("Embeddings created!", time.time() - start, "seconds")

    print(len(ents))


    # cluster data and print how long it took in seconds
    start = time.time()
    print("Start Clustering")
    d = DBSCAN(eps=0.15, min_samples=3, metric='cosine', metric_params=None, algorithm='auto', leaf_size=30,
               n_jobs=-1)
    c = d.fit_predict(vecs_ents)
    print("Done Clustering", time.time() - start, "seconds")

    with open("data/internal/clusters.txt", "w") as f:
        for ent, cluster in zip(ents, c):
            f.write(f"{ent}\t{cluster}\n")
    print("Clusters saved!")



    # save embeddings and print how long it took in seconds
    start = time.time()
    print("Save Embeddings")
    np.save("data/results/ents_embeddings.npy", vecs_ents)
    print("Done Saving Embeddings", time.time() - start, "seconds")


    print("Done")
