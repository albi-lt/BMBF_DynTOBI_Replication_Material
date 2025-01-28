import awswrangler as wr
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import tqdm

class EntityMatcher:
    def __init__(self, path_ents_with_metadata = "data/results/ents_with_metadata.tsv",
                 path_cluter_results = "data/results/clusters.txt",
                 result_save_path = "data/results/output.gz",
                 path_clustering_overview = "data/results/clustering_overview_df.gz",
                 path_orbis_embeddings = "data/results/names_embeddings.npy",
                 path_heise_ents_embeddings = "data/results/ents_embeddings.npy",
                 path_heise_to_orbis_mapping = "data/results/heise_to_orbis.gz",
                 create_orbis_embedding = False,
                 create_heise_embedding = False

                 ):
        self.model = None
        self.path_ents_with_metadata = path_ents_with_metadata
        self.path_cluter_results = path_cluter_results
        self.result_save_path =result_save_path
        self.path_clustering_overview = path_clustering_overview
        self.path_orbis_embeddings = path_orbis_embeddings
        self.path_heise_ents_embeddings =path_heise_ents_embeddings
        self.path_heise_to_orbis_mapping = path_heise_to_orbis_mapping
        self.create_orbis_embedding = create_orbis_embedding
        self.create_heise_embedding = create_heise_embedding


    @staticmethod
    def clean_name(name):
        """ Clean orbis company names, e.g. remove stuff like 'GMBH' """
        name = str(name)
        name = name.replace(" UG (HAFTUNGSBESCHRAENKT)", "")
        name = name.replace(" GMBH & CO. KG", "")
        name = name.replace(" GESELLSCHAFT M.B.H.", "")
        name = name.replace(" GMBH", "")
        name = name.replace(" E.K.", "")
        name = name.replace(" E.V.", "")
        name = name.replace(" OHG", "")
        name = name.capitalize()
        return name

    def load_model(self):
        """ Load sentence transformer model """
        print("Loading model...")
        self.model = SentenceTransformer('all-mpnet-base-v2', device="cuda")
        print("Model loaded.")

    def load_heise_ents_and_cluster_results(self):
        """ Load NEW extracted heise ents and cluster results """

        # %time g = wr.s3.read_parquet("s3://istariaibucket/pdf_urls/ents_with_meta.gz")
        # %time g.to_parquet("../data/internal/ents_with_meta.gz")
        print("Loading heise ents and cluster results...")
        try:
            self.entsdf = pd.read_parquet(self.path_ents_with_metadata)
        except:
            self.entsdf = pd.read_csv(self.path_ents_with_metadata, sep='\t')
        # we only care about organizations
        self.entsdf = self.entsdf.dropna(subset=["ent"])[self.entsdf.label == "ORG"]
        self.ents = self.entsdf.ent.unique()
        if self.create_heise_embedding is True:
            self.create_heise_entity_embeddings()

        # load cluster results
        self.df = pd.read_csv(self.path_cluter_results, sep='\t', header=None)
        self.df.columns = ["ent", "cluster"]
        print(self.df.cluster.nunique(), "Clusters")

        # define some mappings
        self.ent2cluster = self.df.set_index("ent").cluster.to_dict()
        self.ent2count = self.entsdf.ent.value_counts().to_dict()
        self.cluster2ent = self.df.groupby("cluster").ent.apply(list).to_dict()


    def create_heise_entity_embeddings(self):
        """ Create embeddings for heise ents """
        print("Creating heise embeddings...")
        self.heise_ents_embeddings = self.model.encode(self.ents, show_progress_bar=True)
        np.save(self.path_heise_ents_embeddings, self.heise_ents_embeddings)
        print("Heise embeddings created and saved.")

    def get_head_entity_per_cluster(self):
        """
        Get the head entity per cluster, i.e. the entity with the most occurrences in the newsticker texts
        Create a mapping from each ent to its resp. headent
        """
        print("Getting head entity per cluster...")
        self.ent2headent = dict()
        for cluster, cluster_ents in self.cluster2ent.items():
            if cluster == -1 or cluster == 1:
                for ent in cluster_ents:
                    self.ent2headent[ent] = ent
            else:
                counts = np.array([self.ent2count.get(e) for e in cluster_ents])
                head_ent = cluster_ents[counts.argmax()]
                for ent in cluster_ents:
                    self.ent2headent[ent] = head_ent

    def create_cluster_overview_df(self):
        """
        Create a dataframe with some cluster overview information.
        Specifically, we map the ent to its headent, the cluster it belongs to, and the number of members in the cluster
        """
        print("Creating cluster overview df...")
        # create cluster overview dataframe
        self.entdf = pd.DataFrame(self.ent2count.items(), columns=["ent", "count"])
        self.entdf["headent"] = self.entdf.ent.map(self.ent2headent)
        self.entdf["is_headent"] = np.where(self.entdf.ent == self.entdf.headent, 1, 0)
        self.entdf["cluster"] = self.entdf.headent.map(self.ent2cluster)
        self.entdf = self.entdf[~self.entdf["cluster"].isnull()]
        self.entdf["members"] = self.entdf["cluster"].map(self.cluster2ent).apply(len)
        self.entdf.to_parquet(self.path_clustering_overview)

    def create_company_name_embeddings(self):
        """
        Create embeddings for all company names in the orbis dataset.
        :return:
        """

        print("Loading orbis data...")
        self.orbis = wr.s3.read_csv("s3://istariaibucket/orbis/ORBIS.gz", sep='\t')
        # we're only interested in DACH companies
        self.orbis = self.orbis[self.orbis.country.isin(["Germany", "Switzerland", "Austria"])]
        self.orbis["clean_name"] = self.orbis["name"].apply(self.clean_name)
        print("Loaded", len(self.orbis), "companies")
        # mapping from clean company name to orbis id
        self.name2id = {name: _id for _id, name in zip(self.orbis.bvdid, self.orbis.clean_name.values)}
        self.bvdid2name = self.orbis.set_index("bvdid").name.to_dict()

        # get list of names
        self.names = self.orbis.clean_name.values.tolist()
        print(len(self.names), "Company names")


        if self.create_orbis_embedding is True:
            # encode names
            print("Encoding names...")
            self.orbis_vecs_names = self.model.encode(self.names, batch_size=128, show_progress_bar=True)
            print("Saving embeddings...")
            np.save(self.path_orbis_embeddings, self.orbis_vecs_names)

    ## calculate sims

    def calculate_similarity_between_company_names_and_heise_ents(self):
        """ calculate cosinus sim for all heise and orbis ents"""

        # load heise entitiy embeddings
        print("Loading heise entity embeddings...")
        self.vecs_ents = np.load(self.path_heise_ents_embeddings)
        print("Loading orbis company name embeddings...")
        self.vecs_names = np.load(self.path_orbis_embeddings)


        # cant fit all at once on gpu, thus we do it in batches
        #
        print("Calculating similarities...")
        step = 15000
        self.top_vals = [] # store top values
        self.top_idxs = [] # store top indices
        for i in tqdm.tqdm(range(0, self.vecs_names.shape[0], step)):
            # calculate cos sim between heise ents and orbis company names
            sim = util.cos_sim(self.vecs_names[i:i + step, :], self.vecs_ents)
            # print(sim.shape)
            self.top_vals.append(sim.max(1)[0].numpy())
            self.top_idxs.append(sim.max(1)[1].numpy())

        # flatten lists
        self.ti = [a for b in self.top_idxs for a in b]
        self.tv = [a for b in self.top_vals for a in b]
        print(len(self.ti), len(self.tv))

        self.matches = []
        for x in range(len(self.ti)):
            if self.tv[x] > 0.95:
                name = self.names[x]
                ent = self.ents[self.ti[x]]
                self.matches.append((x, name, ent, self.tv[x]))

        print("Finished calculating similarities")
        print("Found", len(self.matches), "matches")

        # save matches
        print("Saving matches to disk...", self.path_heise_to_orbis_mapping)
        self.res = pd.DataFrame(self.matches, columns=["x", "name", "ent", "sim"])
        self.res.to_parquet(self.path_heise_to_orbis_mapping)

    def map_matched_ents_to_cluster_head(self):
        """
        Map the matched ents to the cluster head ent
        :return:
        """
        print("Mapping matched ents to cluster head...")
        self.res["cluster"] = self.res.ent.map(self.ent2cluster)

        # create a new column holding all the ents in the cluster. We dont use cluster -1 (i.e. no cluster) and cluster 1
        #, which is a large cluster of all ents that are not in any other cluster but the cluster does not make sense
        self.res["cluster_member"] = self.res.apply(
            lambda row: self.cluster2ent.get(row.cluster, [row.ent]) if not row.cluster in [-1, 1] else [row.ent],
            axis=1)

        # explode the list of ents in the cluster and join with the original df
        self.res = pd.DataFrame(self.res["cluster_member"].explode().dropna()).join(
            self.res.drop(columns=["cluster_member"]), how="left")
        self.res["bvdid"] = self.res.name.map(self.name2id)
        self.res["mup_name"] = self.res.bvdid.map(self.bvdid2name)
        self.ent2id = self.res.set_index("cluster_member").bvdid.to_dict()
        # ent2id = res.set_index("ent").bvdid.to_dict()

    def clean_double_mappings(self):
        """
         Due to possible false clustering sometimes a cluster of heise ents would map to more than one mup ent.
         This is possible bc we compare each mup ent with each heise ent, i.e. if the heise ents DFG and VVV were put in
         the same cluster but each individually would map to dfg gmbh resp. vvv gmbh, then we would create a mapping
         from one heise ent cluster to several mup ents, which we dont want.  In these cases only keep the ents from
         the heise cluster that exactly match the ent that was matched with the mup ent.
        """
        self.groups = []
        for idx, group in self.res.groupby("cluster"):
            if idx in [-1, 1]:
                self.groups.append(group)
                continue
            if group.bvdid.nunique() > 1:
                group = group[group.cluster_member == group.ent].copy()
            self.groups.append(group)
        self.re = pd.concat(self.groups)

        # sometimes a heise entitiy was matched so several mup ents, i.e. when the name of the mup ents are very similar
        # in these cases only use the most similar one to avoid mapping one heise ent to several mup ents.
        self.groups = []
        for idx, group in self.re.groupby("cluster_member"):
            if group.bvdid.nunique() > 1:
                group = group.sort_values("sim", ascending=False).iloc[:1].copy()
            self.groups.append(group)
        self.re = pd.concat(self.groups)
        self.ent2id = self.re.set_index("cluster_member").bvdid.to_dict()

    def map_mup_to_heise_ents(self):
        """
        Map mup ents to heise ents
        :return:
        """
        # map mup ents to heise ents
        # map ids to ents and
        # group matched ents and ids on newsarticle level
        print("Mapping mup ents to heise ents...")
        self.entsdf["bvdid"] = self.entsdf.ent.apply(lambda x: self.ent2id.get(x, None))
        self.entsdf["mup_name"] = self.entsdf.bvdid.map(self.bvdid2name)

    def save_results(self):
        """
        Save results to S3
        :return:
        """
        print("Saving results to S3...", self.result_save_path)
        self.entsdf.to_parquet(self.result_save_path)

    def run(self):

        self.load_model()
        self.load_heise_ents_and_cluster_results()
        self.get_head_entity_per_cluster()
        self.create_cluster_overview_df()
        self.create_company_name_embeddings()
        self.calculate_similarity_between_company_names_and_heise_ents()
        self.map_matched_ents_to_cluster_head()
        self.clean_double_mappings()
        self.map_mup_to_heise_ents()
        self.save_results()