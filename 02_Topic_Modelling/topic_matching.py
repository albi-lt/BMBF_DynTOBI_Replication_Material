#matching lda and pvtm topics (robustness-check)

import pandas as pd


###functions
#aus dem Skript
def topic_word_frequencies_matching(lda1,vec1,lda2,vec2, measure='cosine similarity'):
    # rows = vocabulary words, columns = number of topics
    d1 = topic_word_dataframe(lda1, vec1.get_feature_names())  #rows (vocabulary words), columns (number of topics)
    d2 = topic_word_dataframe(lda2, vec2.get_feature_names())  # 5187 rows (number of words in vocabulary), 20 columns (number of topics)
    d2.columns = [str(i) + '_2' for i in range(lda2.n_components)]
    # join columns of another data frame, "how" default "left" (use calling frame's index, also indizes von der linken data frame)
    # andere Möglichkeiten für how right, outer (union of the indices), inner (intersection of the indices)
    # 'inner': form intersection of calling frame's index with other's index, preserving the order of the calling one.
    # d12 contains the words as row indexer, and topic names in the columns (the topics of the second dataframe are with the suffix _2)
    d12 = d1.join(d2, how='inner')  # behalte nur die Wörter, die in beiden data frames vorkommen
    print('Vocabulary size of the first model: '+str(len(vec1.get_feature_names())))
    print('Vocabulary size of the second model: ' + str(len(vec2.get_feature_names())))
    print('Joint vocabulary size: ' + str(d12.shape[0]))
    a = d12.filter(regex="^((?!_).)*$").columns.values.tolist() # column names of the first topic word dataframe
    b = d12.filter(like="_2").columns.values.tolist() # column names of the second topic word dataframe
    topic_assignment = []
    scores = []
    if measure == 'cosine similarity':
        for topic in a:
            similarities = [cosine_similarity([d12[topic].values], [d12[topic2].values])[0][0] for topic2 in b]
            assignment = np.argmax(similarities)  # extract the index (topic number) of the maximum similarity score
            # score = np.sort(similarities)[::-1][0] # extract the similarity score (maximum), np.sort does not work!!
            score = similarities[assignment]
            topic_assignment.append(assignment)
            scores.append(score)
    elif measure == 'js distance':
        from scipy.spatial import distance
        for topic in a:
            similarities = [distance.jensenshannon(d12[topic].values.tolist(), d12[topic2].values.tolist()) for topic2 in b]
            assignment = np.argmin(similarities)  # extract the index (topic number) of the maximum similarity score
            # score = np.sort(similarities)[::-1][0] # extract the similarity score (maximum), np.sort does not work!!
            score = similarities[assignment]
            topic_assignment.append(assignment)
            scores.append(score)
    return topic_assignment, scores

def plot_matched_topics(topic_assignment, cosine_values,
                        number_of_topics, filename, description,  save_as_pdf = False,
                        titles=['LDA Topic: ','PVTM Topic: ']):

    import matplotlib.backends.backend_pdf
    if save_as_pdf:
        pdf = matplotlib.backends.backend_pdf.PdfPages(filename+".pdf")
    from wordcloud import WordCloud
    import matplotlib.backends.backend_pdf
    import re
    x, y = np.ogrid[:300, :300]
    shape = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    shape = 255 * shape.astype(int)
    for i in range(number_of_topics):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        topic1 = i
        similarity_score = cosine_values[i]
        topic2 = topic_assignment[i]
        #text1 = topic_words1[topic1]
        plt.subplot(1, 2, 1);
        wordcloud1 = WordCloud(max_font_size=50,
                               background_color="white", mask=shape, colormap ='tab10').fit_words(dict(df_lda_topic_term.iloc[topic1].sort_values(ascending=False)[0:50]))
        plt.imshow(wordcloud1, interpolation="bilinear", );
        plt.axis("off");
        plt.title(titles[0] + str(topic1));
        #text2 = topic_words2[topic2]
        plt.subplot(1, 2, 2);
        wordcloud2 = WordCloud(max_font_size=50,
                               background_color="white", mask=shape).fit_words(dict(df_pvtm_topic_term_m1.iloc[topic2].sort_values(ascending=False)[0:50]))
        plt.imshow(wordcloud2, interpolation="bilinear", );
        plt.axis("off");
        plt.title(titles[1] + str(topic2) + description + str(
            round(similarity_score, 4)));
        plt.show();
        if save_as_pdf:
            pdf.savefig(fig, orientation ='landscape', dpi = 450)

    if save_as_pdf:
        fig = plt.figure()
        plt.plot(range(number_of_topics), np.sort(cosine_values), linewidth=2, markersize=12)
        plt.title('Cosine similarity values', size=14)
        plt.xticks(np.arange(0, number_of_topics + 1, 3.0))
        pdf.savefig(fig, orientation='landscape',dpi = 450)
        pdf.close()
###
#load topic term matrix from lda and pvtm

df_lda_topic_term = pd.read_pickle("02_Topic_Modelling/LDA_120_Topic_Term_Matrix/lda_topic_term_weights.pkl")
df_pvtm_topic_term_m1 = pd.read_pickle("02_Topic_Modelling/02_Topic_Modelling_PVTM/PVTM_Topic_Term_Matrix/method_1_topic_term_weights.pkl") #method1 is Topic-Dokument Matrix * Dokument-Term Matrix, result is normalized
df_pvtm_topic_term_m2 = pd.read_pickle("02_Topic_Modelling/02_Topic_Modelling_PVTM/PVTM_Topic_Term_Matrix/method_2_topic_term_weights.pkl") #method2 is Topic-Dokument Matrix * Dokument-Term Matrix normalized by doclength, result is normalized so that rowsums sum up to 1

#transpose nutzen damit Vokabular Zeile und Topic Spalte ist
df_topic_term_joined = df_lda_topic_term.transpose().join(df_pvtm_topic_term_m1.transpose(), how ='inner')

lda_cols = df_topic_term_joined.filter(regex="^topic((?!_).)*$").columns.values.tolist()
pvtm_cols = df_topic_term_joined.filter(regex="^\d").columns.values.tolist()

measure = 'cosine similarity'
if measure == 'cosine similarity':
    for topic in lda_cols:
        similarities = [cosine_similarity([df_topic_term_joined[topic].values], [df_topic_term_joined[topic2].values])[0][0] for topic2 in pvtm_cols]
        assignment = np.argmax(similarities)  # extract the index (topic number) of the maximum similarity score
        # score = np.sort(similarities)[::-1][0] # extract the similarity score (maximum), np.sort does not work!!
        score = similarities[assignment]
        topic_assignment.append(assignment)
        scores.append(score)


plot_matched_topics(topic_assignment, scores,number_of_topics=120, filename ='05_Topic_Matching\lda_pvtm-m1_matched_topics_topic_word_frequencies_matching-cosine_similarity', description = '\n Cosine similarity score:', save_as_pdf=True)

topic_assignment = []
scores = []
measure = 'js distance'
from scipy.spatial import distance

if measure == 'js distance':
    for topic in lda_cols:
        similarities = [distance.jensenshannon(df_topic_term_joined[topic].values.tolist(), df_topic_term_joined[topic2].values.tolist()) for topic2 in pvtm_cols]
        assignment = np.argmin(similarities)  # extract the index (topic number) of the maximum similarity score
        # score = np.sort(similarities)[::-1][0] # extract the similarity score (maximum), np.sort does not work!!
        score = similarities[assignment]
        topic_assignment.append(assignment)
        scores.append(score)

plot_matched_topics(topic_assignment, scores,number_of_topics=120, filename ='05_Topic_Matching\lda_pvtm-m1_matched_topics_topic_word_frequencies_matching-js_distance', description = '\n Js-distance score:', save_as_pdf=True)

# Method 2
df_topic_term_joined_m2 = df_lda_topic_term.transpose().join(df_pvtm_topic_term_m2.transpose(), how ='inner')

topic_assignment = []
scores = []
measure = 'cosine similarity'
if measure == 'cosine similarity':
    for topic in lda_cols:
        similarities = [cosine_similarity([df_topic_term_joined_m2[topic].values], [df_topic_term_joined_m2[topic2].values])[0][0] for topic2 in pvtm_cols]
        assignment = np.argmax(similarities)  # extract the index (topic number) of the maximum similarity score
        # score = np.sort(similarities)[::-1][0] # extract the similarity score (maximum), np.sort does not work!!
        score = similarities[assignment]
        topic_assignment.append(assignment)
        scores.append(score)

def plot_matched_topics(topic_assignment, cosine_values,
                        number_of_topics, filename, description,  save_as_pdf = False,
                        titles=['LDA Topic: ','PVTM Topic: ']):

    import matplotlib.backends.backend_pdf
    if save_as_pdf:
        pdf = matplotlib.backends.backend_pdf.PdfPages(filename+".pdf")
    from wordcloud import WordCloud
    import matplotlib.backends.backend_pdf
    import re
    x, y = np.ogrid[:300, :300]
    shape = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    shape = 255 * shape.astype(int)
    for i in range(number_of_topics):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        topic1 = i
        similarity_score = cosine_values[i]
        topic2 = topic_assignment[i]
        #text1 = topic_words1[topic1]
        plt.subplot(1, 2, 1);
        wordcloud1 = WordCloud(max_font_size=50,
                               background_color="white", mask=shape, colormap ='tab10').fit_words(dict(df_lda_topic_term.iloc[topic1].sort_values(ascending=False)[0:50]))
        plt.imshow(wordcloud1, interpolation="bilinear", );
        plt.axis("off");
        plt.title(titles[0] + str(topic1));
        #text2 = topic_words2[topic2]
        plt.subplot(1, 2, 2);
        wordcloud2 = WordCloud(max_font_size=50,
                               background_color="white", mask=shape).fit_words(dict(df_pvtm_topic_term_m2.iloc[topic2].sort_values(ascending=False)[0:50]))
        plt.imshow(wordcloud2, interpolation="bilinear", );
        plt.axis("off");
        plt.title(titles[1] + str(topic2) + description + str(
            round(similarity_score, 4)));
        plt.show();
        if save_as_pdf:
            pdf.savefig(fig, orientation ='landscape', dpi = 450)

    if save_as_pdf:
        fig = plt.figure()
        plt.plot(range(number_of_topics), np.sort(cosine_values), linewidth=2, markersize=12)
        plt.title('Cosine similarity values', size=14)
        plt.xticks(np.arange(0, number_of_topics + 1, 3.0))
        pdf.savefig(fig, orientation='landscape',dpi = 450)
        pdf.close()

plot_matched_topics(topic_assignment, scores,number_of_topics=120, filename ='05_Topic_Matching\lda_pvtm-m2_matched_topics_topic_word_frequencies_matching-cosine_similarity', description = '\n Cosine similarity score:', save_as_pdf=True)

topic_assignment = []
scores = []
measure = 'js distance'
from scipy.spatial import distance

if measure == 'js distance':
    for topic in lda_cols:
        similarities = [distance.jensenshannon(df_topic_term_joined_m2[topic].values.tolist(), df_topic_term_joined_m2[topic2].values.tolist()) for topic2 in pvtm_cols]
        assignment = np.argmin(similarities)  # extract the index (topic number) of the maximum similarity score
        # score = np.sort(similarities)[::-1][0] # extract the similarity score (maximum), np.sort does not work!!
        score = similarities[assignment]
        topic_assignment.append(assignment)
        scores.append(score)

plot_matched_topics(topic_assignment, scores,number_of_topics=120, filename ='05_Topic_Matching\lda_pvtm-m2_matched_topics_topic_word_frequencies_matching-js_distance', description = '\n Js-distance score:', save_as_pdf=True)

