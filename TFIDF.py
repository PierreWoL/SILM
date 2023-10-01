from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from Utils import split
from d3l.utils.functions import tokenize_str as tokenize
"""
# Sample list of phrases
phrases = [
    'hello world',
    'goodbye world',
    'hello user',
    'goodbye user',
    'hello',
    'goodbye',
    'user',
    'world'
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(use_idf=True)

# Compute TF-IDF scores
tfidf_matrix = vectorizer.fit_transform(phrases)

# Extract the scores for each phrase
tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=phrases)

#print(tfidf_scores)
avg_tfidf = tfidf_scores.mean(axis=1)
#print(avg_tfidf)
"""




def table_tfidf(table:pd.DataFrame):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(use_idf=True)

    # Function to compute average TF-IDF for a list of phrases
    def compute_avg_tfidf(column):


        # Compute TF-IDF scores

        if isinstance(column,list):
            tfidf_matrix = vectorizer.fit_transform(column)
            # Extract the scores for each phrase
            tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(),
                                        index=column)
            mid = tfidf_scores.mean(axis=1)
            result_dict = {}
            for index, value in mid.items():
                if index not in result_dict:
                    result_dict[index] =value
            return result_dict
        else:
            column_copy = column.copy().apply(tokenize)
            try:
                tfidf_matrix = vectorizer.fit_transform(column_copy)
                tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
                # Compute average TF-IDF scores for each cell
                return tfidf_scores.mean(axis=1).tolist()
            except:
                column_copy[:] = 0
                return column_copy


    if len(table)==1:
        result =[]
        for i in table.columns:
            column_list = split(table[i][0])
            i_tfidf = compute_avg_tfidf(column_list)
            result.append(i_tfidf)

        return result
    else:
        # Compute average TF-IDF for each column and store in a new dataframe
        result = table.apply(compute_avg_tfidf)
        return result


"""df = pd.DataFrame({
    'A': ['cat', 'dog', 'animals', 'mammal animals'],
    'B': ['apple', 'orange', 'fruit', 'organic fruit']
})
df = pd.read_csv("datasets/WDC/Test/T2DV2_253.csv")
df = df.astype(str)

print(table_tfidf(df))

table1 = pd.read_csv("datasets/TabFact/Test/1-2709-4.html.csv")

print(table_tfidf(table1))
"""