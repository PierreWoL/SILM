from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences = ["Resident evil","assassin's cread","witcher: wild hunt","Final fantasy"]
sentence_embeddings = model.encode(sentences)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(sentence_embeddings)
