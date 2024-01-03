import os
import pickle
EMBEDDING_FOLDER = os.path.join(os.getcwd(), "result/embedding", "TabFact")
for embedding in [i for i in os.listdir(EMBEDDING_FOLDER) if i.endswith("pkl") ]: #and  "roberta" in i
            result_folder  = os.path.join("result/Valerie","TabFact")

            with open(os.path.join(EMBEDDING_FOLDER, embedding), 'rb') as file:
                data = pickle.load(file)
            embeddingResults = embedding[0:-4] + "_results.pkl"
            if embeddingResults in os.listdir(result_folder):
                print(embeddingResults)
                with open(os.path.join(result_folder, embeddingResults), 'rb') as file2:
                    dendrogra, linkage_matrix, threCluster_dict, simple_tree = pickle.load(file2)
                for index,  clusters in enumerate(threCluster_dict):
                    print(index,len(clusters[1]) )
