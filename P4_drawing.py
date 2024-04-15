# Plotting Precision and Recall for both bert and roberta as a function of Similarity
from matplotlib import pyplot as plt
import pandas as pd
import re

dataset = "WDC"
# df = pd.read_csv(f"result/P4/{dataset}/AttributeRelationshipScore_Base_EuFalse.csv")
df = pd.read_excel(f"C:/Users/Pierre/Downloads/Dropbox/AttributeRelationshipScore_Base_EuFalse{dataset}.xlsx")
print(df.columns)
embedding_methods = df["Embedding"].unique()



def rename(name):
    renaming = ""
    if 'Pretrain' in name:
        if 'header' in name:
            renaming = 'Pretrain_HI'
        if 'none' in name:
            renaming = 'Pretrain_I'
    elif 'SCT' in name:
        if '8' in name and '_header' in name:
            renaming = 'SILM_SampleCells_TFIDF_HI'
        elif '8' in name and '_none' in name:
            renaming = 'SILM_SampleCells_TFIDF_I'
        elif '1' in name and '_header' in name:
            renaming = 'Starmie_SampleCells_TFIDF_HI'
        elif '1' in name and '_none' in name:
            renaming = 'Starmie_SampleCells_TFIDF_I'
    elif 'SC' in name:
        if '8' in name and '_header' in name:
            renaming = 'SILM_SampleCells_HI'
        elif '8' in name and '_none' in name:
            renaming = 'SILM_SampleCells_I'
        elif '1' in name and '_header' in name:
            renaming = 'Starmie_SampleCells_HI'
        elif '1' in name and '_none' in name:
            renaming = 'Starmie_SampleCells_I'
    print(renaming, name)
    return renaming


def drawing_methods(dataset, methods, LM):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
    # Filter the DataFrame for bert and roberta and plot
    for method in methods:
        df_method = df[df["Embedding"] == method]
        df_method = df_method[df_method["Similarity"] == 0.7]
        ax[0].plot(df_method["Portion"], df_method["Precision"], label=rename(method))
        print(df_method)
        ax[1].plot(df_method["Portion"], df_method["Recall"], label=rename(method))
    # Setting the plot titles and labels
    ax[0].set_title(f'Precision vs Similarity --{LM}')
    ax[0].set_xlabel('Similarity')
    ax[0].set_ylabel('Precision')
    ax[0].legend()
    ax[0].set_ylim(0, 1)
    ax[0].set_yticks([i / 10.0 for i in range(0, 11)])

    ax[1].set_ylim(0, 1)
    ax[1].set_yticks([i / 10.0 for i in range(0, 11)])
    ax[1].set_title(f'Recall vs Similarity --{LM}')
    ax[1].set_xlabel('Similarity')
    ax[1].set_ylabel('Recall')
    ax[1].legend()
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=2)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=2)

    # Display the plots
    plt.tight_layout()
    plt.savefig(f"result/P4/{dataset}/drawing_{LM}.png")
    plt.show()


#drawing_methods(dataset, sbert_methods, "SBERT")
#drawing_methods(dataset, bert_methods, "BERT")
#drawing_methods(dataset, roberta_methods, "RoBERTa")


def drawing_methods_portion(dataset, methods, LM, similarity):
    fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharex=True)
    # Filter the DataFrame for bert and roberta and plot
    for method in methods:
        df_method = df[df["Embedding"] == method].iloc[:,:-2]
        df_method = df_method[df_method["Similarity"] == similarity]
        df_method = df_method.sort_values(by='Portion', ascending=False)
        print(df_method)
        ax[0].plot(df_method["Portion"], df_method["Precision"], label=rename(method))
        ax[1].plot(df_method["Portion"], df_method["Recall"], label=rename(method))
        ax[2].plot(df_method["Portion"], df_method["F-Score"], label=rename(method))
    # Setting the plot titles and labels
    ax[0].set_title(f'Precision vs Portion --{LM}, similarity --{similarity}')
    ax[0].set_xlabel('Portion')
    ax[0].set_ylabel('Precision')
    ax[0].legend()
    ax[0].set_ylim(0, 1)
    ax[0].set_yticks([i / 10.0 for i in range(0, 11)])
    ax[0].set_xlim(0, 1)
    ax[0].set_xticks([i / 10.0 for i in range(0, 11)])

    ax[1].set_title(f'Recall vs Portion --{LM}, similarity --{similarity}')
    ax[1].set_xlabel('Portion')
    ax[1].set_ylabel('Recall')
    ax[1].legend()
    ax[1].set_ylim(0, 1)
    ax[1].set_yticks([i / 10.0 for i in range(0, 11)])
    ax[1].set_xlim(0, 1)
    ax[1].set_xticks([i / 10.0 for i in range(0, 11)])

    ax[2].set_title(f'F-Score vs Portion --{LM}, similarity --{similarity}')
    ax[2].set_xlabel('Portion')
    ax[2].set_ylabel('Recall')
    ax[2].legend()
    ax[2].set_ylim(0, 1)
    ax[2].set_yticks([i / 10.0 for i in range(0, 11)])
    ax[2].set_xlim(0, 1)
    ax[2].set_xticks([i / 10.0 for i in range(0, 11)])

    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=3)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=3)
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=3)

    # Display the plots
    plt.tight_layout()
    plt.savefig(f"C:/Users/Pierre/Downloads/Dropbox/{dataset}/drawing_{LM}_sim{similarity}.png")
    plt.show()
sbert_methods = [i for i in embedding_methods if "_sbert_" in i and '1' not in i]
roberta_methods = [i for i in embedding_methods if "_roberta_" in i and '1' not in i]
bert_methods = [i for i in embedding_methods if "_bert_" in i and '1' not in i]

for i in [0.7, 0.8, 0.9]: #0.7, 0.8, 0.9  0.4, 0.5, 0.6, 0.7
    drawing_methods_portion(dataset, sbert_methods, "SBERT", i)
    drawing_methods_portion(dataset, bert_methods, "BERT", i)
    drawing_methods_portion(dataset, roberta_methods, "RoBERTa", i)
