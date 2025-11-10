from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
import numpy as np


# 计算 Jaccard Similarity
def jaccard_similarity(phrase1, phrase2):
    set1, set2 = set(phrase1.split()), set(phrase2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
token ="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"
# 手动下载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token =token )
model = AutoModel.from_pretrained(model_name, use_auth_token =token )


def testMetrics(groundTruth, GeneratedText):
    overlall_metric = {}
    P, R, F1 = score(GeneratedText, groundTruth, lang="en", model_type="bert-base-uncased")
    overallR = {}

    bert_f1 = F1.mean().item()
    overlall_metric["bertscore"] = P.tolist(), R.tolist(), F1.tolist()
    #print(overlall_metric["bertscore"])
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2',token = token)
    gt_embeddings = sbert_model.encode(groundTruth)
    gen_embeddings = sbert_model.encode(GeneratedText)

    cosine_sims = [cosine_similarity([gt], [gen])[0, 0] for gt, gen in zip(gt_embeddings, gen_embeddings)]
    cosine_sim_avg = np.mean(cosine_sims)
    overlall_metric["cosine similarity"] = cosine_sims


    rouge = Rouge()
    rouge_scores = rouge.get_scores(GeneratedText, groundTruth, avg=True)
    rouge_l_f1 = rouge_scores["rouge-l"]["f"]
    #print(rouge_scores)
    # overlall_metric["rouge"] = rouge_scores
    #print(f"BERTScore F1: {bert_f1:.4f}")
    #print(f"Cosine Similarity (SBERT): {cosine_sim_avg:.4f}")
    #print(f"ROUGE-L F1: {rouge_l_f1:.4f}")

    overallR["bertscore f1"] = bert_f1
    overallR["cosine similarity"] = cosine_sim_avg
    overallR["rouge"] = rouge_l_f1
    return overlall_metric, overallR


"""
# 示例短语列表
ground_truths = ["ontology learning", "knowledge graph construction", "semantic reasoning"]
generated_phrases = ["ontology generation", "graph knowledge building", "semantic inference"]
metrics = testMetrics(ground_truths, generated_phrases)
print(metrics)
"""

