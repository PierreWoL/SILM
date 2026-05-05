from dataclasses import dataclass
from functools import cache
import networkx as nx
import numpy as np
from openai import OpenAI


@dataclass
class PostProcessHP:
    absolute_percentile: float = 0
    relative_percentile: float = 1
    remove_self_loops: bool = True
    remove_inverse_edges: bool = True
    llm_check = False
    llm_model = "gpt-3.5-turbo-1106"  # "gpt-3.5-turbo-1106"  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


@cache
def self_loops(G: nx.DiGraph) -> set:
    return {(u, v) for u, v in G.edges if u == v}


@cache
def inverse_edges(G: nx.DiGraph) -> set:
    def weight(u, v):
        return G[u][v].get("weight", 1)

    return {
        (u, v) for u, v in G.edges if G.has_edge(v, u) and weight(v, u) > weight(u, v)
    }


@cache
def absolute_percentile_edges(G: nx.DiGraph, percentile: float) -> set:
    if percentile == 1:
        return set(G.edges)
    edges = list(G.edges)
    weights = np.array([G[u][v].get("weight", 1) for u, v in edges])
    bottom_indices = np.argpartition(weights, int(percentile * len(edges)))[
                     : int(percentile * len(edges))
                     ]
    return {edges[i] for i in bottom_indices}


@cache
def relative_percentile_edges(G: nx.DiGraph, percentile: float) -> set:
    """Nucleus pruning: keep only the top percentile_to_keep of outgoing edges from the node."""
    assert 0 <= percentile <= 1

    if percentile == 1:
        return set()

    def prune_edges_out_from_node(node):
        edges = list(G.out_edges(node))
        if len(edges) == 0:
            return set()

        weights = np.array([G[u][v].get("weight", 1) for u, v in edges])
        weights_sorted = np.sort(weights)[::-1]  # sort in descending order
        prune_idx = np.argmax(
            (weights_sorted / weights_sorted.sum()).cumsum() > percentile
        )
        prune_value = weights_sorted[prune_idx]
        to_remove = {(u, v) for (u, v), w in zip(edges, weights) if w <= prune_value}
        return to_remove

    edges_to_remove = set()
    for n in G.nodes:
        edges_to_remove.update(prune_edges_out_from_node(n))
    return edges_to_remove


def llmInvalid(G: nx.DiGraph, modelName: str) -> set:
    edges_to_remove = set()
    roots = [node for node in G.nodes if G.in_degree(node) == 0]
    is_a_relations = [(parent, child) for parent, child in G.edges if parent not in roots]
    """
    if 'gpt' not in modelName:
        local_model_path = snapshot_download(modelName)
        llm = LLM(
            model=local_model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=8192,
            max_seq_len_to_capture=8192,
            max_num_seqs=512,
            enable_chunked_prefill=False,
        )
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048,
            # stop=None,
            seed=42,
        )
        tokenizer = llm.get_tokenizer()
        pbar = textpbar(len(is_a_relations))
        # for parent, child in is_a_relations:
        #    print(f"{child} is-a {parent}")
        for pages in batch(is_a_relations, 5):
            prompts = []
            for parent, child in pages:
                content = f'Is "{parent}" the parent entity type of "{child}"? Answer "Yes" or "No" only.'
                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
            outputs = llm.generate(
                prompts,
                sampling_params=sampling_params,
            )

            def extract_answer_regex(text):
                match = re.split(r"</think>\s*\n\n", text, flags=re.IGNORECASE)
                return match[1] if len(match) > 1 else text

            for page, out in zip(pages, outputs):
                answer = extract_answer_regex(out.outputs[0].text)
                if 'No' in answer or 'no' in answer:
                    edges_to_remove.update(page)
                    print(f"{page} is inappropriate is-a relationship.")
                pbar.update()
   
    else:
     """
    key = ""
    client = OpenAI(api_key=key)
    for parent, child in is_a_relations:
        content = f'Is "{parent}" the possible parent concept of "{child}"? Answer "Yes" or "No" only.'
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        response = client.chat.completions.create(
            model=modelName,
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            stop=["\n\n"]
        )
        new_assistant_reply = response.choices[0].message.content
        if 'No' in new_assistant_reply or 'no' in new_assistant_reply:
            edges_to_remove.update((parent, child))
            print(f"{(parent, child)} is inappropriate is-a relationship.")
    return edges_to_remove


def post_process(G: nx.DiGraph, hp: PostProcessHP) -> tuple[nx.DiGraph, int]:
    """Prune edges and nodes from a graph.

    Args:
        G: The input graph.
        edge_percentile: The bottom percentile of edges with the lowest weight are pruned.
        percentile_threshold: Outgoing edges with weight percentile > threshold are pruned.
        remove_self_loops: Remove self loops.
        remove_inverse_edges: Remove any pair (y, x) if p(y, x) < p(x, y).
    """
    edges_to_remove = set()
    edges_to_remove.update(absolute_percentile_edges(G, hp.absolute_percentile))
    edges_to_remove.update(relative_percentile_edges(G, hp.relative_percentile))
    if hp.remove_inverse_edges:
        edges_to_remove.update(inverse_edges(G))
    if hp.remove_self_loops:
        edges_to_remove.update(self_loops(G))
    if hp.llm_check:
        edges_to_remove.update(llmInvalid(G, modelName=hp.llm_model))
    # This also removes nodes with no incoming/outgoing edges
    G = nx.edge_subgraph(G, G.edges - edges_to_remove).copy()
    print("edges to be removed")
    for tuple_pc in edges_to_remove:
        print(tuple_pc)
    # Add the root node if it doesn't exist (helps for baselines)
    if "root" not in G.graph or G.graph["root"] not in G:
        root = "Thing"
        G = G.copy()  # type: ignore
        G.add_node(root, title=root)
        G.graph["root"] = root
        # Add edges from the root to all other nodes with no incoming edges
        for node in G.nodes:
            if G.in_degree(node) == 0 and node != root:
                G.add_edge(root, node, weight=1)

    return G, len(edges_to_remove)
