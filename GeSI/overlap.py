from sentence_transformers import SentenceTransformer, util


def count_overlapping_nodes(tree1_nodes, tree2_nodes, similarity_threshold=0.9):
    """
    Given two sets of tree nodes (entity type names), count the number of overlapping nodes
    based on semantic similarity (default threshold = 0.9).

    :param tree1_nodes: List of node names from tree 1
    :param tree2_nodes: List of node names from tree 2
    :param similarity_threshold: Minimum similarity score to consider nodes as overlapping
    :return: Count of overlapping nodes and the matched node pairs
    """
    token = "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"

    model = SentenceTransformer('all-MiniLM-L6-v2',token = token)  # Use a lightweight SBERT model

    # Encode all node names
    print(tree1_nodes)
    tree1_nodes = [str(i) for i in tree1_nodes]
    tree2_nodes = [str(i) for i in tree2_nodes]

    embeddings1 = model.encode(tree1_nodes, convert_to_tensor=True)
    embeddings2 = model.encode(tree2_nodes, convert_to_tensor=True)

    # Compute pairwise cosine similarity
    similarity_matrix = util.cos_sim(embeddings1, embeddings2)

    # Find overlapping nodes based on threshold
    overlapping_count = 0
    matched_pairs = []

    for i, node1 in enumerate(tree1_nodes):
        for j, node2 in enumerate(tree2_nodes):
            if similarity_matrix[i][j] >= similarity_threshold:
                overlapping_count += 1
                matched_pairs.append((node1, node2))

    return overlapping_count, matched_pairs


