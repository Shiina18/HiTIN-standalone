from typing import Dict, Tuple, List

import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hitin import CIRCA


class VanillaBERT(nn.Module):
    """Unused, for comparison only"""

    def __init__(self, bert_path, num_labels=2, classifier_dropout=None):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(
            classifier_dropout if classifier_dropout is not None
            else self.bert.config.hidden_dropout_prob  # 0.1
        )
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, *args, **kwargs):
        output = self.bert(*args, **kwargs)
        pooled_output = output.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


################################################################################
# Simple refactor of the original code. I don't bother changing it a lot.

def _build_tree_core(adj_matrix, depth=2) -> Dict[int, CIRCA.CodingTreeNode]:
    adj_matrix = np.array(adj_matrix)
    tree = CIRCA.CodingTree(adj_matrix)
    mode = 'v1' if adj_matrix.shape[0] <= 2 else 'v2'
    tree.build_coding_tree(mode=mode, k=depth)
    tree = CIRCA.get_child_h(tree, k=depth)
    tree = CIRCA.map_id(tree, k=depth)
    # Note that here nodes are ordered by layer so that [layer 0 nodes, layer 1 nodes, ...]
    id2node = CIRCA.update_node(tree.tree_node)
    return id2node


def build_tree(adj_matrix, depth=2):
    """adj_matrix should exclude the root node."""
    id2node = _build_tree_core(adj_matrix=adj_matrix, depth=depth)
    tree = {
        'node_size': [0] * (depth + 1),
        'leaf_size': len(adj_matrix),
        'edges': [[] for _ in range(depth + 1)],
    }

    idx_offsets_by_layer = [0]
    for layer in range(depth + 1):
        layer_node_ids = [i for i, n in id2node.items() if n.child_h == layer]
        idx_offsets_by_layer.append(layer_node_ids[0] + len(layer_node_ids))
        tree['node_size'][layer] = len(layer_node_ids)

    for _, n in id2node.items():
        if n.child_h > 0:
            node_idx = n.ID - idx_offsets_by_layer[n.child_h]
            child_idx_offset = idx_offsets_by_layer[n.child_h - 1]
            tree['edges'][n.child_h].extend(
                [(node_idx, child_idx - child_idx_offset) for child_idx in n.children]
            )

    return tree

"""
adj_matrix = [
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # root
    [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
]
adj_matrix = adj_matrix[1:]  # remove root
tree = build_tree(adj_matrix)
"""

################################################################################


class MLP(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, num_layers: int = 1
    ):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(in_features, hidden_features)]
            + [nn.Linear(hidden_features, hidden_features) for _ in range(num_layers - 1)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_features) for _ in range(num_layers)]
        )

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)

            # TODO: ad-hoc
            x = torch.transpose(x, -1, -2)
            x = self.batch_norms[i](x)
            x = torch.transpose(x, -1, -2)

            x = F.relu(x)
        return x


class Matrix(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weight, x)


class TreeIsomorphismNetwork(nn.Module):

    def __init__(self, tree, embedding_size, num_labels):
        super().__init__()
        self.adj_matrices_by_layer = nn.ModuleList([self._get_inter_layer_adj_matrix(es) for es in tree['edges'][1:]])
        self.mlps = nn.ModuleList([MLP(embedding_size, embedding_size) for _ in range(len(tree['edges']) - 1)])
        self.dropout = nn.Dropout(0.5)  # configurable
        self.classifier = nn.Linear(embedding_size * (len(self.mlps) + 1), num_labels)

    @staticmethod
    def _get_inter_layer_adj_matrix(edges: List[Tuple[int, int]]):
        """
        edge (node_idx, child_node_idx)
        indices of nodes in each layer are shifted so that the index starts with 0 in each layer
        """
        if not edges:
            return torch.Tensor()
        num_rows = max(row for row, col in edges) + 1
        num_cols = max(col for row, col in edges) + 1
        adj_matrix = torch.zeros(num_rows, num_cols)
        for row, col in edges:
            adj_matrix[row, col] = 1
        return Matrix(adj_matrix)

    def forward(self, node_embeddings):
        """(batch_size, num_labels, node_embedding_size)"""
        node_embeddings_by_layer = [node_embeddings]
        for i in range(len(self.mlps)):
            node_embeddings = self.adj_matrices_by_layer[i](node_embeddings)
            node_embeddings = self.mlps[i](node_embeddings)
            # (batch_size, num_nodes_this_layer, node_embedding_size)
            node_embeddings_by_layer.append(node_embeddings)

        # pooling, configurable
        concated_pooled_embeddings = torch.cat(
            [self.dropout(x.mean(dim=-2)) for x in node_embeddings_by_layer], dim=-1
        )
        logits = self.classifier(concated_pooled_embeddings)
        return logits


class HiTIN(nn.Module):
    def __init__(
        self, bert_path, tree, node2parent: Dict[int, int],
        num_labels=2, classifier_dropout=None
    ):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(
            classifier_dropout if classifier_dropout is not None
            else self.bert.config.hidden_dropout_prob  # 0.1
        )
        # self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        # initial node embedding
        node_embedding_size = 768  # configurable
        self.node_embedding_dropout = nn.Dropout(0.1)  # configurable
        self.linear_duplication = nn.Linear(1, num_labels)
        self.linear_projection = nn.Linear(self.bert.config.hidden_size, node_embedding_size)

        self.gnn = TreeIsomorphismNetwork(
            tree, embedding_size=node_embedding_size, num_labels=num_labels
        )

        node_indices = []
        parent_indices = []
        for n, p in node2parent.items():
            node_indices.append(n)
            parent_indices.append(p)
        self.node_indices = Matrix(torch.LongTensor(node_indices))
        self.parent_indices = Matrix(torch.LongTensor(parent_indices))

    def calc_initial_node_embeddings(self, pooled_output):
        """
        Equation 8 of the paper. Note that the original implementation is different
        from the equation 8, and I will follow the original implementation
        with a little modification.

        input is of size (batch_size, bert_hidden_size)
        output is of size (batch_size, num_labels, node_embedding_size)
        """
        pooled_output = torch.unsqueeze(pooled_output, dim=-2)
        pooled_output = self.node_embedding_dropout(self.linear_projection(pooled_output))

        pooled_output = torch.transpose(pooled_output, -1, -2)
        pooled_output = self.node_embedding_dropout(self.linear_duplication(pooled_output))
        node_embeddings = torch.transpose(pooled_output, -1, -2)

        return node_embeddings

    def forward(self, *args, **kwargs):
        output = self.bert(*args, **kwargs)
        pooled_output = self.dropout(output.pooler_output)
        node_embedidngs = self.calc_initial_node_embeddings(pooled_output)
        logits = self.gnn(node_embedidngs)
        return logits
