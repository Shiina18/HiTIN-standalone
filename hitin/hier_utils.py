from typing import List

import numpy as np
import torch
import transformers


class HierarchicalStructure:
    """class levels are separated by `#`; its name has its parent label name as prefix
    classes format example
    labels = [
        'Seating',  # level 1
        'Tables',
        'Seating#Chairs',  # level 2
        'Seating#Sofas',
        'Tables#Desks',
        'Seating#Chairs#Dining chairs',  # level 3
        'Seating#Chairs#Office chairs',
        'Seating#Sofas#Sectional sofas',
        'Tables#Desks#Computer desks',
    ]
    """
    def __init__(self, labels: List[str]):
        id2label = {i: x for i, x in enumerate(labels)}
        label2id = {x: i for i, x in enumerate(labels)}

        node2parent = {}
        for label in labels:
            levels = label.split('#')
            if len(levels) > 1:
                node2parent[label2id[label]] = label2id['#'.join(levels[:-1])]

        adj_matrix = np.zeros((len(labels), len(labels)))
        for n, p in node2parent.items():
            adj_matrix[n][p] = 1
            adj_matrix[p][n] = 1

        self.id2label = id2label
        self.label2id = label2id
        self.node2parent = node2parent
        self.adj_matrix = adj_matrix


def make_trainer(loss_fct, lambda_=1e-6):
    class MyTrainer(transformers.Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            """`inputs` must be a dict with `labels` as the key for labels"""
            labels = inputs.pop('labels')
            outputs = model(**inputs)

            classifier_weight = model.gnn.classifier.weight
            weight_n = torch.index_select(classifier_weight, 0, model.node_indices.weight)
            weight_pn = torch.index_select(classifier_weight, 0, model.parent_indices.weight)
            recursive_regularization = torch.sum(torch.pow(weight_n - weight_pn, 2))

            loss = loss_fct(outputs, labels) + lambda_ * 1 / 2 * recursive_regularization
            return (loss, {'outputs': outputs}) if return_outputs else loss
    return MyTrainer
