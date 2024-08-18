# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TransformerModel(nn.Module):
    def __init__(
            self,
            lm,
            device,
            output_dim=0,
            hidden_mlp=0,
            nmb_prototypes=0,
            normalize=False,
            eval_mode=False,
            resize=-1,
            cls=True
    ):
        super(TransformerModel, self).__init__()
        self.eval_mode = eval_mode
        # Load the pretrained transformer model
        self.transformer = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        # Get the hidden size of the transformer model
        hidden_size = self.transformer.config.hidden_size
        if resize > 0:
            self.transformer.resize_token_embeddings(resize)
        # normalize output features
        self.l2norm = normalize
        # Projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(hidden_size, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_mlp),
                nn.LayerNorm(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )
        # Prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
        # print(self.prototypes)
        # cls token id
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id
        print(self.cls_token_id)
        self.cls=cls

    def _extract_columns(self, x, z, cls_indices=None):
        """
        Helper function for extracting column vectors from LM outputs.
        """
        x_flat = x.view(-1)
        column_vectors = z.view((x_flat.shape[0], -1))

        if cls_indices is None:
            indices = [idx for idx, token_id in enumerate(x_flat) \
                       if token_id == self.cls_token_id]
        else:
            indices = []
            seq_len = x.shape[-1]
            for rid in range(len(cls_indices)):
                indices += [idx + rid * seq_len for idx in cls_indices[rid]]
        return column_vectors[indices]

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        x_vals = inputs[:-1]  # Separate out cls_indices
        cls_indices = inputs[-1]
        for j in range(len(x_vals)):
            x_view1 = x_vals[j].to(self.device)
            z_view1 = self.transformer(x_view1)[0]

            cls_view1 = cls_indices[j]
            if self.cls is True:
                _out = self._extract_columns(x_view1, z_view1, cls_view1)
            else:
                _out = torch.mean(z_view1, dim=1, keepdim=True)[0]
            if j == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
        result = self.forward_head(output)
        return result

    def infer(self, x):
        x = x.to(self.device)
        z = self.transformer(x)[0]
        if self.projection_head is not None:
            z = self.projection_head(z)
        if self.cls is True:
            return self._extract_columns(x, z)
        else:
            return torch.mean(z, dim=1, keepdim=True)[0]


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out
