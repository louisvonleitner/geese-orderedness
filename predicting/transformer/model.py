import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


class goose_number_transformer(nn.Module):

    def __init__(self, n_known_geese, d_model, n_heads, n_layers, dropout):
        super().__init__()

        self.d_model = d_model
        self.n_known_geese = n_known_geese
        self.input_length = n_known_geese + 1

        # model parameters
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = 4 * self.d_model
        self.activation = "relu"
        self.bias = True
        self.dropout = dropout

        # in features = 9 = 3 positional values, 3 velocity values, 3 acceleration values
        self.input_projection = nn.Linear(in_features=9, out_features=d_model)
        # output projection just returns value "number of geese"
        self.output_projection = nn.Linear(in_features=d_model, out_features=1)

        self.token_type_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,  # same as the attention paper
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=0e-5,  # standard
            bias=self.bias,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

    def forward(self, X, token_type_ids):

        # embed
        # input is of dimension (batch_size, n_tokens, 8)
        X_projected = self.input_projection(X)

        type_embeddings = self.token_type_embedding(token_type_ids)

        X_combined = X_projected + type_embeddings

        # encode tokens in transformer
        X_transformed = self.transformer_encoder(X_combined)

        cls_output = X_transformed[:, 0, :]

        # decode tokens to prediction
        y_pred = self.output_projection(cls_output)

        # squeeze output
        y_pred = y_pred.squeeze(-1)

        return y_pred
