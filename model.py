import torch

from typing import Literal
from torch import nn
from torch.linalg import vector_norm
from transformers import AutoModel


class ParagraphCRBiEncoder(nn.Module):
    def __init__(self, pretrained_model_checkpoint: str = "malteos/scincl", freeze_layers: int = 10):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_checkpoint)

        frozen_layers = [f"encoder.layer.{i}." for i in range(freeze_layers)]
        for parameter_name, parameter in self.encoder.named_parameters():
            if any(parameter_name.startswith(frozen_layer) for frozen_layer in frozen_layers):
                parameter.requires_grad = False

    def embed(self, input_ids, token_type_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]

    def forward(
        self, batch: dict[str, torch.Tensor], instance_type: str = Literal["triplet", "quadruplet"]
    ) -> tuple[torch.Tensor, ...]:
        if instance_type == "quadruplet":
            query_embs = self.embed(
                batch["query_input_ids"], batch["query_token_type_ids"], batch["query_attention_mask"]
            )
            pos_1_embs = self.embed(
                batch["pos_1_input_ids"], batch["pos_1_token_type_ids"], batch["pos_1_attention_mask"]
            )
            pos_2_embs = self.embed(
                batch["pos_2_input_ids"], batch["pos_2_token_type_ids"], batch["pos_2_attention_mask"]
            )
            neg_embs = self.embed(batch["neg_input_ids"], batch["neg_token_type_ids"], batch["neg_attention_mask"])

            distance_query_pos_1 = vector_norm(query_embs - pos_1_embs, dim=1)
            distance_query_pos_2 = vector_norm(query_embs - pos_2_embs, dim=1)
            distance_query_neg = vector_norm(query_embs - neg_embs, dim=1)

            distance_pos_1_pos_2 = vector_norm(pos_1_embs - pos_2_embs, dim=1)
            distance_pos_1_neg = vector_norm(pos_1_embs - neg_embs, dim=1)
            distance_pos_2_neg = vector_norm(pos_2_embs - neg_embs, dim=1)

            return (
                distance_query_pos_1,
                distance_query_pos_2,
                distance_query_neg,
                distance_pos_1_pos_2,
                distance_pos_1_neg,
                distance_pos_2_neg,
            )
        else:
            query_embs = self.embed(
                batch["query_input_ids"], batch["query_token_type_ids"], batch["query_attention_mask"]
            )
            pos_embs = self.embed(batch["pos_input_ids"], batch["pos_token_type_ids"], batch["pos_attention_mask"])
            neg_embs = self.embed(batch["neg_input_ids"], batch["neg_token_type_ids"], batch["neg_attention_mask"])

            return vector_norm(query_embs - pos_embs, dim=1), vector_norm(query_embs - neg_embs, dim=1)
