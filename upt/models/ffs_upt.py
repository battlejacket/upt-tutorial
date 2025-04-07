import einops
import torch
from torch import nn


class ffsUPT(nn.Module):
    def __init__(self, conditioner, encoder, approximator, decoder):
        super().__init__()
        self.conditioner = conditioner
        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder

    def forward(
            self,
            input_feat,
            input_pos,
            supernode_idxs,
            output_pos,
            batch_idx,
            re,
    ):
        condition = self.conditioner(re)

        # encode data
        latent = self.encoder(
            input_feat=input_feat,
            input_pos=input_pos,
            supernode_idxs=supernode_idxs,
            batch_idx=batch_idx,
            condition=condition,
        )

        # propagate forward
        latent = self.approximator(latent, condition=condition)

        # decode
        pred = self.decoder(
            x=latent,
            output_pos=output_pos,
            condition=condition,
        )

        return pred
