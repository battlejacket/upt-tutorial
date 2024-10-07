from torch import nn


class UPTSparseImageClassifier(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()
        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder

    def forward(self, input_feat, input_pos, supernode_idxs, batch_idx):
        # encode data
        latent = self.encoder(
            input_feat=input_feat,
            input_pos=input_pos,
            supernode_idxs=supernode_idxs,
            batch_idx=batch_idx,
        )

        # propagate forward
        latent = self.approximator(latent)

        # decode
        pred = self.decoder(latent)

        return pred
