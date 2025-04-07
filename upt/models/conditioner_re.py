from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
from torch import nn


class ConditionerRe(nn.Module):
    def __init__(self, dim, num_values):
        super().__init__()
        cond_dim = dim * 4
        self.num_values = num_values
        self.dim = dim
        self.cond_dim = cond_dim
        self.register_buffer(
            "re_embed",
            get_sincos_1d_from_seqlen(seqlen=num_values, dim=dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, re):
        # checks + preprocess
        assert re.numel() == len(re)
        re = re.flatten()
        # embed
        embed = self.mlp(self.re_embed[re])
        return embed
