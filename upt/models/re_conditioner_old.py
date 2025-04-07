from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
from kappamodules.init import init_xavier_uniform_zero_bias, init_truncnormal_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn

from models.base.single_model_base import SingleModelBase


class reConditionerPdearena(SingleModelBase):
    def __init__(self, dim, cond_dim=None, init_weights="xavier_uniform", **kwargs):
        super().__init__(**kwargs)
        # self.num_total_timesteps = self.data_container.get_dataset().getdim_timestep()
        self.dim = dim
        self.cond_dim = cond_dim or dim * 4
        self.init_weights = init_weights
        self.static_ctx["condition_dim"] = self.cond_dim
        # # buffer/modules
        # self.register_buffer(
        #     "timestep_embed",
        #     get_sincos_1d_from_seqlen(seqlen=self.num_total_timesteps, dim=dim),
        # )
        self.re_embed = ContinuousSincosEmbed(dim=dim, ndim=1)

        self.re_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, self.cond_dim),
            nn.GELU(),
        )
        # init
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights == "truncnormal":
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, re):
        # checks + preprocess
        assert re.numel() == len(re)
        # timestep = timestep.flatten()
        # re = re.view(-1, 1).float()
        re = re.float()
        # # for rollout timestep is simply initialized as 0 -> repeat to batch dimension
        # if timestep.numel() == 1:
        #     timestep = timestep.repeat(re.numel())
        # embed
        # timestep_embed = self.timestep_mlp(self.timestep_embed[timestep])
        re_embed = self.re_mlp(self.re_embed(re))
        # return timestep_embed + re_embed
        return re_embed
