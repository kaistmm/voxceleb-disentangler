import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from pytorch_revgrad import RevGrad

class EnvDiscriminator(nn.Module):
    def __init__(self,  disc_dims, disc_out_dims, margin, **kwargs):
        super(EnvDiscriminator, self).__init__()
        self.margin     = margin
        self.input_dims = disc_dims
        self.nOut       = disc_out_dims

        self.grad_rev = RevGrad()

        self.classifier = th.nn.Sequential(*[
                            th.nn.BatchNorm1d(self.input_dims),
                            th.nn.ELU(inplace=True),
                            th.nn.Linear(self.input_dims, self.nOut),

                            th.nn.BatchNorm1d(self.nOut),
                            th.nn.ELU(inplace=True),
                            th.nn.Linear(self.nOut, self.nOut)])


    def reset_parameters(self):
        self.classifier.reset_parameters()

    def forward(self, x, use_grl=False):
        if use_grl:
            x = self.grad_rev(x)

        B, S, D = x.size()
        x = self.classifier(x.reshape(-1, D)).reshape(B, S, -1)

        ## calculate distances
        out_anchor   = F.normalize(x[:,0,:], p=2, dim=1)
        out_positive = F.normalize(x[:,1,:], p=2, dim=1)
        out_negative = F.normalize(x[:,2,:], p=2, dim=1)

        pos_dist    = F.pairwise_distance(out_anchor, out_positive)
        neg_dist    = F.pairwise_distance(out_anchor, out_negative)

        ## loss function
        nloss_env = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))

        return nloss_env


def MainModel(**kwargs):
    model = EnvDiscriminator(**kwargs)
    return model