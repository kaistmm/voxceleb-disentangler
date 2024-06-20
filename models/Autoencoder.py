import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_dims, spk_dims, env_dims, **kwargs):
        super(AutoEncoder, self).__init__()

        self.input_dims = input_dims

        self.spk_dims  = spk_dims
        self.env_dims  = env_dims
        self.code_dims = self.spk_dims + self.env_dims
        
        self.encoder  = nn.Sequential(*[
                        nn.BatchNorm1d(self.input_dims),
                        nn.Linear(self.input_dims, self.code_dims)])

        self.decoder  = nn.Sequential(*[
                        nn.BatchNorm1d(self.code_dims),
                        nn.Linear(self.code_dims, self.input_dims),
                        ])


    def forward(self, x, feat_swp=False):
        S, B, D = x.size()

        x    = x.reshape(-1, D) # [S*B, D]
        code = self.encoder(x)  # [S*B, D']
        
        scode = F.normalize(code[:, :self.spk_dims], p=2, dim=-1).reshape(S, B, -1) # [S,B,D'']
        ecode = F.normalize(code[:, self.spk_dims:], p=2, dim=-1).reshape(S, B, -1) # [S,B,D'']

        if feat_swp:
            scode = scode[[2,0,1]]

        code_  = torch.concat([scode, ecode], dim=-1).reshape(S*B, -1) # [S,B,D'] -> [S*B,D']
        
        output = self.decoder(code_).reshape(S, B, D)

        scode = code[:,:self.spk_dims].reshape(S,B,-1) # [S,B,D"]
        ecode = code[:,self.spk_dims:].reshape(S,B,-1) # [S,B,D"]

        return output, scode, ecode
    
    def disentangle_feat(self, x):
        """
        x : [B, D]
        """

        code  = self.encoder(x)
        scode = code[:,:self.spk_dims]
        ecode = code[:,self.spk_dims:] 

        return scode, ecode


def MainModel(input_dims, spk_dims, env_dims, **kwargs):
    model = AutoEncoder(input_dims, spk_dims, env_dims, **kwargs)
    return model