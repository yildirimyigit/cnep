import torch
import torch.nn as nn

class WTA_CNP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=1, n_max_obs=10, n_max_tar=10, num_layers=3, batch_size=32):
        pass

    def forward(self, obs, tar):
        # obs: (batch_size, n_o (<n_max_obs), input_dim+output_dim)
        # tar: (batch_size, n_t (<n_max_tar), input_dim)
        encoded_obs = self.encoder(obs)  # (batch_size, n_o (<n_max_obs), hidden_dim)
        encoded_rep = encoded_obs.mean(dim=1).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        repeated_encoded_rep = torch.repeat_interleave(encoded_rep, tar.shape[1], dim=1)  # each encoded_rep is repeated to match tar
        rep_tar = torch.cat([repeated_encoded_rep, tar], dim=-1)
        
        pred = self.decoder(rep_tar)
        
        return pred, encoded_rep

    def loss(self, pred, real):
        # pred: (batch_size, n_t (<n_max_tar), 2*output_dim)
        # real: (batch_size, n_t (<n_max_tar), output_dim)
        pred_mean = pred[:, :, :self.output_dim]
        pred_std = nn.functional.softplus(pred[:, :, self.output_dim:])

        pred_dist = torch.distributions.Normal(pred_mean, pred_std)

        return (-pred_dist.log_prob(real)).mean()