import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class WTA_CNP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, n_max_obs=10, n_max_tar=10, encoder_hidden_dims=[256,256,256],
                 num_decoders=4, decoder_hidden_dims=[128,128], batch_size=32, nll_coef=16.81, batch_entropy_coef=10.672, ind_entropy_coef=7.553, scale_coefs=False):
        super(WTA_CNP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_max_obs = n_max_obs
        self.n_max_tar = n_max_tar
        self.encoder_num_layers = len(encoder_hidden_dims)
        self.num_decoders = num_decoders
        self.decoder_num_layers = len(decoder_hidden_dims)
        self.batch_size = batch_size

        self.nll_coef = nll_coef
        self.batch_entropy_coef = batch_entropy_coef
        self.ind_entropy_coef = ind_entropy_coef

        self.do_scale = scale_coefs
        if self.do_scale:
            self.scale_coefs()

        #self.doubt_coef, self.batch_entropy_coef, self.ind_entropy_coef = self.calculate_coef()

        encoder_layers = []
        for i in range(self.encoder_num_layers-1):
            if i == 0:
                encoder_layers.append(nn.Linear(input_dim+output_dim, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(encoder_hidden_dims[-2], encoder_hidden_dims[-1]))
        self.encoder = nn.Sequential(*encoder_layers)

        self.decoders = nn.ModuleList()

        #TODO: fix (look at the decoder structure when num_layers=2 and num_layers=1. They produce the same network)
        for _ in range(num_decoders):
            decoder_layers = []
            if self.decoder_num_layers > 1:
                for i in range(self.decoder_num_layers-1):
                    if i == 0:
                        decoder_layers.append(nn.Linear(encoder_hidden_dims[-1]+input_dim, decoder_hidden_dims[i]))
                    else:
                        decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
                    decoder_layers.append(nn.ReLU())
            else:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1]+input_dim, decoder_hidden_dims[0]))

            decoder_layers.append(nn.Linear(decoder_hidden_dims[-1], output_dim*2))  # 2 for mean and std
            self.decoders.append(nn.Sequential(*decoder_layers))

        self.gate = nn.Sequential(
            nn.Linear(encoder_hidden_dims[-1], num_decoders),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs, tar, latent=False):
        # obs: (batch_size, n_o (<n_max_obs), input_dim+output_dim)
        # tar: (batch_size, n_t (<n_max_tar), input_dim)
        n_t = tar.shape[1]
        
        encoded_obs = self.encoder(obs)  # (batch_size, n_o (<n_max_obs), hidden_dim)

        encoded_rep = encoded_obs.mean(dim=1).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        repeated_encoded_rep = torch.repeat_interleave(encoded_rep, n_t, dim=1)  # each encoded_rep is repeated to match tar
        rep_tar = torch.cat([repeated_encoded_rep, tar], dim=-1)

        pred = torch.zeros(self.num_decoders, self.batch_size, n_t, self.output_dim*2, device=tar.device)  # tar is used to get device

        for i in range(self.num_decoders):
            pred[i] = self.decoders[i](rep_tar)

        gate_vals = self.gate(encoded_rep)  # (batch_size, num_decoders)
        if latent:
            return pred, gate_vals, encoded_rep
        
        return pred, gate_vals

    def loss(self, pred, gate_vals, real):
        # pred: (num_decoders, batch_size, n_t (<n_max_tar), 2*output_dim)
        # real: (batch_size, n_t (<n_max_tar), output_dim)
        # gate_vals: (batch_size, 1, num_decoders)

        pred_means = pred[:, :, :, :self.output_dim]
        pred_stds = torch.nn.functional.softplus(torch.exp(pred[:, :, :, self.output_dim:]))  # predicted value is log_std. In comb. with softplus to ensure positivity

        pred_dists = torch.distributions.Normal(pred_means, pred_stds)  # <num_decoders>-dimensional multivariate gaussian
        dec_loss = (-pred_dists.log_prob(real)).mean((-2, -1))  # (num_decoders, batch_size) - mean over tar and output_dim
        dec_ids = torch.argmax(gate_vals.squeeze(1), dim=-1).unsqueeze(0)
        losses = dec_loss.gather(0, dec_ids)  # 0:dim, loss per individual

        #############
        # Actual loss
        weighted_nll_per_ind = torch.mul(gate_vals.squeeze(1), dec_loss.T)  # (batch_size, num_decoders)
        nll = weighted_nll_per_ind.mean()  # scalar - mean over individuals and decoders

        #############
        # Overall entropy. We want to increase entropy; i.e. for a batch, the model should use all decoders not just one
        gate_means = torch.mean(gate_vals, dim=0).squeeze(-1).squeeze(-1)
        batch_entropy = self.entropy(gate_means)  # scalar

        #############
        # Gate std: sometimes all gates are the same, we want to penalize low std; i.e we want to increase std
        ind_entropy = self.entropy(gate_vals).mean()  # scalar

        return self.nll_coef*nll - self.batch_entropy_coef*batch_entropy + self.ind_entropy_coef*ind_entropy, torch.min(losses)
   
    def scale_coefs(self):
        if self.num_decoders == 1:
            self.batch_entropy_coef = torch.tensor([0])
            self.ind_entropy_coef = torch.tensor([0])
            self.nll_coef = torch.tensor([1.0])
            return
        high_entropy_base = torch.tensor([0.5, 0.5])
        high_entropy_base_value = self.entropy(high_entropy_base)
        high_entropy_current = torch.ones(1, self.num_decoders)/self.num_decoders
        high_entropy_current_value = self.entropy(high_entropy_current)

        batch_size_incurred_weight_change = torch.tensor(high_entropy_current_value.item()/high_entropy_base_value.item())

        self.batch_entropy_coef /= batch_size_incurred_weight_change
        self.ind_entropy_coef /= batch_size_incurred_weight_change
        self.nll_coef *= (torch.tensor(self.num_decoders)/2)  #torch.tensor(self.batch_size)
    
    def entropy(self, t: torch.Tensor):
        if t.nelement == 1:
            return torch.tensor(0.0)
        if torch.any(t<0):
            raise ValueError("log() not defined for negative values")
        if torch.any(t==0):
            t = t + 1e-10
            t = t/t.sum()
        return torch.sum(-t*torch.log(t), dim=-1)

    # Overloaded to() method to also move coefs to device
    def to(self, device):
        new_self = super(WTA_CNP, self).to(device)
        if self.do_scale:
            new_self.nll_coef = new_self.nll_coef.to(device)
            new_self.batch_entropy_coef = new_self.batch_entropy_coef.to(device)
            new_self.ind_entropy_coef = new_self.ind_entropy_coef.to(device)

        return new_self
