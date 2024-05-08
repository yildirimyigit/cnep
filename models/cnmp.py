import torch
import torch.nn as nn
import torch.nn.functional as F

class CNMP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, n_max=10, m_max=10, encoder_hidden_dims=[128,128,128], decoder_hidden_dims=[128,128,128], batch_size=32, device='cpu'):
        super(CNMP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_max = n_max
        self.m_max = m_max
        self.encoder_num_layers = len(encoder_hidden_dims)
        assert self.encoder_num_layers > 1, "Encoder must have more than 1 hidden layer"
        self.decoder_num_layers = len(decoder_hidden_dims)
        assert self.decoder_num_layers > 1, "Decoders must have more than 1 hidden layer"
        self.batch_size = batch_size
        self.device = device

        # Encoder
        e_layers = []
        e_layers.append(nn.Linear(input_dim+output_dim, encoder_hidden_dims[0]))
        e_layers.append(nn.ReLU())
        for i in range(1, self.encoder_num_layers-1):
            e_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
            e_layers.append(nn.ReLU())
        e_layers.append(nn.Linear(encoder_hidden_dims[-2], encoder_hidden_dims[-1]))

        self.encoder = nn.Sequential(*e_layers)

        # Decoder
        d_layers = []
        d_layers.append(nn.Linear(encoder_hidden_dims[-1]+input_dim, decoder_hidden_dims[0]))
        d_layers.append(nn.ReLU())
        for i in range(1, self.decoder_num_layers-1):
            d_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
            d_layers.append(nn.ReLU())
        d_layers.append(nn.Linear(decoder_hidden_dims[-1], output_dim*2))  # x2 for mean and std

        self.decoder = nn.Sequential(*d_layers)

        self.to(self.device)

    def forward(self, obs, tar, obs_mask, latent=False):
        # obs: (batch_size, n_max, input_dim+output_dim)
        # tar: (batch_size, m_max, input_dim)
        # obs_mask: (batch_size, n_max)

        # encoding
        encoded_obs = self.encoder(obs)  # (batch_size, n_max, encoder_hidden_dims[-1])
        obs_mask_exp = obs_mask.unsqueeze(-1).type_as(encoded_obs)  # (batch_size, n_max, 1)
        masked_encoded_obs = encoded_obs * obs_mask_exp  # (batch_size, n_max, encoder_hidden_dims[-1])

        # masked mean
        sum_masked_encoded_obs = masked_encoded_obs.sum(dim=1)  # (batch_size, encoder_hidden_dims[-1])
        sum_obs_mask = obs_mask_exp.sum(dim=1) # (batch_size, 1)
        r = (sum_masked_encoded_obs / sum_obs_mask).unsqueeze(1) # avg representations: (batch_size, 1, encoder_hidden_dims[-1])

        # repeat to concatenate with tar
        r_repeated = r.repeat(1, self.m_max, 1)
        rep_tar = torch.cat([r_repeated, tar], dim=-1)

        pred = self.decoder(rep_tar)  # (batch_size, m_max, output_dim*2)
        if latent:
            return pred, r
        
        return pred
    
    def val(self, obs, tar, obs_mask, latent=False):
        # obs: (batch_size, n_max, input_dim+output_dim)
        # tar: (batch_size, t_steps, input_dim)
        # obs_mask: (batch_size, n_max)

        # encoding
        encoded_obs = self.encoder(obs)
        obs_mask_exp = obs_mask.unsqueeze(-1).type_as(encoded_obs)  # (batch_size, n_max, 1)
        masked_encoded_obs = encoded_obs * obs_mask_exp  # (batch_size, n_max, encoder_hidden_dims[-1])

        # masked mean
        sum_masked_encoded_obs = masked_encoded_obs.sum(dim=1)  # (batch_size, encoder_hidden_dims[-1])
        sum_obs_mask = obs_mask_exp.sum(dim=1) # (batch_size, 1)
        r = (sum_masked_encoded_obs / sum_obs_mask).unsqueeze(1) # avg representations: (batch_size, 1, encoder_hidden_dims[-1])

        # repeat to concatenate with tar
        n_tar = tar.shape[1]
        r_repeated = r.repeat(1, n_tar, 1)
        rep_tar = torch.cat([r_repeated, tar], dim=-1)

        pred = self.decoder(rep_tar)  # (batch_size, m_max, output_dim*2)
        if latent:
            return pred, r
        
        return pred


    def loss(self, pred, real, tar_mask):
        # pred: (batch_size, m_max, 2*output_dim)
        # real: (batch_size, m_max, output_dim)
        # tar_mask: (batch_size, m_max)

        pred_mean = pred[:, :, :self.output_dim]
        pred_std = F.softplus(pred[:, :, self.output_dim:]) + 1e-6  # predicted value is std. In comb. with softplus and minor addition to ensure positivity

        pred_dist = torch.distributions.Normal(pred_mean, pred_std)

        tar_mask_expanded = tar_mask.unsqueeze(-1).expand_as(pred_mean)

        # Log probability under predicted distributions
        log_prob = -pred_dist.log_prob(real)

        # Only get the log_prob for unmasked targets
        masked_log_prob = log_prob * tar_mask_expanded.float()
        sum_masked_log_prob = masked_log_prob.sum(dim=-1).sum(dim=-1)  # (batch_size) - sum over tar and output_dim

        valid_counts = tar_mask_expanded.sum(dim=-1).sum(dim=-1)  # (batch_size)
        mean_log_probs = sum_masked_log_prob / valid_counts  # (batch_size)
        return mean_log_probs.mean()
