import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Batched and masked implementation of the CNEP model
'''
class CNEP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, n_max_obs=10, n_max_tar=10, encoder_hidden_dims=[256,256,256],
                 num_decoders=4, decoder_hidden_dims=[128,128], batch_size=32, nll_coef=16.81, batch_entropy_coef=10.672, ind_entropy_coef=7.553, scale_coefs=False, device='cuda') -> None:
        super(CNEP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_max = n_max_obs
        self.m_max = n_max_tar
        self.encoder_num_layers = len(encoder_hidden_dims)
        assert self.encoder_num_layers > 1, "Encoder must have more than 1 hidden layer"
        self.num_decoders = num_decoders
        self.decoder_num_layers = len(decoder_hidden_dims)
        assert self.decoder_num_layers > 1, "Decoders must have more than 1 hidden layer"
        self.batch_size = batch_size
        self.device = device

        self.nll_coef = nll_coef
        self.batch_entropy_coef = batch_entropy_coef
        self.ind_entropy_coef = ind_entropy_coef

        self.do_scale = scale_coefs
        if self.do_scale:
            self.scale_coefs()

        # Encoder
        layers = []
        layers.append(nn.Linear(input_dim+output_dim, encoder_hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(1, self.encoder_num_layers-1):
            layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(encoder_hidden_dims[-2], encoder_hidden_dims[-1]))

        self.encoder = nn.Sequential(*layers)

        # Decoders
        self.decoders = nn.ModuleList()
        for _ in range(num_decoders):
            layers = []
            layers.append(nn.Linear(encoder_hidden_dims[-1]+input_dim, decoder_hidden_dims[0]))
            layers.append(nn.ReLU())
            for i in range(1, self.decoder_num_layers-1):
                layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(decoder_hidden_dims[-1], output_dim*2))  # x2 for mean and std
            self.decoders.append(nn.Sequential(*layers))

        # Gate
        self.gate = nn.Sequential(
            nn.Linear(encoder_hidden_dims[-1], num_decoders),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs, tar, obs_mask, latent=False):
        # obs: (batch_size, n_max_obs, input_dim+output_dim)
        # tar: (batch_size, n_max_tar, input_dim)
        # obs_mask: (batch_size, n_max_obs)

        # encoding
        encoded_obs = self.encoder(obs)  # (batch_size,  encoder_hidden_dims[-1])
        obs_mask_exp = obs_mask.unsqueeze(-1).type_as(encoded_obs)  # (batch_size, n_max_obs, 1)
        masked_encoded_obs = encoded_obs * obs_mask_exp  # (batch_size, n_max_obs, encoder_hidden_dims[-1])

        # masked mean
        sum_masked_encoded_obs = masked_encoded_obs.sum(dim=1)  # (batch_size, encoder_hidden_dims[-1])
        sum_obs_mask = obs_mask_exp.sum(dim=1) # (batch_size, 1)
        r = (sum_masked_encoded_obs / sum_obs_mask).unsqueeze(1) # (batch_size, 1, encoder_hidden_dims[-1])

        # decoding
        tar_num = tar.shape[1]
        r_repeated = r.repeat(1, tar_num, 1)
        rep_tar = torch.cat([r_repeated, tar], dim=-1)

        pred = torch.zeros(self.num_decoders, self.batch_size, tar_num, self.output_dim*2, device=self.device)

        for i in range(self.num_decoders):
            pred[i] = self.decoders[i](rep_tar)

        gate_vals = (self.gate(r)).squeeze(1)  # (batch_size, num_decoders)
        if latent:
            return pred, gate_vals, r
        
        return pred, gate_vals

    def loss(self, pred, gate_vals, real, tar_mask):
        # pred: (num_decoders, batch_size, n_max_tar, 2*output_dim)
        # real: (batch_size, n_max_tar, output_dim)
        # gate_vals: (batch_size, num_decoders)
        # tar_mask: (batch_size, n_max_tar)

        pred_mean = pred[:, :, :, :self.output_dim]
        pred_std = F.softplus(pred[:, :, :, self.output_dim:]) + 1e-6  # predicted value is std. In comb. with softplus and minor addition to ensure positivity

        pred_dist = torch.distributions.Normal(pred_mean, pred_std)

        real_expanded = real.unsqueeze(0).expand_as(pred_mean)
        tar_mask_expanded = tar_mask.unsqueeze(0).unsqueeze(-1).expand_as(pred_mean)

        # Log probability under predicted distributions
        log_prob = -pred_dist.log_prob(real_expanded)

        # Only get the log_prob for unmasked targets
        masked_log_prob = log_prob * tar_mask_expanded.float()
        sum_masked_log_prob = masked_log_prob.sum(dim=-1).sum(dim=-1)  # (num_decoders, batch_size) - sum over tar and output_dim

        valid_counts = tar_mask_expanded.sum(dim=-1).sum(dim=-1)  # (num_decoders, batch_size)
        mean_log_prob = sum_masked_log_prob / valid_counts  # (num_decoders, batch_size)

        dec_ids = torch.argmax(gate_vals, dim=-1).unsqueeze(0)
        losses = mean_log_prob.gather(0, dec_ids)  # 0:dim, loss per individual

        #############
        # Actual loss
        weighted_nll_per_ind = torch.mul(gate_vals, mean_log_prob.T)  # (batch_size, num_decoders)
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
        new_self = super(CNEP, self).to(device)
        if self.do_scale:
            new_self.nll_coef = new_self.nll_coef.to(device)
            new_self.batch_entropy_coef = new_self.batch_entropy_coef.to(device)
            new_self.ind_entropy_coef = new_self.ind_entropy_coef.to(device)

        return new_self
