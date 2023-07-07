import torch
import torch.nn as nn
import torch.nn.functional as F

class WTA_CNP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, n_max_obs=10, n_max_tar=10, encoder_hidden_dims=[256,256,256],
                 num_decoders=4, decoder_hidden_dims=[128,128], batch_size=32, nll_coeff=10.0, other_loss_coeff=2e-18):
        super(WTA_CNP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_max_obs = n_max_obs
        self.n_max_tar = n_max_tar
        self.encoder_num_layers = len(encoder_hidden_dims)
        self.num_decoders = num_decoders
        self.decoder_num_layers = len(decoder_hidden_dims)
        self.batch_size = batch_size

        self.nll_coeff = nll_coeff
        self.other_loss_coeff = other_loss_coeff

        self.doubt_coef, self.entropy_coef, self.gate_std_coef = self.calculate_coef()

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

    def forward(self, obs, tar):
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

        return pred, gate_vals

    def loss(self, pred, gate_vals, real):
        # pred: (num_decoders, batch_size, n_t (<n_max_tar), 2*output_dim)
        # real: (batch_size, n_t (<n_max_tar), output_dim)

        pred_means = pred[:, :, :, :self.output_dim]
        pred_stds = nn.functional.softplus(pred[:, :, :, self.output_dim:])

        pred_dists = torch.distributions.Normal(pred_means, pred_stds)  # <num_decoders>-many gaussians
        dec_loss = (-pred_dists.log_prob(real)).mean((-2, -1))  # (num_decoders, batch_size) - mean over tar and output_dim

        #############
        # Actual loss
        nll = torch.matmul(gate_vals, dec_loss).mean()  # (batch_size, batch_size).mean() = scalar

        #############
        # Doubt is defined over individual gates. We want to penalize the model for being unsure about a single prediction; i.e we want to decrease doubt
        doubt = (torch.prod(gate_vals, dim=-1)).mean()  # scalar

        #############
        # Overall entropy. We want to increase entropy; i.e. for a batch, the model should use all decoders not just one
        gate_means = torch.mean(gate_vals, dim=0).squeeze(-1).squeeze(-1)
        entropy = torch.distributions.Categorical(probs=gate_means).entropy()  # scalar

        #############
        # Gate std: sometimes all gates are the same, we want to penalize low std; i.e we want to increase std
        gate_std = torch.std(gate_vals)

        # return self.nll_coeff*nll + self.other_loss_coeff*(doubt*self.doubt_coef - entropy*self.entropy_coef - gate_std*self.gate_std_coef), nll  # 4, 0.1 for increasing the importance of nll
        return 45*nll + (doubt*self.doubt_coef - entropy*self.entropy_coef - gate_std*self.gate_std_coef), nll  # 4, 0.1 for increasing the importance of nll
        # return 5*nll + doubt - entropy - gate_std, nll  # 4, 0.1 for increasing the importance of nll

    # def loss(self, pred, gate_vals, real):
    #     # pred: (num_decoders, batch_size, n_t (<n_max_tar), 2*output_dim)
    #     # gate_vals: (batch_size, num_decoders)
    #     # real: (batch_size, n_t (<n_max_tar), output_dim)

    #     pred_means = pred[:, :, :, :self.output_dim]  # (num_decoders, batch_size, n_t (<n_max_tar), output_dim)
    #     pred_stds = nn.functional.softplus(pred[:, :, :, self.output_dim:])

    #     pred_dists = torch.distributions.Normal(pred_means, pred_stds)  # <num_decoders>-many gaussians. Not a list but a singleGaussian Mixture Model object
    #     dec_loss = (-pred_dists.log_prob(real)).mean((-2, -1))  # (num_decoders, batch_size) - mean over tar and output_dim

    #     #############
    #     # Actual loss
    #     nll = torch.matmul(gate_vals, dec_loss).mean()  # (batch_size, batch_size).mean() = scalar
    #     # nll = dec_loss[torch.argmax(gate_vals)]

    #     #############
    #     # Overall entropy. We want to increase entropy; i.e. for a batch, the model should use all decoders not just one
    #     gate_means = torch.mean(gate_vals, dim=0).squeeze(-1).squeeze(-1)
    #     entropy = torch.distributions.Categorical(probs=gate_means).entropy()  # scalar

    #     #############
    #     # Overall mutual information. We want to increase mutual information to encourage the model to use all decoders
    #     mutual_info = torch.zeros(self.num_decoders, self.num_decoders, device=gate_vals.device)
    #     for i in range(self.num_decoders-1):
    #         dist_i = torch.distributions.Normal(pred_means[i], pred_stds[i])
    #         for j in range(i+1, self.num_decoders):
    #             dist_j = torch.distributions.Normal(pred_means[j], pred_stds[j])
    #             mutual_info[i, j] = torch.distributions.kl.kl_divergence(dist_i, dist_j).mean() + torch.distributions.kl.kl_divergence(dist_j, dist_i).mean()
    #     # for i in range(self.num_decoders-1):
    #     #     for j in range(i+1, self.num_decoders):
    #     #         mutual_info[i, j] = 0.5 * (F.kl_div(pred_means[i], pred_means[j]).mean() + F.kl_div(pred_means[j], pred_means[i]).mean())
        
    #     # return nll - distance.sum()/400000, nll  # 4, 0.1 for increasing the importance of nll
    #     return self.nll_coeff * nll - self.other_loss_coeff * mutual_info.sum() + self.entropy_coef * entropy, nll
    #     # return 5*nll + doubt - entropy - gate_std, nll  # 4, 0.1 for increasing the importance of nll
        
    def calculate_coef(self):
        # Doubt, entropy and std need to be scaled

        # Doubt coefficient, best: [1, 0, ..., 0], worst: [1/num_decoders, ..., 1/num_decoders]
        good_individual, bad_individual = torch.eye(1, self.num_decoders), torch.ones(1, self.num_decoders)/self.num_decoders
        # doubt_min, doubt_max = torch.prod(good_individual).mean(), torch.prod(bad_individual).mean()
        doubt_min, doubt_max = torch.distributions.Categorical(probs=good_individual).entropy(), torch.distributions.Categorical(probs=bad_individual).entropy()

        doubt_coef = 1/(doubt_max - doubt_min)

        # Entropy coefficient, best: [1/num_decoders, ..., 1/num_decoders], worst: [1, 0, ..., 0]
        # if self.batch_size > 1:
        #     good_gate_distr, bad_gate_distr = torch.ones(1, self.num_decoders)/self.num_decoders, torch.eye(1, self.num_decoders)
        #     entropy_max, entropy_min = torch.distributions.Categorical(probs=good_gate_distr).entropy(), torch.distributions.Categorical(probs=bad_gate_distr).entropy()
            
        #     entropy_coef = 1/(entropy_max - entropy_min)
        # else:
        #     entropy_coef = torch.tensor([0])

        good_gate_distr, bad_gate_distr = torch.ones(1, self.num_decoders)/self.num_decoders, torch.eye(1, self.num_decoders)
        entropy_max, entropy_min = torch.distributions.Categorical(probs=good_gate_distr).entropy(), torch.distributions.Categorical(probs=bad_gate_distr).entropy()
        
        if self.batch_size > 1:
            entropy_coef = 1/(entropy_max - entropy_min)
        else:
            entropy_coef = torch.tensor([0])

        # Gate std coefficient, best: eye(num_decoders, num_decoder).repeat(batch_size//num_decoders, 1), worst: 1
        if self.batch_size > self.num_decoders:
            good_gate_distr = torch.eye(self.num_decoders, self.num_decoders).repeat(self.batch_size//self.num_decoders, 1)
            bad_gate_distr = torch.ones_like(good_gate_distr)/self.num_decoders

        else:
            good_gate_distr = torch.eye(self.batch_size, self.num_decoders)
            bad_gate_distr = torch.ones_like(good_gate_distr)/self.num_decoders

        gate_std_max, gate_std_min = torch.std(good_gate_distr), torch.std(bad_gate_distr)
        gate_std_coef = 1/(gate_std_max - gate_std_min)

        return doubt_coef, entropy_coef, gate_std_coef
    
    # Overloaded to() method to also move doubt_coef, entropy_coef and gate_std_coef
    def to(self, device):
        new_self = super(WTA_CNP, self).to(device)
        new_self.doubt_coef = new_self.doubt_coef.to(device)
        new_self.entropy_coef = new_self.entropy_coef.to(device)
        new_self.gate_std_coef = new_self.gate_std_coef.to(device)

        return new_self