import torch
import torch.nn as nn

class WTA_CNP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, n_max_obs=10, n_max_tar=10, encoder_hidden_dims=[256,256,256],
                 num_decoders=4, decoder_hidden_dims=[128,128], batch_size=32):
        super(WTA_CNP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_max_obs = n_max_obs
        self.n_max_tar = n_max_tar
        self.encoder_num_layers = len(encoder_hidden_dims)
        self.num_decoders = num_decoders
        self.decoder_num_layers = len(decoder_hidden_dims)
        self.batch_size = batch_size

        encoder_layers = []
        for i in range(self.encoder_num_layers-1):
            if i == 0:
                encoder_layers.append(nn.Linear(input_dim+output_dim, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(encoder_hidden_dims[-2], encoder_hidden_dims[-1]))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self.decoder_num_layers-1):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1]+input_dim, decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_hidden_dims[-1], output_dim*2))

        self.decoders = nn.ModuleList([nn.Sequential(*decoder_layers) for _ in range(num_decoders)])

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

        nll = torch.matmul(gate_vals, dec_loss).mean()  # (batch_size, batch_size)

        #############

        doubt = torch.prod(gate_vals, dim=1).mean()  # scalar

        return nll + doubt