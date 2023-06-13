# %%
import torch
import matplotlib.pyplot as plt

from models.wta_cnp import WTA_CNP
from data.data_generators import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# %%
# Hyperparameters
batch_size = 32
n_max_obs, n_max_tar = 6, 6

t_steps = 200
num_demos = 128
num_classes = 4
num_indiv = num_demos//num_classes  # number of demos per class
noise_clip = 0.0
dx, dy = 1, 1

num_val = 32
num_val_indiv = num_val//num_classes

colors = ['r', 'g', 'b', 'm']

# %%
# Generating the data
x = torch.linspace(0, 1, 200).repeat(num_indiv, 1)
y = torch.zeros(num_demos, t_steps, dy)

vx = torch.linspace(0, 1, 200).repeat(num_val_indiv, 1)
vy = torch.zeros(num_val, t_steps, dy)

generator_functions = [generate_sin, generate_cos, generate_cx_sigm, generate_reverse_cx_sigm]

for i in range(num_classes):
    noise = torch.clamp(torch.randn(x.shape)*1e-4**0.5, min=0) - noise_clip
    y[i*num_indiv:(i+1)*num_indiv] = torch.unsqueeze(generator_functions[i](x) + noise, 2)
    noise = torch.clamp(torch.randn(vx.shape)*1e-4**0.5, min=0) - noise_clip
    vy[i*num_val_indiv:(i+1)*num_val_indiv] = torch.unsqueeze(generator_functions[i](vx) + noise, 2)

x = torch.unsqueeze(x.repeat(num_classes, 1), 2)  # since dx = 1
vx = torch.unsqueeze(vx.repeat(num_classes, 1), 2)
print("X:", x.shape, "Y:", y.shape, "VX:", vx.shape, "VY:", vy.shape)

for i in range(num_indiv):
    plt.plot(x[i, :, 0], y[i, :, 0], 'r', alpha=0.3)
    plt.plot(x[i+num_indiv, :, 0], y[i+num_indiv, :, 0], 'g', alpha=0.3)
    plt.plot(x[i+2*num_indiv, :, 0], y[i+2*num_indiv, :, 0], 'b', alpha=0.3)
    plt.plot(x[i+3*num_indiv, :, 0], y[i+3*num_indiv, :, 0], 'magenta', alpha=0.3)
plt.show()

x, y = x.to(device), y.to(device)

# %%
def get_batch(x, y, traj_ids):
    n_t = torch.randint(1, n_max_tar, (1,)).item()
    n_o = torch.randint(1, n_max_obs, (1,)).item()

    obs = torch.zeros(batch_size, n_o, dx+dy, device=device)
    tar = torch.zeros(batch_size, n_t, dx, device=device)
    tar_val = torch.zeros(batch_size, n_t, dy, device=device)

    for i in range(len(traj_ids)):
        random_query_ids = torch.randperm(t_steps)
        o_ids = random_query_ids[:n_o]
        t_ids = random_query_ids[n_o:n_o+n_t]

        obs[i, :, :] = torch.cat((x[traj_ids[i], o_ids], y[traj_ids[i], o_ids]), dim=-1)
        tar[i, :, :] = x[traj_ids[i], t_ids]
        tar_val[i, :, :] = y[traj_ids[i], t_ids]

    # print("Obs:", obs.shape, "Tar:", tar.shape, "Tar_val:", tar_val.shape)
    return obs, tar, tar_val

def get_validation_batch(vx, vy, traj_ids, device=device):
    num_obs = torch.randint(1, n_max_obs, (1,)).item()

    obs = torch.zeros(batch_size, num_obs, dx+dy, device=device)
    tar = torch.zeros(batch_size, t_steps, dx, device=device)
    tar_val = torch.zeros(batch_size, t_steps, dy, device=device)

    for i in range(len(traj_ids)):
        random_query_ids = torch.randperm(t_steps)
        o_ids = random_query_ids[:num_obs]

        obs[i, :, :] = torch.cat((vx[traj_ids[i], o_ids], vy[traj_ids[i], o_ids]), dim=-1)
        tar[i, :, :] = vx[traj_ids[i]]
        tar_val[i, :, :] = vy[traj_ids[i]]

    return obs, tar, tar_val

# %%
import wandb

sweep_config = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_epoch_err'},
    'parameters': 
    {
        'nll_coeff': {'max': 10.0, 'min': 0.1},
        'other_loss_coeff': {'max': 10.0, 'min': 0.1}
     }
}

sweep_id = wandb.sweep(
  sweep=sweep_config, 
  project='wta-loss-components-sweep'
  )

# %%
import time

def model_train():
    run = wandb.init()
    nll_coeff = wandb.config.nll_coeff
    other_loss_coeff = wandb.config.other_loss_coeff

    model = WTA_CNP(1, 1, 6, 6, [128, 128, 256], num_decoders=4, decoder_hidden_dims=[256, 128, 128], batch_size=batch_size, nll_coeff=nll_coeff, other_loss_coeff=other_loss_coeff).to(device)
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

    file_name = int(time.time())

    epochs = 1_500_000
    epoch_iter = num_demos//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
    v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)
    avg_loss = 0

    val_per_epoch = 1000
    min_val_error = 1000000

    mse_loss = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0

        traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

        for i in range(epoch_iter):
            optimizer.zero_grad()
            obs, tar_x, tar_y = get_batch(x, y, traj_ids[i])
            pred, gate = model(obs, tar_x)
            loss, nll = model.loss(pred, gate, tar_y)
            loss.backward()
            optimizer.step()

            epoch_loss += nll.item()

        if epoch % val_per_epoch == 0:
            with torch.no_grad():
                v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
                
                val_epoch_err = 0

                for j in range(v_epoch_iter):
                    o, t, tr = get_validation_batch(vx, vy, v_traj_ids[j])
                    p, g = model(o, t)
                    dec_id = torch.argmax(g.squeeze(1), dim=-1)
                    vp_means = p[dec_id, torch.arange(batch_size), :, :dy]
                    val_epoch_err += mse_loss(vp_means, tr).item()

                if val_epoch_err < min_val_error:
                    min_val_error = val_epoch_err
                    print(f'New best: {min_val_error}')
                    torch.save(model.state_dict(), f'saved_models/wtacnp_synth_{file_name}.pt')

                wandb.log({
                    'epoch': epoch, 
                    'train_loss': avg_loss/99.0, 
                    'val_epoch_err': val_epoch_err
                })


        avg_loss += epoch_loss

        if epoch % 100 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, avg_loss/100))
            avg_loss = 0

# %%
wandb.agent(sweep_id, function=model_train, count=2)

# %%
# Testing the best model
# model = WTA_CNP(1, 1, 10, 10, [256, 256, 256], 4, [128, 128], batch_size)
# model.load_state_dict(torch.load(f'saved_models/wtacnp_synth_{file_name}.pt'))
# model.eval()

# o, t, tr = get_validation_batch(vx, vy)

# with torch.no_grad():
#     p, g = model(o, t)

# p, g = p.cpu().numpy(), g.cpu()
# t, tr = t.cpu().numpy(), tr.cpu().numpy()

# for i in range(batch_size):
#     dec_id = torch.argmax(g[i, :, :]).item()
#     print(dec_id)
#     plt.plot(range(t_steps), p[dec_id, i, :, 0], colors[dec_id], alpha=0.3)

# %%
### TESTING

# def generate_test_batch(n=4):
#     obs = torch.zeros(n, 2, 2)
#     obs[0] = torch.tensor([[0, 0], [1, 0]])
#     obs[1] = torch.tensor([[0, 0], [1, 1]])
#     obs[2] = torch.tensor([[0, 1], [1, 0]])
#     obs[3] = torch.tensor([[0, 1], [1, 1]])
#     tar = vx[:4, torch.arange(t_steps)]

#     return obs, tar


# model = WTA_CNP(1, 1, 6, 6, [128, 128, 256], num_decoders=4, decoder_hidden_dims=[256, 128, 128], batch_size=4)
# model.load_state_dict(torch.load(f'saved_models/wtacnp_synth_{file_name}.pt'))
# model.eval()

# o, t = generate_test_batch()

# # print(o)

# with torch.no_grad():
#     p, g = model(o, t)

# p, g = p.cpu().numpy(), g.cpu()

# for i in range(4):

#     # print(o[i, 0, 0], o[i, 0, 1], o[i, 1, 0], o[i, 1, 1])

#     dec_id = torch.argmax(g[i, :, :]).item()
#     plt.plot(range(t_steps), p[dec_id, i, :, 0], colors[dec_id], alpha=0.7)

#     plt.scatter(o[i, 0, 0]*200, o[i, 0, 1], color='k', marker='x')
#     plt.scatter(o[i, 1, 0]*200, o[i, 1, 1], color='k', marker='x')

# for i in range(32):
#     plt.plot(range(t_steps), tr[i, :, 0].cpu().numpy(), 'k', alpha=0.1, linestyle='dashed')

# plt.show()

# %%
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(pytorch_total_params)

# %%



