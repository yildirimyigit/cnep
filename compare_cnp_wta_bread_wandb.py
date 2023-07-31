# %%
from models.cnp import CNP
from models.wta_cnp import WTA_CNP

from data.data_generators import *
import torch

if torch.cuda.is_available():
    device_wta = torch.device("cuda:0")
    device_cnp = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda:0")
else:
    device_wta = torch.device("cpu")
    device_cnp = torch.device("cpu")

# %%
batch_size = 2
n_max_obs, n_max_tar = 6, 6

t_steps = 200
num_demos = 2
num_classes = 2
num_indiv = num_demos//num_classes  # number of demos per class
noise_clip = 0.0
dx, dy = 1, 1

num_val = 2
num_val_indiv = num_val//num_classes

colors = ['r', 'b']

num_inc = 0
num_exc = 0

fixed_obs_ratio = 0.00000

# %%
x = torch.linspace(0, 1, 200).repeat(num_indiv, 1)
y = torch.zeros(num_demos, t_steps, dy)

vx = torch.linspace(0, 1, 200).repeat(num_val_indiv, 1)
vy = torch.zeros(num_val, t_steps, dy)

# noise = torch.clamp(torch.randn(x.shape)*1e-7**0.5, min=0) - noise_clip
# coeff = (torch.rand(num_indiv)*0.75+0.25).unsqueeze(-1)
coeff = (torch.tensor([0.6])).unsqueeze(-1)
y[:num_indiv] = torch.unsqueeze(generate_sin(x)*coeff, 2)
y[num_indiv:] = -1 * y[:num_indiv]

# coeff = (torch.rand(num_val_indiv)*0.75+0.25).unsqueeze(-1)
# noise = torch.clamp(torch.randn(vx.shape)*1e-4**0.5, min=0) - noise_clip
vy[:num_val_indiv] = torch.unsqueeze(generate_sin(vx)*coeff, 2)
vy[num_val_indiv:] = -1 * vy[:num_val_indiv]

x = torch.unsqueeze(x.repeat(num_classes, 1), 2)  # since dx = 1
vx = torch.unsqueeze(vx.repeat(num_classes, 1), 2)
print("X:", x.shape, "Y:", y.shape, "VX:", vx.shape, "VY:", vy.shape)

x0, y0 = x.to(device_wta), y.to(device_wta)
x1, y1 = x.to(device_cnp), y.to(device_cnp)

# %%
def get_batch(x, y, traj_ids, device=device_wta):
    global num_inc, num_exc

    # if the following holds, we condition on 0, -1 or [0, -1] to estimate the rest
    # if not, we condition on random-numbered random points
    if torch.rand(1) < fixed_obs_ratio:
        num_inc += 1
        
        if torch.rand(1) < 0.33:
            n_o = 1  # [0]
            n_t = t_steps - 1
            o_ids = torch.tensor([0])
            t_ids = torch.randperm(t_steps-1)+1  # excluding [0]
        elif torch.rand(1) < 0.66:
            n_o = 1  # [-1]
            n_t = t_steps - 1
            o_ids = torch.tensor([-1])
            t_ids = torch.randperm(t_steps-1)  # excluding [-1]
        else:
            n_o = 2  # [0, -1]
            n_t = t_steps - 2
            o_ids = torch.tensor([0, -1])
            t_ids = torch.randperm(t_steps-2)+1  # excluding [0] and [-1]

        fixed = True
    
    else:
        num_exc += 1
        n_o = torch.randint(1, n_max_obs, (1,)).item()
        n_t = torch.randint(1, n_max_tar, (1,)).item()
        fixed = False

    tar = torch.zeros(batch_size, n_t, dx, device=device)
    tar_val = torch.zeros(batch_size, n_t, dy, device=device)
    obs = torch.zeros(batch_size, n_o, dx+dy, device=device)
    
    for i in range(len(traj_ids)):
        if not fixed:
            random_query_ids = torch.randperm(t_steps) 
            
            o_ids = random_query_ids[:n_o]
            t_ids = random_query_ids[n_o:n_o+n_t]

        obs[i, :, :] = torch.cat((x[traj_ids[i], o_ids], y[traj_ids[i], o_ids]), dim=-1)
        tar[i, :, :] = x[traj_ids[i], t_ids]
        tar_val[i, :, :] = y[traj_ids[i], t_ids]

    return obs, tar, tar_val

def get_validation_batch(vx, vy, traj_ids, device=device_wta):
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

import wandb

sweep_config = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss_wta'},
    'parameters': 
    {
        'nll_coef': {'max': 20.0, 'min': 0.0},
        'entropy_coef': {'max': 20.0, 'min': 0.0},
        'gate_std_coef': {'max': 20.0, 'min': 0.0}
     }
}

# sweep_id = wandb.sweep(
#  sweep=sweep_config,
#  project='bread-loss-components-sweep-3'
#  )

sweep_id = 'ido0meh1'

import time
import os


def train():
    run = wandb.init()
    nll_coef = wandb.config.nll_coef
    entropy_coef = wandb.config.entropy_coef
    gate_std_coef = wandb.config.gate_std_coef

    model_wta = WTA_CNP(1, 1, n_max_obs, n_max_tar, [128, 128, 128], num_decoders=2, decoder_hidden_dims=[128, 128, 128], batch_size=batch_size,
                        nll_coef=nll_coef, entropy_coef=entropy_coef, gate_std_coef=gate_std_coef).to(device_wta)
    optimizer_wta = torch.optim.Adam(lr=1e-4, params=model_wta.parameters())

    model_cnp = CNP(input_dim=1, hidden_dim=128, output_dim=1, n_max_obs=n_max_obs, n_max_tar=n_max_tar, num_layers=3, batch_size=batch_size).to(device_cnp)
    optimizer_cnp = torch.optim.Adam(lr=1e-4, params=model_cnp.parameters())

    timestamp = int(time.time())
    root_folder = f'outputs/bread/{str(timestamp)}/'

    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    if not os.path.exists(f'{root_folder}saved_models/'):
        os.makedirs(f'{root_folder}saved_models/')

    if not os.path.exists(f'{root_folder}img/'):
        os.makedirs(f'{root_folder}img/')

    torch.save(y, f'{root_folder}y.pt')


    epochs = 300_000
    epoch_iter = num_demos//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
    v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)
    avg_loss_wta, avg_loss_cnp = 0, 0

    val_per_epoch = 1000
    min_val_loss_wta, min_val_loss_cnp = 1000000, 1000000

    mse_loss = torch.nn.MSELoss()

    training_loss_wta, validation_error_wta = [], []
    training_loss_cnp, validation_error_cnp = [], []

    wta_tr_loss_path = f'{root_folder}wta_training_loss.pt'
    wta_val_err_path = f'{root_folder}wta_validation_error.pt'
    cnp_tr_loss_path = f'{root_folder}cnp_training_loss.pt'
    cnp_val_err_path = f'{root_folder}cnp_validation_error.pt'

    for epoch in range(epochs):
        epoch_loss_wta, epoch_loss_cnp = 0, 0

        traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

        for i in range(epoch_iter):
            optimizer_wta.zero_grad()
            optimizer_cnp.zero_grad()

            obs_wta, tar_x_wta, tar_y_wta = get_batch(x, y, traj_ids[i], device_wta)
            obs_cnp, tar_x_cnp, tar_y_cnp = obs_wta.clone(), tar_x_wta.clone(), tar_y_wta.clone()

            pred_wta, gate_wta = model_wta(obs_wta, tar_x_wta)
            pred_cnp, encoded_rep_cnp = model_cnp(obs_cnp, tar_x_cnp)

            loss_wta, wta_nll = model_wta.loss(pred_wta, gate_wta, tar_y_wta)

            loss_wta.backward()
            optimizer_wta.step()

            loss_cnp = model_cnp.loss(pred_cnp, tar_y_cnp)
            loss_cnp.backward()
            optimizer_cnp.step()

            epoch_loss_wta += wta_nll.item()
            epoch_loss_cnp += loss_cnp.item()

        training_loss_wta.append(epoch_loss_wta)
        training_loss_cnp.append(epoch_loss_cnp)

        if epoch % val_per_epoch == 0:
            with torch.no_grad():
                v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
                val_loss_wta, val_loss_cnp = 0, 0

                for j in range(v_epoch_iter):
                    o_wta, t_wta, tr_wta = get_validation_batch(vx, vy, v_traj_ids[j], device=device_wta)
                    o_cnp, t_cnp, tr_cnp = o_wta.clone(), t_wta.clone(), tr_wta.clone()

                    p_wta, g_wta = model_wta(o_wta, t_wta)
                    dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
                    vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
                    val_loss_wta += mse_loss(vp_means, tr_wta).item()

                    pred_cnp, encoded_rep = model_cnp(o_cnp, t_cnp)
                    val_loss_cnp += mse_loss(pred_cnp[:, :, :model_cnp.output_dim], tr_cnp)


                validation_error_wta.append(val_loss_wta)
                if val_loss_wta < min_val_loss_wta:
                    min_val_loss_wta = val_loss_wta
                    print(f'(WTA)New best: {min_val_loss_wta}')
                    torch.save(model_wta.state_dict(), f'{root_folder}saved_models/wta_on_synth.pt')

                validation_error_cnp.append(val_loss_cnp.item())
                if val_loss_cnp < min_val_loss_cnp:
                    min_val_loss_cnp = val_loss_cnp
                    print(f'(CNP)New best: {min_val_loss_cnp}')
                    torch.save(model_cnp.state_dict(), f'{root_folder}saved_models/cnp_on_synth.pt')

                wandb.log({
                    'epoch': epoch, 
                    'train_loss': avg_loss_wta/100.0, 
                    'val_loss_wta': val_loss_wta
                })

        avg_loss_wta += epoch_loss_wta
        avg_loss_cnp += epoch_loss_cnp

        if epoch % 100 == 0:
            print("Epoch: {}, WTA-Loss: {}, CNP-Loss: {}".format(epoch, avg_loss_wta/100, avg_loss_cnp/100))
            avg_loss_wta, avg_loss_cnp = 0, 0

        if epoch % 100000:
            torch.save(torch.Tensor(training_loss_wta), wta_tr_loss_path)
            torch.save(torch.Tensor(validation_error_wta), wta_val_err_path)
            torch.save(torch.Tensor(training_loss_cnp), cnp_tr_loss_path)
            torch.save(torch.Tensor(validation_error_cnp), cnp_val_err_path)

wandb.agent(sweep_id, function=train, count=100, project='bread-loss-components-sweep-3')
