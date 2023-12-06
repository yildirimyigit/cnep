# %%
import h5py
import os

root = '/home/yigit/projects/mbcnp/data/raw/mocapact/'
files = []

# Iterate directory
for file_path in os.listdir(root):
    if file_path.endswith('.hdf5') and os.path.isfile(os.path.join(root, file_path)):
        # add filename to list
        files.append(file_path)
print(files)

# %%
import numpy as np

desired_observables = ['actuator_activation', 'joints_pos', 'joints_vel', 'sensors_gyro', 'end_effectors_pos', 
                       'sensors_torque', 'sensors_touch', 'sensors_velocimeter', 'world_zaxis']

def get_obs_indices(path):
    indices = []

    f = h5py.File(path, 'r+')
    walker_obs_dict = f['observable_indices']['walker']
    for k in walker_obs_dict.keys():
        if k in desired_observables:
            dum = walker_obs_dict[k][:]
            indices.extend(dum)
    f.close()

    return np.array(indices)

# Get indices
indices = get_obs_indices(os.path.join(root, 'CMU_049_06.hdf5'))


# %%
#region read mocapact data
full_obs, full_act = [], []

for file in files:
    fp = os.path.join(root, file)
    # Open file
    f = h5py.File(fp, 'r+')

    demos = {}

    num_start_rollouts = f['n_start_rollouts'][()]  # concatenate snippets to create this many rollouts
    for i in range(num_start_rollouts):
        demos.update({i: {}})
        demos[i].update({'obs': {}})
        demos[i].update({'act': {}})
    
    num_snippets = 0
    for key in f.keys():
        if key.startswith('CMU_'):
            num_snippets += 1

    for key in f.keys():
        if key.startswith('CMU_'):
            start, end = int(key.split('-')[-2]), int(key.split('-')[-1])
            for i in range(num_start_rollouts):
                obs = np.array(f[key][str(i)]['observations']['proprioceptive'])
                act = np.array(f[key][str(i)]['actions'])
                for j in range(len(act)):
                    demos[i]['obs'].update({start+j: obs[j, indices]})
                    demos[i]['act'].update({start+j: act[j]})

    for key in f.keys():
        for i in range(num_start_rollouts):
            if key.startswith('CMU_') and f[key]['early_termination'][i] == True:
                if i in demos.keys():
                    demos.pop(i)

    for key in demos.keys():
        full_obs.append(np.array(list(demos[key]['obs'].values())))
        full_act.append(np.array(list(demos[key]['act'].values())))

    f.close()

# print(len(full_obs), len(full_act))
#endregion
min_length = 1000
for i in range(len(full_obs)):
    if len(full_obs[i]) < min_length:
        min_length = len(full_obs[i])

processed_obs, processed_act = [], []
for i in range(len(full_obs)):
    processed_obs.append(full_obs[i][np.linspace(0, len(full_obs[i])-1, min_length, dtype=int)])
    processed_act.append(full_act[i][np.linspace(0, len(full_obs[i])-1, min_length, dtype=int)])


# %%
from models.wta_cnp import WTA_CNP
import torch

def get_available_gpu_with_most_memory():
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)  # Switch to the GPU to accurately measure memory
        gpu_memory.append((i, torch.cuda.memory_stats()['reserved_bytes.all.current'] / (1024 ** 2)))

    gpu_memory.sort(key=lambda x: x[1], reverse=True)

    return gpu_memory[0][0]

if torch.cuda.is_available():
    available_gpu = get_available_gpu_with_most_memory()
    if available_gpu == 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{available_gpu}")
else:
    device = torch.device("cpu")

print("Device :", device)

###

torch.set_float32_matmul_precision('high')

# %%
batch_size = 2
n_max_obs, n_max_tar = 6, 6

t_steps = min_length
num_val = 4
num_demos = len(full_obs)-num_val
num_classes = 2
num_indiv = num_demos//num_classes  # number of demos per class

dx, dy = len(indices), len(full_act[0][0])

num_val_indiv = num_val//num_classes

colors = ['tomato', 'aqua']

# %%
x = torch.zeros(num_demos, t_steps, dx, device=device)
y = torch.zeros(num_demos, t_steps, dy, device=device)
vx = torch.zeros(num_val, t_steps, dx, device=device)
vy = torch.zeros(num_val, t_steps, dy, device=device)

vind = torch.cat((torch.randint(0, num_indiv, (num_val_indiv, 1)), torch.randint(num_indiv, num_demos, (num_val_indiv, 1))), dim=0)
tr_ctr, val_ctr = 0, 0

print(vind)

for i in range(len(full_obs)):
    if i in vind:
        print(i, '*****')
        vx[val_ctr] = torch.tensor(processed_obs[i], dtype=torch.float32)
        vy[val_ctr] = torch.tensor(processed_act[i], dtype=torch.float32)
        val_ctr += 1
    else:
        print(i)
        x[tr_ctr] = torch.tensor(processed_obs[i], dtype=torch.float32)
        y[tr_ctr] = torch.tensor(processed_act[i], dtype=torch.float32)
        tr_ctr += 1

print("X:", x.shape, "Y:", y.shape, "VX:", vx.shape, "VY:", vy.shape)

# %%
def get_batch(x, y, traj_ids, device=device):
    n_o = torch.randint(1, n_max_obs, (1,)).item()
    n_t = torch.randint(1, n_max_tar, (1,)).item()
    
    tar = torch.zeros(batch_size, n_t, dx, device=device)
    tar_val = torch.zeros(batch_size, n_t, dy, device=device)
    obs = torch.zeros(batch_size, n_o, dx+dy, device=device)

    for i in range(len(traj_ids)):
        random_query_ids = torch.randperm(t_steps)
        
        o_ids = random_query_ids[:n_o]
        t_ids = random_query_ids[n_o:n_o+n_t]

        obs[i, :, :] = torch.cat((x[traj_ids[i], o_ids], y[traj_ids[i], o_ids]), dim=-1)
        tar[i, :, :] = x[traj_ids[i], t_ids]
        tar_val[i, :, :] = y[traj_ids[i], t_ids]

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
model_wta_ = WTA_CNP(dx, dy, n_max_obs, n_max_tar, [1024, 1024, 1024], num_decoders=2, decoder_hidden_dims=[512, 512, 512], batch_size=batch_size, scale_coefs=True).to(device)
optimizer_wta = torch.optim.Adam(lr=5e-5, params=model_wta_.parameters())

if torch.__version__ >= "2.0":
    model_wta = torch.compile(model_wta_)

# %%
import time
import os

timestamp = int(time.time())
root_folder = f'outputs/experimental/{dy}D/{str(timestamp)}/'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f'{root_folder}saved_models/'):
    os.makedirs(f'{root_folder}saved_models/')

# if not os.path.exists(f'{root_folder}img/'):
#     os.makedirs(f'{root_folder}img/')

torch.save(y, f'{root_folder}y.pt')


epochs = 50_000_000
epoch_iter = num_demos//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)
avg_loss_wta = 0

val_per_epoch = 1000
min_val_loss_wta = 1000000

mse_loss = torch.nn.MSELoss()

training_loss_wta, validation_error_wta = [], []

wta_tr_loss_path = f'{root_folder}wta_training_loss.pt'
wta_val_err_path = f'{root_folder}wta_validation_error.pt'

for epoch in range(epochs):
    epoch_loss_wta = 0

    # traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size
    traj_ids, v_traj_ids = [], []
    inds = torch.randperm(num_indiv)
    vinds = torch.randperm(num_val_indiv)
    for i in inds:
        traj_ids.append([inds[i], num_demos-inds[i]-1])

    for i in vinds:
        v_traj_ids.append([vinds[i], num_val-vinds[i]-1])

    for i in range(epoch_iter):
        optimizer_wta.zero_grad()

        obs_wta, tar_x_wta, tar_y_wta = get_batch(x, y, traj_ids[i], device)
        pred_wta, gate_wta = model_wta(obs_wta, tar_x_wta)
        loss_wta, wta_nll = model_wta.loss(pred_wta, gate_wta, tar_y_wta)
        loss_wta.backward()
        optimizer_wta.step()

        epoch_loss_wta += wta_nll.item()

    training_loss_wta.append(epoch_loss_wta)

    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            # v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
            val_loss_wta = 0

            for j in range(v_epoch_iter):
                o_wta, t_wta, tr_wta = get_validation_batch(vx, vy, v_traj_ids[j], device=device)

                p_wta, g_wta = model_wta(o_wta, t_wta)
                dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
                vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
                val_loss_wta += mse_loss(vp_means, tr_wta).item()

            validation_error_wta.append(val_loss_wta)
            print(f'Current val error: {val_loss_wta}')
            if val_loss_wta < min_val_loss_wta:
                min_val_loss_wta = val_loss_wta
                print(f'(WTA)New best: {min_val_loss_wta}')
                torch.save(model_wta_.state_dict(), f'{root_folder}saved_models/wta_on_synth.pt')

    avg_loss_wta += epoch_loss_wta

    if epoch % val_per_epoch == 0:
        print("Epoch: {}, WTA-Loss: {}".format(epoch, avg_loss_wta/val_per_epoch))
        avg_loss_wta = 0

torch.save(torch.Tensor(training_loss_wta), wta_tr_loss_path)
torch.save(torch.Tensor(validation_error_wta), wta_val_err_path)
