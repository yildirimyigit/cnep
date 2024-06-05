# %%
from models.cnep import CNEP
from models.cnmp import CNMP

from data.data_generators import *
import torch


torch.set_float32_matmul_precision('high')

def get_free_gpu():
    gpu_util = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)  # Switch GPU
        gpu_util.append((i, torch.cuda.utilization()))
    gpu_util.sort(key=lambda x: x[1])
    return gpu_util[0][0]

if torch.cuda.is_available():
    available_gpu = get_free_gpu()
    if available_gpu == 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{available_gpu}")
else:
    device = torch.device("cpu")

print("Device :", device)

# %%
batch_size = 32
n_max, m_max = 10, 10  # max number of points in context set and target set

t_steps = 200
num_demos = 128
num_classes = 4
num_indiv = num_demos//num_classes  # number of demos per class
noise_clip = 0.0
dx, dy = 1, 1

num_val = 32
num_val_indiv = num_val//num_classes

# %%
x = torch.linspace(0, 1, 200).repeat(num_indiv, 1)
y = torch.zeros(num_demos, t_steps, dy)

vx = torch.linspace(0, 1, 200).repeat(num_val_indiv, 1)
vy = torch.zeros(num_val, t_steps, dy)

for i in range(num_classes):
    start_ind = i * num_indiv
    coeff = (i + 1) / 2 * torch.pi
    phase_shifts = torch.rand(num_indiv) * 5e-2 * torch.pi

    # Expand dimensions for proper broadcasting (without the extra unsqueeze)
    expanded_x = x.expand(num_indiv, -1)  # Shape: (32, 200)
    expanded_x += 0.2
    expanded_phase_shifts = phase_shifts.unsqueeze(1).expand(-1, x.shape[1])  # Shape: (32, 200)

    # Now the element-wise operations work correctly
    y[start_ind: start_ind + num_indiv, :, :] = (
        torch.unsqueeze(generate_sin(expanded_x * coeff + expanded_phase_shifts), 2) + 1
    ) / 2.0


    noise = torch.unsqueeze(torch.clamp(torch.randn(vx.shape)*0.01, min=0) - noise_clip, -1)

    start_val_ind = i*num_val_indiv
    start_ind = i*num_indiv
    vy[start_val_ind:start_val_ind+num_val_indiv] = y[start_ind:start_ind+num_val_indiv].clone() + noise

x = torch.unsqueeze(x.repeat(num_classes, 1), 2)  # since dx = 1
vx = torch.unsqueeze(vx.repeat(num_classes, 1), 2)
print("X:", x.shape, "Y:", y.shape, "VX:", vx.shape, "VY:", vy.shape)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

colors = [sns.color_palette('tab10')[0], sns.color_palette('tab10')[1], sns.color_palette('tab10')[2], sns.color_palette('tab10')[3]]
sns.set_palette('tab10')

plt.figure(figsize=(6, 4))
for i in range(num_val):
    plt.plot(vx[i, :, 0].cpu(), vy[i, :, 0].cpu(), color=colors[i//num_val_indiv], alpha=0.5)
    # plt.plot(vx[i, :, 0].cpu(), vy[i, :, 0].cpu(), 'k', alpha=0.5)

# plt.legend(loc='lower left', fontsize=14)
plt.grid(True)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.title(f'Sine Waves', fontsize=16)

# %%
# import numpy as np
# import os

# save_path = 'data/synthetic/4_sines/'

# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# try:
#     os.makedirs(f'{save_path}4_sines_0')
#     os.makedirs(f'{save_path}4_sines_1')
#     os.makedirs(f'{save_path}4_sines_2')
#     os.makedirs(f'{save_path}4_sines_3')
# except:
#     pass


# t2 = []
# for i in range(num_demos):
#     traj = np.zeros((1, t_steps, 2))
#     traj[0, :, 0] = x[i, :, 0].cpu().numpy()
#     traj[0, :, 1] = y[i, :, 0].cpu().numpy()
        
#     np.save(f'{save_path}4_sines_{i//num_indiv}/{i%num_indiv}.npy', traj)

#     if i // num_indiv == 2:
#         t2.append(traj)

# for t in t2:
#     plt.plot(t[0, :, 0], t[0, :, 1], 'k', alpha=0.5)

# %%
obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
tar_x = torch.zeros((batch_size, m_max, dx), dtype=torch.float32, device=device)
tar_y = torch.zeros((batch_size, m_max, dy), dtype=torch.float32, device=device)
obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)

def prepare_masked_batch(t: list, traj_ids: list):
    obs.fill_(0)
    tar_x.fill_(0)
    tar_y.fill_(0)
    obs_mask.fill_(False)
    tar_mask.fill_(False)

    for i, traj_id in enumerate(traj_ids):
        traj = t[traj_id]

        n = torch.randint(1, n_max, (1,)).item()
        m = torch.randint(1, m_max, (1,)).item()

        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = permuted_ids[n:n+m]
        
        obs[i, :n, :dx] = (n_ids/t_steps).unsqueeze(1)
        obs[i, :n, dx:] = traj[n_ids]
        obs_mask[i, :n] = True
        
        tar_x[i, :m] = (m_ids/t_steps).unsqueeze(1)
        tar_y[i, :m] = traj[m_ids]
        tar_mask[i, :m] = True

val_obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
val_tar_x = torch.zeros((batch_size, t_steps, dx), dtype=torch.float32, device=device)
val_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)
val_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)

def prepare_masked_val_batch(t: list, traj_ids: list):
    val_obs.fill_(0)
    val_tar_x.fill_(0)
    val_tar_y.fill_(0)
    val_obs_mask.fill_(False)

    for i, traj_id in enumerate(traj_ids):
        traj = t[traj_id]

        n = torch.randint(1, n_max, (1,)).item()

        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = torch.arange(t_steps)
        
        val_obs[i, :n, :dx] = (n_ids/t_steps).unsqueeze(1)
        val_obs[i, :n, dx:] = traj[n_ids]
        val_obs_mask[i, :n] = True
        
        val_tar_x[i] = (m_ids/t_steps).unsqueeze(1)
        val_tar_y[i] = traj[m_ids]

# %%
model_ = CNEP(1, 1, n_max, m_max, [64,64], num_decoders=4, decoder_hidden_dims=[32, 32], batch_size=batch_size, scale_coefs=True, device=device)
optimizer = torch.optim.Adam(lr=1e-4, params=model_.parameters())

model0_ = CNEP(1, 1, n_max, m_max, [64,64], num_decoders=4, decoder_hidden_dims=[32, 32], batch_size=batch_size, scale_coefs=True, device=device)
model0_.batch_entropy_coef = 0.0
optimizer0 = torch.optim.Adam(lr=1e-4, params=model0_.parameters())

model1_ = CNEP(1, 1, n_max, m_max, [64,64], num_decoders=4, decoder_hidden_dims=[32, 32], batch_size=batch_size, scale_coefs=True, device=device)
model1_.ind_entropy_coef = 0.0
optimizer1 = torch.optim.Adam(lr=1e-4, params=model1_.parameters())

model2_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=4, decoder_hidden_dims=[32, 32], batch_size=batch_size, scale_coefs=True, device=device)
optimizer2 = torch.optim.Adam(lr=1e-4, params=model2_.parameters())
model2_.batch_entropy_coef = 0.0
model2_.ind_entropy_coef = 0.0

cnmp_ = CNMP(1, 1, n_max, m_max, [64,64], decoder_hidden_dims=[128,128], batch_size=batch_size, device=device)
optimizer3 = torch.optim.Adam(lr=1e-4, params=cnmp_.parameters())

def get_parameter_count(model):
    total_num = 0
    for param in model.parameters():
        total_num += param.shape.numel()
    return total_num

print("cnep:", get_parameter_count(model_))
print("cnmp:", get_parameter_count(cnmp_))


if torch.__version__ >= "2.0":
    model, model0, model1, model2, cnmp = torch.compile(model_), torch.compile(model0_), torch.compile(model1_), torch.compile(model2_), torch.compile(cnmp_)
else:
    model, model0, model1, model2, cnmp = model_, model0_, model1_, model2_, cnmp_

# %%
import time
import os

timestamp = int(time.time())
root_folder = f'outputs/ablation/sines_4/orig_0_1_2_cnmp/{str(timestamp)}/'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f'{root_folder}saved_models/'):
    os.makedirs(f'{root_folder}saved_models/')

if not os.path.exists(f'{root_folder}img/'):
    os.makedirs(f'{root_folder}img/')

torch.save(y, f'{root_folder}y.pt')


epochs = 1_000_000
epoch_iter = num_demos//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)
avg_loss, avg_loss0, avg_loss1, avg_loss2, avg_loss3 = 0, 0, 0, 0, 0

val_per_epoch = 2000
min_vl, min_vl0, min_vl1, min_vl2, min_vl3 = 1000000, 1000000, 1000000, 1000000, 1000000

mse_loss = torch.nn.MSELoss()

tl, tl0, tl1, tl2, tl3 = [], [], [], [], []
ve, ve0, ve1, ve2, ve3 = [], [], [], [], []

cnep_tl_path = f'{root_folder}cnep_training_loss.pt'
cnep_ve_path = f'{root_folder}cnep_validation_error.pt'

for epoch in range(epochs):
    epoch_loss, epoch_loss0, epoch_loss1, epoch_loss2, epoch_loss3 = 0, 0, 0, 0, 0

    traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

    for i in range(epoch_iter):
        prepare_masked_batch(y, traj_ids[i])

        optimizer.zero_grad()
        pred, gate = model(obs, tar_x, obs_mask)
        loss, nll = model.loss(pred, gate, tar_y, tar_mask)
        loss.backward()
        optimizer.step()


        optimizer0.zero_grad()
        pred0, gate0 = model0(obs, tar_x, obs_mask)
        loss0, nll0 = model0.loss(pred0, gate0, tar_y, tar_mask)
        loss0.backward()
        optimizer0.step()


        optimizer1.zero_grad()
        pred1, gate1 = model1(obs, tar_x, obs_mask)
        loss1, nll1 = model1.loss(pred1, gate1, tar_y, tar_mask)
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        pred2, gate2 = model2(obs, tar_x, obs_mask)
        loss2, nll2 = model2.loss(pred2, gate2, tar_y, tar_mask)
        loss2.backward()
        optimizer2.step()

        optimizer3.zero_grad()
        pred3 = cnmp(obs, tar_x, obs_mask)
        nll3 = cnmp.loss(pred3, tar_y, tar_mask)
        nll3.backward()
        optimizer3.step()

        epoch_loss += nll.item()
        epoch_loss0 += nll0.item()
        epoch_loss1 += nll1.item()
        epoch_loss2 += nll2.item()
        epoch_loss3 += nll3.item()

    epoch_loss = epoch_loss/num_demos
    epoch_loss0 = epoch_loss0/num_demos
    epoch_loss1 = epoch_loss1/num_demos
    epoch_loss2 = epoch_loss2/num_demos
    epoch_loss3 = epoch_loss3/num_demos

    
    tl.append(epoch_loss)
    tl0.append(epoch_loss0)
    tl1.append(epoch_loss1)
    tl2.append(epoch_loss2)
    tl3.append(epoch_loss3)

    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
            val_loss, val_loss0, val_loss1, val_loss2, val_loss3 = 0, 0, 0, 0, 0

            for j in range(v_epoch_iter):
                prepare_masked_val_batch(vy, v_traj_ids[j])

                p, g = model.val(val_obs, val_tar_x, val_obs_mask)
                dec_id = torch.argmax(g.squeeze(1), dim=-1)
                vp_means = p[dec_id, torch.arange(batch_size), :, :dy]
                val_loss += mse_loss(vp_means, val_tar_y).item()

                p, g = model0.val(val_obs, val_tar_x, val_obs_mask)
                dec_id = torch.argmax(g.squeeze(1), dim=-1)
                vp_means = p[dec_id, torch.arange(batch_size), :, :dy]
                val_loss0 += mse_loss(vp_means, val_tar_y).item()

                p, g = model1.val(val_obs, val_tar_x, val_obs_mask)
                dec_id = torch.argmax(g.squeeze(1), dim=-1)
                vp_means = p[dec_id, torch.arange(batch_size), :, :dy]
                val_loss1 += mse_loss(vp_means, val_tar_y).item()

                p, g = model2.val(val_obs, val_tar_x, val_obs_mask)
                dec_id = torch.argmax(g.squeeze(1), dim=-1)
                vp_means = p[dec_id, torch.arange(batch_size), :, :dy]
                val_loss2 += mse_loss(vp_means, val_tar_y).item()

                p = cnmp.val(val_obs, val_tar_x, val_obs_mask)
                vp_means = p[:, :, :dy]
                val_loss3 += mse_loss(vp_means, val_tar_y).item()


            val_loss /= num_val
            val_loss0 /= num_val
            val_loss1 /= num_val
            val_loss2 /= num_val
            val_loss3 /= num_val

            ve.append(val_loss)
            ve0.append(val_loss0)
            ve1.append(val_loss1)
            ve2.append(val_loss2)
            ve3.append(val_loss3)

            if val_loss < min_vl:
                min_vl = val_loss
                print(f'New best CNEP: {min_vl}')
                torch.save(model_.state_dict(), f'{root_folder}saved_models/org.pt')

            if val_loss0 < min_vl0:
                min_vl0 = val_loss0
                print(f'New best CNEP-A0: {min_vl0}')
                torch.save(model0_.state_dict(), f'{root_folder}saved_models/abl0.pt')

            if val_loss1 < min_vl1:
                min_vl1 = val_loss1
                print(f'New best CNEP-A1: {min_vl1}')
                torch.save(model1_.state_dict(), f'{root_folder}saved_models/abl1.pt')

            if val_loss2 < min_vl2:
                min_vl2 = val_loss2
                print(f'New best CNEP-A2: {min_vl2}')
                torch.save(model2_.state_dict(), f'{root_folder}saved_models/abl2.pt')

            if val_loss3 < min_vl3:
                min_vl3 = val_loss3
                print(f'New best CNMP: {min_vl3}')
                torch.save(cnmp_.state_dict(), f'{root_folder}saved_models/cnmp.pt')
            
            print(f'Bests: {min_vl}, {min_vl0}, {min_vl1}, {min_vl2}, {min_vl3}')

    avg_loss += epoch_loss
    avg_loss0 += epoch_loss0
    avg_loss1 += epoch_loss1
    avg_loss2 += epoch_loss2
    avg_loss3 += epoch_loss3

    if epoch % val_per_epoch == 0:
        print("Epoch: {}, CNEP: {}, CNEP-A0: {}, CNEP-A1: {}, CNEP-A2: {}, CNMP: {}".format(epoch, avg_loss/val_per_epoch, avg_loss0/val_per_epoch, avg_loss1/val_per_epoch, avg_loss2/val_per_epoch, avg_loss3/val_per_epoch))
        avg_loss, avg_loss0, avg_loss1, avg_loss2, avg_loss3 = 0, 0, 0, 0, 0

torch.save(torch.Tensor(tl), cnep_tl_path)
torch.save(torch.Tensor(ve), cnep_ve_path)
torch.save(torch.Tensor(tl0), cnep_tl_path+'_a0')
torch.save(torch.Tensor(ve0), cnep_ve_path+'_a0')
torch.save(torch.Tensor(tl1), cnep_tl_path+'_a1')
torch.save(torch.Tensor(ve1), cnep_ve_path+'_a1')
torch.save(torch.Tensor(tl2), cnep_tl_path+'_a2')
torch.save(torch.Tensor(ve2), cnep_ve_path+'_a2')
torch.save(torch.Tensor(tl3), cnep_tl_path+'_cnmp')
torch.save(torch.Tensor(ve3), cnep_ve_path+'_cnmp')

# %%



