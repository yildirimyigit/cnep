# %%
from models.cnep import CNEP

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

# # %%
# batch_size = 32
# n_max, m_max = 10, 10  # max number of points in context set and target set

# t_steps = 200
# num_demos = 128
# num_classes = 4
# num_indiv = num_demos//num_classes  # number of demos per class
# noise_clip = 0.0
# dx, dy = 1, 1

# num_val = 32
# num_val_indiv = num_val//num_classes

# # %%
# x = torch.linspace(0, 1, 200).repeat(num_indiv, 1)
# y = torch.zeros(num_demos, t_steps, dy)

# vx = torch.linspace(0, 1, 200).repeat(num_val_indiv, 1)
# vy = torch.zeros(num_val, t_steps, dy)

# for i in range(num_classes):
#     start_ind = i * num_indiv
#     coeff = (i + 1) / 2 * torch.pi
#     phase_shifts = torch.rand(num_indiv) * 5e-2 * torch.pi

#     # Expand dimensions for proper broadcasting (without the extra unsqueeze)
#     expanded_x = x.expand(num_indiv, -1)  # Shape: (32, 200)
#     expanded_x += 0.2
#     expanded_phase_shifts = phase_shifts.unsqueeze(1).expand(-1, x.shape[1])  # Shape: (32, 200)

#     # Now the element-wise operations work correctly
#     y[start_ind: start_ind + num_indiv, :, :] = (
#         torch.unsqueeze(generate_sin(expanded_x * coeff + expanded_phase_shifts), 2) + 1
#     ) / 2.0


#     noise = torch.unsqueeze(torch.clamp(torch.randn(vx.shape)*0.01, min=0) - noise_clip, -1)

#     start_val_ind = i*num_val_indiv
#     start_ind = i*num_indiv
#     vy[start_val_ind:start_val_ind+num_val_indiv] = y[start_ind:start_ind+num_val_indiv].clone() + noise

# x = torch.unsqueeze(x.repeat(num_classes, 1), 2)  # since dx = 1
# vx = torch.unsqueeze(vx.repeat(num_classes, 1), 2)
# print("X:", x.shape, "Y:", y.shape, "VX:", vx.shape, "VY:", vy.shape)

# # %%
# obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
# tar_x = torch.zeros((batch_size, m_max, dx), dtype=torch.float32, device=device)
# tar_y = torch.zeros((batch_size, m_max, dy), dtype=torch.float32, device=device)
# obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
# tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)

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

# val_obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
# val_tar_x = torch.zeros((batch_size, t_steps, dx), dtype=torch.float32, device=device)
# val_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)
# val_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)

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

# # %%
# model2_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=2, decoder_hidden_dims=[130, 130], batch_size=batch_size, scale_coefs=True, device=device)
# optimizer2 = torch.optim.Adam(lr=3e-4, params=model2_.parameters())

# model4_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=4, decoder_hidden_dims=[64, 64], batch_size=batch_size, scale_coefs=True, device=device)
# optimizer4 = torch.optim.Adam(lr=3e-4, params=model4_.parameters())

# model8_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=8, decoder_hidden_dims=[32, 32], batch_size=batch_size, scale_coefs=True, device=device)
# optimizer8 = torch.optim.Adam(lr=3e-4, params=model8_.parameters())

# def get_parameter_count(model):
#     total_num = 0
#     for param in model.parameters():
#         total_num += param.shape.numel()
#     return total_num

# print("cnep2:", get_parameter_count(model2_))
# print("cnep4:", get_parameter_count(model4_))
# print("cnep8:", get_parameter_count(model8_))

# if torch.__version__ >= "2.0":
#     model2, model4, model8 = torch.compile(model2_), torch.compile(model4_), torch.compile(model8_)
# else:
#     model2, model4, model8 = model2_, model4_, model8_


# import time
# import os

# timestamp = int(time.time())
# root_folder = f'outputs/ablation/sines_4/2_4_8/{str(timestamp)}/'

# if not os.path.exists(root_folder):
#     os.makedirs(root_folder)

# if not os.path.exists(f'{root_folder}saved_models/'):
#     os.makedirs(f'{root_folder}saved_models/')

# if not os.path.exists(f'{root_folder}img/'):
#     os.makedirs(f'{root_folder}img/')

# torch.save(y, f'{root_folder}y.pt')


# epochs = 3_000_000
# epoch_iter = num_demos//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
# v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)
# avg_loss2, avg_loss4, avg_loss8 = 0, 0, 0

# val_per_epoch = 5000
# min_vl2, min_vl4, min_vl8 = 1000000, 1000000, 1000000

# mse_loss = torch.nn.MSELoss()

# tl2, tl4, tl8 = [], [], []
# ve2, ve4, ve8 = [], [], []

# cnep_tl_path = f'{root_folder}cnep_training_loss.pt'
# cnep_ve_path = f'{root_folder}cnep_validation_error.pt'

# for epoch in range(epochs):
#     epoch_loss2, epoch_loss4, epoch_loss8 = 0, 0, 0

#     traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

#     for i in range(epoch_iter):
#         prepare_masked_batch(y, traj_ids[i])

#         optimizer2.zero_grad()
#         pred2, gate2 = model2(obs, tar_x, obs_mask)
#         loss2, nll2 = model2.loss(pred2, gate2, tar_y, tar_mask)
#         loss2.backward()
#         optimizer2.step()


#         optimizer4.zero_grad()
#         pred4, gate4 = model4(obs, tar_x, obs_mask)
#         loss4, nll4 = model4.loss(pred4, gate4, tar_y, tar_mask)
#         loss4.backward()
#         optimizer4.step()


#         optimizer8.zero_grad()
#         pred8, gate8 = model8(obs, tar_x, obs_mask)
#         loss8, nll8 = model8.loss(pred8, gate8, tar_y, tar_mask)
#         loss8.backward()
#         optimizer8.step()

#         epoch_loss2 += nll2.item()
#         epoch_loss4 += nll4.item()
#         epoch_loss8 += nll8.item()

#     epoch_loss2 = epoch_loss2/epoch_iter
#     epoch_loss4 = epoch_loss4/epoch_iter
#     epoch_loss8 = epoch_loss8/epoch_iter

#     tl2.append(epoch_loss2)
#     tl4.append(epoch_loss4)
#     tl8.append(epoch_loss8)

#     if epoch % val_per_epoch == 0:
#         with torch.no_grad():
#             v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
#             val_loss2, val_loss4, val_loss8 = 0, 0, 0

#             for j in range(v_epoch_iter):
#                 prepare_masked_val_batch(vy, v_traj_ids[j])

#                 p_wta, g_wta = model2.val(val_obs, val_tar_x, val_obs_mask)
#                 dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
#                 vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
#                 val_loss2 += mse_loss(vp_means, val_tar_y).item()

#                 p_wta, g_wta = model4.val(val_obs, val_tar_x, val_obs_mask)
#                 dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
#                 vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
#                 val_loss4 += mse_loss(vp_means, val_tar_y).item()

#                 p_wta, g_wta = model8.val(val_obs, val_tar_x, val_obs_mask)
#                 dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
#                 vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
#                 val_loss8 += mse_loss(vp_means, val_tar_y).item()

#             ve2.append(val_loss2)
#             ve4.append(val_loss4)
#             ve8.append(val_loss8)

#             if val_loss2 < min_vl2:
#                 min_vl2 = val_loss2
#                 print(f'New best 2: {min_vl2}')
#                 torch.save(model2_.state_dict(), f'{root_folder}saved_models/cnep2.pt')

#             if val_loss4 < min_vl4:
#                 min_vl4 = val_loss4
#                 print(f'New best 4: {min_vl4}')
#                 torch.save(model4_.state_dict(), f'{root_folder}saved_models/cnep4.pt')

#             if val_loss8 < min_vl8:
#                 min_vl8 = val_loss8
#                 print(f'New best 8: {min_vl8}')
#                 torch.save(model8_.state_dict(), f'{root_folder}saved_models/cnep8.pt')
            
#             print(f'Bests: {min_vl2}, {min_vl4}, {min_vl8}')

#     avg_loss2 += epoch_loss2
#     avg_loss4 += epoch_loss4
#     avg_loss8 += epoch_loss8

#     if epoch % val_per_epoch == 0:
#         print("Epoch: {}, CNEP Losses: {}, {}, {}".format(epoch, avg_loss2/val_per_epoch, avg_loss4/val_per_epoch, avg_loss8/val_per_epoch))
#         avg_loss2, avg_loss4, avg_loss8 = 0, 0, 0

# torch.save(torch.Tensor(tl2), cnep_tl_path+'_2')
# torch.save(torch.Tensor(ve2), cnep_ve_path+'_2')
# torch.save(torch.Tensor(tl4), cnep_tl_path+'_4')
# torch.save(torch.Tensor(ve4), cnep_ve_path+'_4')
# torch.save(torch.Tensor(tl8), cnep_tl_path+'_8')
# torch.save(torch.Tensor(ve8), cnep_ve_path+'_8')


# ################# BIMODAL #####################

# batch_size = 32
# n_max, m_max = 10, 10  # max number of points in context set and target set

# t_steps = 200
# num_demos = 128
# num_classes = 2
# num_indiv = num_demos//num_classes  # number of demos per class
# noise_clip = 0.0
# dx, dy = 1, 1

# num_val = 32
# num_val_indiv = num_val//num_classes

# x = torch.linspace(0, 1, 200).repeat(num_demos, 1).unsqueeze(-1)
# vx = torch.linspace(0, 1, 200).repeat(num_val, 1).unsqueeze(-1)
# y = torch.zeros((num_demos, x.shape[1], 1))
# vy = torch.zeros((num_val, vx.shape[1], 1))

# frequencies = [1, 2.5, 4, 5.5]  # Example frequencies
# amplitudes = [1.5, 1, 0.8, 0.6]  # Example amplitudes
# phases = [0, torch.pi / 2, torch.pi, 3 * torch.pi / 2]  # Example phases


# for i in range(num_demos):
#     y[i, :, 0] = amplitudes[i%2] * torch.sin(2 * torch.pi * frequencies[i%2] * x[i, :, 0] + phases[i%2]) + torch.randn(1) * 0.05
# for i in range(num_val):
#     vy[i, :, 0] = amplitudes[i%2] * torch.sin(2 * torch.pi * frequencies[i%2] * vx[i, :, 0] + phases[i%2]) + torch.randn(1) * 0.05

# print("X:", x.shape, "Y:", y.shape, "VX:", vx.shape, "VY:", vy.shape)
# x, y, vx, vy = x.to(device), y.to(device), vx.to(device), vy.to(device)


# obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
# tar_x = torch.zeros((batch_size, m_max, dx), dtype=torch.float32, device=device)
# tar_y = torch.zeros((batch_size, m_max, dy), dtype=torch.float32, device=device)
# obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
# tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)

# val_obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
# val_tar_x = torch.zeros((batch_size, t_steps, dx), dtype=torch.float32, device=device)
# val_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)
# val_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)


# model2_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=2, decoder_hidden_dims=[130, 130], batch_size=batch_size, scale_coefs=True, device=device)
# optimizer2 = torch.optim.Adam(lr=3e-4, params=model2_.parameters())

# model4_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=4, decoder_hidden_dims=[64, 64], batch_size=batch_size, scale_coefs=True, device=device)
# optimizer4 = torch.optim.Adam(lr=3e-4, params=model4_.parameters())

# model8_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=8, decoder_hidden_dims=[32, 32], batch_size=batch_size, scale_coefs=True, device=device)
# optimizer8 = torch.optim.Adam(lr=3e-4, params=model8_.parameters())

# def get_parameter_count(model):
#     total_num = 0
#     for param in model.parameters():
#         total_num += param.shape.numel()
#     return total_num

# print("cnep2:", get_parameter_count(model2_))
# print("cnep4:", get_parameter_count(model4_))
# print("cnep8:", get_parameter_count(model8_))

# if torch.__version__ >= "2.0":
#     model2, model4, model8 = torch.compile(model2_), torch.compile(model4_), torch.compile(model8_)
# else:
#     model2, model4, model8 = model2_, model4_, model8_


# import time
# import os

# timestamp = int(time.time())
# root_folder = f'outputs/ablation/sines_2/2_4_8/{str(timestamp)}/'

# if not os.path.exists(root_folder):
#     os.makedirs(root_folder)

# if not os.path.exists(f'{root_folder}saved_models/'):
#     os.makedirs(f'{root_folder}saved_models/')

# if not os.path.exists(f'{root_folder}img/'):
#     os.makedirs(f'{root_folder}img/')

# torch.save(y, f'{root_folder}y.pt')


# epochs = 2_000_000
# epoch_iter = num_demos//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
# v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)
# avg_loss2, avg_loss4, avg_loss8 = 0, 0, 0

# val_per_epoch = 5000
# min_vl2, min_vl4, min_vl8 = 1000000, 1000000, 1000000

# mse_loss = torch.nn.MSELoss()

# tl2, tl4, tl8 = [], [], []
# ve2, ve4, ve8 = [], [], []

# cnep_tl_path = f'{root_folder}cnep_training_loss.pt'
# cnep_ve_path = f'{root_folder}cnep_validation_error.pt'

# for epoch in range(epochs):
#     epoch_loss2, epoch_loss4, epoch_loss8 = 0, 0, 0

#     traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

#     for i in range(epoch_iter):
#         prepare_masked_batch(y, traj_ids[i])

#         optimizer2.zero_grad()
#         pred2, gate2 = model2(obs, tar_x, obs_mask)
#         loss2, nll2 = model2.loss(pred2, gate2, tar_y, tar_mask)
#         loss2.backward()
#         optimizer2.step()


#         optimizer4.zero_grad()
#         pred4, gate4 = model4(obs, tar_x, obs_mask)
#         loss4, nll4 = model4.loss(pred4, gate4, tar_y, tar_mask)
#         loss4.backward()
#         optimizer4.step()


#         optimizer8.zero_grad()
#         pred8, gate8 = model8(obs, tar_x, obs_mask)
#         loss8, nll8 = model8.loss(pred8, gate8, tar_y, tar_mask)
#         loss8.backward()
#         optimizer8.step()

#         epoch_loss2 += nll2.item()
#         epoch_loss4 += nll4.item()
#         epoch_loss8 += nll8.item()

#     epoch_loss2 = epoch_loss2/epoch_iter
#     epoch_loss4 = epoch_loss4/epoch_iter
#     epoch_loss8 = epoch_loss8/epoch_iter

#     tl2.append(epoch_loss2)
#     tl4.append(epoch_loss4)
#     tl8.append(epoch_loss8)

#     if epoch % val_per_epoch == 0:
#         with torch.no_grad():
#             v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
#             val_loss2, val_loss4, val_loss8 = 0, 0, 0

#             for j in range(v_epoch_iter):
#                 prepare_masked_val_batch(vy, v_traj_ids[j])

#                 p_wta, g_wta = model2.val(val_obs, val_tar_x, val_obs_mask)
#                 dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
#                 vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
#                 val_loss2 += mse_loss(vp_means, val_tar_y).item()

#                 p_wta, g_wta = model4.val(val_obs, val_tar_x, val_obs_mask)
#                 dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
#                 vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
#                 val_loss4 += mse_loss(vp_means, val_tar_y).item()

#                 p_wta, g_wta = model8.val(val_obs, val_tar_x, val_obs_mask)
#                 dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
#                 vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
#                 val_loss8 += mse_loss(vp_means, val_tar_y).item()

#             ve2.append(val_loss2)
#             ve4.append(val_loss4)
#             ve8.append(val_loss8)

#             if val_loss2 < min_vl2:
#                 min_vl2 = val_loss2
#                 print(f'New best 2: {min_vl2}')
#                 torch.save(model2_.state_dict(), f'{root_folder}saved_models/cnep2.pt')

#             if val_loss4 < min_vl4:
#                 min_vl4 = val_loss4
#                 print(f'New best 4: {min_vl4}')
#                 torch.save(model4_.state_dict(), f'{root_folder}saved_models/cnep4.pt')

#             if val_loss8 < min_vl8:
#                 min_vl8 = val_loss8
#                 print(f'New best 8: {min_vl8}')
#                 torch.save(model8_.state_dict(), f'{root_folder}saved_models/cnep8.pt')
            
#             print(f'Bests: {min_vl2}, {min_vl4}, {min_vl8}')

#     avg_loss2 += epoch_loss2
#     avg_loss4 += epoch_loss4
#     avg_loss8 += epoch_loss8

#     if epoch % val_per_epoch == 0:
#         print("Epoch: {}, CNEP Losses: {}, {}, {}".format(epoch, avg_loss2/val_per_epoch, avg_loss4/val_per_epoch, avg_loss8/val_per_epoch))
#         avg_loss2, avg_loss4, avg_loss8 = 0, 0, 0

# torch.save(torch.Tensor(tl2), cnep_tl_path+'_2')
# torch.save(torch.Tensor(ve2), cnep_ve_path+'_2')
# torch.save(torch.Tensor(tl4), cnep_tl_path+'_4')
# torch.save(torch.Tensor(ve4), cnep_ve_path+'_4')
# torch.save(torch.Tensor(tl8), cnep_tl_path+'_8')
# torch.save(torch.Tensor(ve8), cnep_ve_path+'_8')

################# UNIMODAL #####################

batch_size = 6
n_max, m_max = 6, 6  # max number of points in context set and target set

t_steps = 100
num_demos = 24
num_classes = 1
num_indiv = num_demos//num_classes  # number of demos per class
noise_clip = 0.0
dx, dy = 1, 1

num_val = 6
num_val_indiv = num_val//num_classes

import numpy as np


x = np.arange(0,1.00,0.01)
nrTraj=30
sigmaNoise=0.02

A = np.array([.2, .2, .01, -.05])
X = np.vstack((np.sin(5*x), x**2, x, np.ones((1,len(x)))))
Y = np.zeros((nrTraj, len(x)))

for traj in range(nrTraj):
    sample = -np.dot(A + sigmaNoise * np.random.randn(1,4), X)[0]
    Y[traj] = sample

x = torch.linspace(0, 1, t_steps).repeat(num_demos, 1).unsqueeze(-1)
vx = torch.linspace(0, 1, t_steps).repeat(num_val, 1).unsqueeze(-1)
all_y = torch.from_numpy(Y).float().unsqueeze(-1)
y = all_y[:num_demos]
vy = all_y[-num_val:]

print("X:", x.shape, "Y:", y.shape, "VX:", vx.shape, "VY:", vy.shape)
x, y, vx, vy = x.to(device), y.to(device), vx.to(device), vy.to(device)

obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
tar_x = torch.zeros((batch_size, m_max, dx), dtype=torch.float32, device=device)
tar_y = torch.zeros((batch_size, m_max, dy), dtype=torch.float32, device=device)
obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)

val_obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
val_tar_x = torch.zeros((batch_size, t_steps, dx), dtype=torch.float32, device=device)
val_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)
val_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)


model2_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=2, decoder_hidden_dims=[130, 130], batch_size=batch_size, scale_coefs=True, device=device)
optimizer2 = torch.optim.Adam(lr=3e-4, params=model2_.parameters())

model4_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=4, decoder_hidden_dims=[64, 64], batch_size=batch_size, scale_coefs=True, device=device)
optimizer4 = torch.optim.Adam(lr=3e-4, params=model4_.parameters())

model8_ = CNEP(1, 1, n_max, n_max, [64,64], num_decoders=8, decoder_hidden_dims=[32, 32], batch_size=batch_size, scale_coefs=True, device=device)
optimizer8 = torch.optim.Adam(lr=3e-4, params=model8_.parameters())

def get_parameter_count(model):
    total_num = 0
    for param in model.parameters():
        total_num += param.shape.numel()
    return total_num

print("cnep2:", get_parameter_count(model2_))
print("cnep4:", get_parameter_count(model4_))
print("cnep8:", get_parameter_count(model8_))

if torch.__version__ >= "2.0":
    model2, model4, model8 = torch.compile(model2_), torch.compile(model4_), torch.compile(model8_)
else:
    model2, model4, model8 = model2_, model4_, model8_


import time
import os

timestamp = int(time.time())
root_folder = f'outputs/ablation/unimodal/2_4_8/{str(timestamp)}/'

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
avg_loss2, avg_loss4, avg_loss8 = 0, 0, 0

val_per_epoch = 5000
min_vl2, min_vl4, min_vl8 = 1000000, 1000000, 1000000

mse_loss = torch.nn.MSELoss()

tl2, tl4, tl8 = [], [], []
ve2, ve4, ve8 = [], [], []

cnep_tl_path = f'{root_folder}cnep_training_loss.pt'
cnep_ve_path = f'{root_folder}cnep_validation_error.pt'

for epoch in range(epochs):
    epoch_loss2, epoch_loss4, epoch_loss8 = 0, 0, 0

    traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

    for i in range(epoch_iter):
        prepare_masked_batch(y, traj_ids[i])

        optimizer2.zero_grad()
        pred2, gate2 = model2(obs, tar_x, obs_mask)
        loss2, nll2 = model2.loss(pred2, gate2, tar_y, tar_mask)
        loss2.backward()
        optimizer2.step()


        optimizer4.zero_grad()
        pred4, gate4 = model4(obs, tar_x, obs_mask)
        loss4, nll4 = model4.loss(pred4, gate4, tar_y, tar_mask)
        loss4.backward()
        optimizer4.step()


        optimizer8.zero_grad()
        pred8, gate8 = model8(obs, tar_x, obs_mask)
        loss8, nll8 = model8.loss(pred8, gate8, tar_y, tar_mask)
        loss8.backward()
        optimizer8.step()

        epoch_loss2 += nll2.item()
        epoch_loss4 += nll4.item()
        epoch_loss8 += nll8.item()

    epoch_loss2 = epoch_loss2/epoch_iter
    epoch_loss4 = epoch_loss4/epoch_iter
    epoch_loss8 = epoch_loss8/epoch_iter

    tl2.append(epoch_loss2)
    tl4.append(epoch_loss4)
    tl8.append(epoch_loss8)

    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
            val_loss2, val_loss4, val_loss8 = 0, 0, 0

            for j in range(v_epoch_iter):
                prepare_masked_val_batch(vy, v_traj_ids[j])

                p_wta, g_wta = model2.val(val_obs, val_tar_x, val_obs_mask)
                dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
                vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
                val_loss2 += mse_loss(vp_means, val_tar_y).item()

                p_wta, g_wta = model4.val(val_obs, val_tar_x, val_obs_mask)
                dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
                vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
                val_loss4 += mse_loss(vp_means, val_tar_y).item()

                p_wta, g_wta = model8.val(val_obs, val_tar_x, val_obs_mask)
                dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
                vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
                val_loss8 += mse_loss(vp_means, val_tar_y).item()

            ve2.append(val_loss2)
            ve4.append(val_loss4)
            ve8.append(val_loss8)

            if val_loss2 < min_vl2:
                min_vl2 = val_loss2
                print(f'New best 2: {min_vl2}')
                torch.save(model2_.state_dict(), f'{root_folder}saved_models/cnep2.pt')

            if val_loss4 < min_vl4:
                min_vl4 = val_loss4
                print(f'New best 4: {min_vl4}')
                torch.save(model4_.state_dict(), f'{root_folder}saved_models/cnep4.pt')

            if val_loss8 < min_vl8:
                min_vl8 = val_loss8
                print(f'New best 8: {min_vl8}')
                torch.save(model8_.state_dict(), f'{root_folder}saved_models/cnep8.pt')
            
            print(f'Bests: {min_vl2}, {min_vl4}, {min_vl8}')

    avg_loss2 += epoch_loss2
    avg_loss4 += epoch_loss4
    avg_loss8 += epoch_loss8

    if epoch % val_per_epoch == 0:
        print("Epoch: {}, CNEP Losses: {}, {}, {}".format(epoch, avg_loss2/val_per_epoch, avg_loss4/val_per_epoch, avg_loss8/val_per_epoch))
        avg_loss2, avg_loss4, avg_loss8 = 0, 0, 0

torch.save(torch.Tensor(tl2), cnep_tl_path+'_2')
torch.save(torch.Tensor(ve2), cnep_ve_path+'_2')
torch.save(torch.Tensor(tl4), cnep_tl_path+'_4')
torch.save(torch.Tensor(ve4), cnep_ve_path+'_4')
torch.save(torch.Tensor(tl8), cnep_tl_path+'_8')
torch.save(torch.Tensor(ve8), cnep_ve_path+'_8')