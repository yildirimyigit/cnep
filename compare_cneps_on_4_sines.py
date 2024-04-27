# %%
from models.cnp import CNP
from models.wta_cnp import WTA_CNP

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
batch_size = 4
n_max_obs, n_max_tar = 10, 10

t_steps = 200
num_demos = 32
num_classes = 4
num_indiv = num_demos//num_classes  # number of demos per class
noise_clip = 0.0
dx, dy = 1, 1

num_val = 8
num_val_indiv = num_val//num_classes

# %%
import seaborn as sns

colors = [sns.color_palette('tab10')[0], sns.color_palette('tab10')[1], sns.color_palette('tab10')[2], sns.color_palette('tab10')[3]]
sns.set_palette('tab10')

x = torch.linspace(0, 1, 200).repeat(num_indiv, 1)
y = torch.zeros(num_demos, t_steps, dy)

vx = torch.linspace(0, 1, 200).repeat(num_val_indiv, 1)
vy = torch.zeros(num_val, t_steps, dy)

for i in range(num_classes):
    start_ind = i*num_indiv
    coeff = (i+1)/2*torch.pi
    y[start_ind:start_ind+num_indiv] = (torch.unsqueeze(generate_sin(x*coeff), 2) +1)/2.0

    noise = torch.unsqueeze(torch.clamp(torch.randn(vx.shape)*1e-4**0.5, min=0) - noise_clip, -1)

    start_ind = i*num_val_indiv
    vy[start_ind:start_ind+num_val_indiv] = y[start_ind:start_ind+num_val_indiv].clone() + noise

x = torch.unsqueeze(x.repeat(num_classes, 1), 2)  # since dx = 1
vx = torch.unsqueeze(vx.repeat(num_classes, 1), 2)
print("X:", x.shape, "Y:", y.shape, "VX:", vx.shape, "VY:", vy.shape)

# from matplotlib import pyplot as plt

# plt.figure(figsize=(8, 6))
# for i in range(num_demos):
#     plt.plot(x[i, :, 0].cpu(), y[i, :, 0].cpu(), label=f'Sine Wave {i+1}', linewidth=2.0, color=colors[i])
#     # plt.plot(vx[i, :, 0].cpu(), vy[i, :, 0].cpu(), 'k', alpha=0.5)

# plt.legend(loc='lower left', fontsize=14)
# plt.grid(True)
# plt.xlabel('Time (s)', fontsize=14)
# plt.ylabel('Amplitude', fontsize=14)
# plt.title(f'Sine Wave of 3 Different Frequencies', fontsize=16)
# plt.savefig(f'/home/yigit/papers/yildirim_23_ral/fig/3.png', bbox_inches='tight')

# x0, y0 = x.to(device_wta), y.to(device_wta)
# x1, y1 = x.to(device_cnp), y.to(device_cnp)

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
model2 = WTA_CNP(1, 1, n_max_obs, n_max_tar, [128,128,128], num_decoders=2, decoder_hidden_dims=[306,306,306], batch_size=batch_size, scale_coefs=True).to(device)
optimizer2 = torch.optim.Adam(lr=1e-4, params=model2.parameters())

model4 = WTA_CNP(1, 1, n_max_obs, n_max_tar, [128,128,128], num_decoders=4, decoder_hidden_dims=[201,201,201], batch_size=batch_size, scale_coefs=True).to(device)
optimizer4 = torch.optim.Adam(lr=1e-4, params=model4.parameters())

model8 = WTA_CNP(1, 1, n_max_obs, n_max_tar, [128,128,128], num_decoders=8, decoder_hidden_dims=[128,128,128], batch_size=batch_size, scale_coefs=True).to(device)
optimizer8 = torch.optim.Adam(lr=1e-4, params=model8.parameters())

def get_parameter_count(model):
    total_num = 0
    for param in model.parameters():
        total_num += param.shape.numel()
    return total_num

print("cnep2:", get_parameter_count(model2))
print("cnep4:", get_parameter_count(model4))
print("cnep8:", get_parameter_count(model8))

# %%
import time
import os

timestamp = int(time.time())
root_folder = f'outputs/ablation/sines_4/2_4_8/{str(timestamp)}/'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f'{root_folder}saved_models/'):
    os.makedirs(f'{root_folder}saved_models/')

if not os.path.exists(f'{root_folder}img/'):
    os.makedirs(f'{root_folder}img/')

torch.save(y, f'{root_folder}y.pt')


epochs = 5_000_000
epoch_iter = num_demos//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)
avg_loss2, avg_loss4, avg_loss8 = 0, 0, 0

val_per_epoch = 1000
min_vl2, min_vl4, min_vl8 = 1000000, 1000000, 1000000

mse_loss = torch.nn.MSELoss()

tl2, tl4, tl8 = [], [], []
ve2, ve4, ve8 = [], [], []

# wta_tr_loss_path = f'{root_folder}wta_training_loss.pt'
# wta_val_err_path = f'{root_folder}wta_validation_error.pt'

for epoch in range(epochs):
    epoch_loss2, epoch_loss4, epoch_loss8 = 0, 0, 0

    traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

    for i in range(epoch_iter):
        optimizer2.zero_grad()

        obs, tar_x, tar_y = get_batch(x, y, traj_ids[i], device)

        pred2, gate2 = model2(obs, tar_x)
        loss2, nll2 = model2.loss(pred2, gate2, tar_y)
        loss2.backward()
        optimizer2.step()

        pred4, gate4 = model4(obs, tar_x)
        loss4, nll4 = model4.loss(pred4, gate4, tar_y)
        loss4.backward()
        optimizer4.step()

        pred8, gate8 = model8(obs, tar_x)
        loss8, nll8 = model8.loss(pred8, gate8, tar_y)
        loss8.backward()
        optimizer8.step()

        epoch_loss2 += nll2.item()
        epoch_loss4 += nll4.item()
        epoch_loss8 += nll8.item()

    tl2.append(epoch_loss2)
    tl4.append(epoch_loss4)
    tl8.append(epoch_loss8)

    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
            val_loss2, val_loss4, val_loss8 = 0, 0, 0

            for j in range(v_epoch_iter):
                o_wta, t_wta, tr_wta = get_validation_batch(vx, vy, v_traj_ids[j], device=device)

                p_wta, g_wta = model2(o_wta, t_wta)
                dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
                vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
                val_loss2 += mse_loss(vp_means, tr_wta).item()

                p_wta, g_wta = model4(o_wta, t_wta)
                dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
                vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
                val_loss4 += mse_loss(vp_means, tr_wta).item()

                p_wta, g_wta = model8(o_wta, t_wta)
                dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
                vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
                val_loss8 += mse_loss(vp_means, tr_wta).item()

            ve2.append(val_loss2)
            ve4.append(val_loss4)
            ve8.append(val_loss8)

            if val_loss2 < min_vl2:
                min_vl2 = val_loss2
                print(f'New best 2: {min_vl2}')
                torch.save(model2.state_dict(), f'{root_folder}saved_models/wta2.pt')

            if val_loss4 < min_vl4:
                min_vl4 = val_loss4
                print(f'New best 4: {min_vl4}')
                torch.save(model4.state_dict(), f'{root_folder}saved_models/wta4.pt')

            if val_loss8 < min_vl8:
                min_vl8 = val_loss8
                print(f'New best 8: {min_vl8}')
                torch.save(model8.state_dict(), f'{root_folder}saved_models/wta8.pt')
  
        # if epoch % (val_per_epoch*10) == 0:
        #     draw_val_plot(root_folder, epoch)


    avg_loss2 += epoch_loss2/epoch_iter
    avg_loss4 += epoch_loss4/epoch_iter
    avg_loss8 += epoch_loss8/epoch_iter

    if epoch % val_per_epoch == 0:
        print("Epoch: {}, WTA-Losses: {}, {}, {}".format(epoch, avg_loss2/val_per_epoch, avg_loss4/val_per_epoch, avg_loss8/val_per_epoch))
        avg_loss2, avg_loss4, avg_loss8 = 0, 0, 0

# torch.save(torch.Tensor(training_loss_wta), wta_tr_loss_path)
# torch.save(torch.Tensor(validation_error_wta), wta_val_err_path)

# %%



