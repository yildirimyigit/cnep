import torch


def read_frames_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    frames = []
    current_frame = []

    for line in lines:
        if line.strip().isdigit():
            # New frame, save the previous one
            if current_frame:
                frames.append(current_frame)
            current_frame = []
        else:
            # Extract numerical values from each line
            values = [float(val) for val in line.strip().split()[1:]]
            current_frame.extend(values)

    # Add the last frame
    if current_frame:
        frames.append(current_frame)

    return torch.tensor(frames)

file_path = 'experimental/data/0.txt'
data0 = read_frames_from_file(file_path)
file_path = 'experimental/data/1.txt'
data1 = read_frames_from_file(file_path)
# file_path = 'experimental/data/2.txt'
# data2 = read_frames_from_file(file_path)
file_path = 'experimental/data/3.txt'
data2 = read_frames_from_file(file_path)
file_path = 'experimental/data/4.txt'
data3 = read_frames_from_file(file_path)

min_frames = min(data0.shape[0], data1.shape[0], data2.shape[0], data3.shape[0])

data = torch.zeros((4, min_frames, data0.shape[1]))
data[0] = data0[torch.linspace(0, data0.shape[0] - 1, min_frames).long()]
data[1] = data1[torch.linspace(0, data1.shape[0] - 1, min_frames).long()]
data[2] = data2[torch.linspace(0, data2.shape[0] - 1, min_frames).long()]
data[3] = data3[torch.linspace(0, data3.shape[0] - 1, min_frames).long()]

print(data.shape)


# %%
from models.wta_cnp import WTA_CNP

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

batch_size = 2
n_max_obs, n_max_tar = 10, 10

t_steps = data.shape[1]
num_demos = 2
num_classes = 2
num_indiv = num_demos//num_classes  # number of demos per class
noise_clip = 0.0
dx, dy = 1, data.shape[2]

num_val = 2
num_val_indiv = num_val//num_classes

colors = ['tomato', 'aqua', 'limegreen', 'gold']

y = data[1:3].clone().to(device)
x = torch.unsqueeze(torch.linspace(0, 1, t_steps).repeat(num_demos, 1), -1).to(device)

vx = x[:2].clone()
# noise = torch.clamp(torch.randn(x.shape)*1e-4**0.5, min=0).to(device)
vy = torch.cat((data[0].clone().unsqueeze(0), data[-1].clone().unsqueeze(0)), 0).to(device)

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
optimizer_wta = torch.optim.Adam(lr=1e-5, params=model_wta_.parameters())

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

if not os.path.exists(f'{root_folder}img/'):
    os.makedirs(f'{root_folder}img/')

torch.save(y, f'{root_folder}y.pt')


epochs = 5_000_000
epoch_iter = num_demos//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)
avg_loss_wta = 0

val_per_epoch = 1000
min_val_loss_wta, min_val_loss_cnp = 1000000, 1000000

mse_loss = torch.nn.MSELoss()

training_loss_wta, validation_error_wta = [], []
training_loss_cnp, validation_error_cnp = [], []

wta_tr_loss_path = f'{root_folder}wta_training_loss.pt'
wta_val_err_path = f'{root_folder}wta_validation_error.pt'

for epoch in range(epochs):
    epoch_loss_wta = 0

    traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

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
            v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
            val_loss_wta = 0

            for j in range(v_epoch_iter):
                o_wta, t_wta, tr_wta = get_validation_batch(vx, vy, v_traj_ids[j], device=device)

                p_wta, g_wta = model_wta(o_wta, t_wta)
                dec_id = torch.argmax(g_wta.squeeze(1), dim=-1)
                vp_means = p_wta[dec_id, torch.arange(batch_size), :, :dy]
                val_loss_wta += mse_loss(vp_means, tr_wta).item()

            validation_error_wta.append(val_loss_wta)
            if val_loss_wta < min_val_loss_wta:
                min_val_loss_wta = val_loss_wta
                print(f'(WTA)New best: {min_val_loss_wta}')
                torch.save(model_wta_.state_dict(), f'{root_folder}saved_models/wta_on_synth.pt')
  
        # if epoch % (val_per_epoch*10) == 0:
        #     draw_val_plot(root_folder, epoch)


    avg_loss_wta += epoch_loss_wta

    if epoch % val_per_epoch == 0:
        print("Epoch: {}, WTA-Loss: {}".format(epoch, avg_loss_wta/val_per_epoch))
        avg_loss_wta = 0

torch.save(torch.Tensor(training_loss_wta), wta_tr_loss_path)
torch.save(torch.Tensor(validation_error_wta), wta_val_err_path)

# %%



