from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import os
import csv


is_save = True


def crop_left(im): 
    return transforms.functional.crop(im, top=0, left=0, height=420, width=560)


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.device = 'cuda:0'

        # Define a downsampling layer
        self.downsample = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1, stride=1),  # Reducing channels from 576 to 128
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))  # Reducing spatial dimensions to 10x10
        ).to(self.device)

    def hook(self, module, input, output):
        print(f"Hook called for layer: {module}")
        # Apply downsampling
        output = output.to(self.device)
        downsampled_output = self.downsample(output)
        self.features.append(downsampled_output)

    def register_hooks(self, layer_names):
        self.hooks = []
        for name, module in self.model.model.named_modules():  # Adjusted for YOLOv8 specific submodule access
            if name in layer_names:
                print(f"Registering hook for layer: {name}")
                hook = module.register_forward_hook(self.hook)
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def extract_features(self, img_path):
        self.features = []
        img = Image.open(img_path).convert('RGB')  # Load image using PIL
        transform = transforms.Compose([
            transforms.Lambda(crop_left),  # Crop the left side
            transforms.Pad(padding=(40, 110, 40, 110)),  # Pad to 640x640
            transforms.ToTensor()
        ])
        img = transform(img).unsqueeze(0).to('cuda:0')  # Transform to tensor and add batch dimension
        self.model(img)
        return self.features


# %%
num_demos = 24
t_steps = 400
dims = 256 * 16 * 16
feats = torch.zeros(num_demos, dims)
trajs = torch.zeros(num_demos, t_steps, 8)

if is_save:
    model = YOLO('yolov8m.pt').to('cuda:0')
    feature_extractor = FeatureExtractor(model)

    layer_names = ['model.21']
    feature_extractor.register_hooks(layer_names)

    # Extract features for a given image
    for i in range(num_demos):
        img_path = f'data/{i}/img.jpeg'
        features = feature_extractor.extract_features(img_path)
        feats[i] = features[0].view(-1)

        data_folder = f'/home/yigit/projects/cnep/baxter/data/{i}/'
        # iterate over all files in the in_folder
        for filename in os.listdir(data_folder):
            d = os.path.join(data_folder, filename)
            if filename.endswith('.csv'):
                temp_data = []
                with open(d, 'r') as f:
                    for j, line in enumerate(csv.reader(f)):
                        if j > 0:
                            temp_data.append([float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10])])  # p, q, gripper

        ids = torch.linspace(0, len(temp_data)-1, t_steps).int()

        for j in range(t_steps):
            trajs[i, j] = torch.tensor(temp_data[ids[j]])

    torch.save(trajs, 'trajs.pt')
    torch.save(feats, 'feats.pt')

    feature_extractor.remove_hooks()
else:
    trajs = torch.load('trajs.pt')
    feats = torch.load('feats.pt')

# %%
import sys
import torch
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns


folder_path = '../models/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

from cnep import CNEP
from cnmp import CNMP

torch.set_float32_matmul_precision('high')

def get_free_gpu():
    gpu_util = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)  # Switch GPU
#        gpu_util.append((i, torch.cuda.memory_stats()['reserved_bytes.all.current'] / (1024 ** 2)))
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
num_demos, v_num_demos = 2, 2
num_classes = 2  # Number of modes
num_indiv = num_demos // num_classes  # Number of trajectories per mode
num_val_indiv = v_num_demos // num_classes  # Number of trajectories per mode

dx = 1
dg = dims
dy = 8
batch_size = 2
n_max, m_max = 20, 20
t_steps = trajs.shape[1]

# %%
obs = torch.zeros((batch_size, n_max, dx+dg+dy), dtype=torch.float32, device=device)
tar_x = torch.zeros((batch_size, m_max, dx+dg), dtype=torch.float32, device=device)
tar_y = torch.zeros((batch_size, m_max, dy), dtype=torch.float32, device=device)
obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)

def prepare_masked_batch(traj_ids: list):
    obs.fill_(0)
    tar_x.fill_(0)
    tar_y.fill_(0)
    obs_mask.fill_(False)
    tar_mask.fill_(False)

    for i, traj_id in enumerate(traj_ids):
        traj = trajs[traj_id]
        feat = feats[traj_id]
        n = torch.randint(1, n_max, (1,)).item()
        m = torch.randint(1, m_max, (1,)).item()

        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = permuted_ids[n:n+m]
        
        obs[i, :n, :dx] = (n_ids/t_steps).unsqueeze(1)  # X
        obs[i, :n, dx:dx+dg] = feat.repeat(n, 1)  # G
        obs[i, :n, dx+dg:] = traj[n_ids]  # Y
        obs_mask[i, :n] = True
        
        tar_x[i, :m, :dx] = (m_ids/t_steps).unsqueeze(1)
        tar_x[i, :m, dx:] = feat.repeat(m, 1)
        tar_y[i, :m] = traj[m_ids]
        tar_mask[i, :m] = True

val_obs = torch.zeros((batch_size, n_max, dx+dg+dy), dtype=torch.float32, device=device)
val_tar_x = torch.zeros((batch_size, t_steps, dx+dg), dtype=torch.float32, device=device)
val_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)
val_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)

def prepare_masked_val_batch(traj_ids: list):
    val_obs.fill_(0)
    val_tar_x.fill_(0)
    val_tar_y.fill_(0)
    val_obs_mask.fill_(False)

    for i, traj_id in enumerate(traj_ids):
        traj = trajs[traj_id]
        feat = feats[traj_id]
        n = torch.randint(1, n_max, (1,)).item()

        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = torch.arange(t_steps)
        
        val_obs[i, :n, :dx] = (n_ids/t_steps).unsqueeze(1)
        val_obs[i, :n, dx:dx+dg] = feat.repeat(n, 1)
        val_obs[i, :n, dx+dg:] = traj[n_ids]
        val_obs_mask[i, :n] = True
        
        val_tar_x[i, :, :dx] = (m_ids/t_steps).unsqueeze(1)
        val_tar_x[i, :, dx:] = feat.repeat(t_steps, 1)
        val_tar_y[i] = traj[m_ids]

# %%
cnep_ = CNEP(dx+dg, dy, n_max, n_max, [512,128], num_decoders=2, decoder_hidden_dims=[128, 128], batch_size=batch_size, scale_coefs=True, device=device)
optimizer_cnep = torch.optim.Adam(lr=3e-4, params=cnep_.parameters())

cnmp_ = CNMP(dx+dg, dy, n_max, m_max, [512,128], decoder_hidden_dims=[256, 256], batch_size=batch_size, device=device)
optimizer_cnmp = torch.optim.Adam(lr=3e-4, params=cnmp_.parameters())

def get_parameter_count(model):
    total_num = 0
    for param in model.parameters():
        total_num += param.shape.numel()
    return total_num

print("cnep:", get_parameter_count(cnep_))
print("cnmp:", get_parameter_count(cnmp_))

if torch.__version__ >= "2.0":
    cnep, cnmp = torch.compile(cnep_), torch.compile(cnmp_)
else:
    cnep, cnmp = cnep_, cnmp_

# %%
import time
import os

timestamp = int(time.time())
root_folder = f'../outputs/baxter/cnmp_cnep/{str(timestamp)}/'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f'{root_folder}saved_models/'):
    os.makedirs(f'{root_folder}saved_models/')


epochs = 1_000_000
epoch_iter = num_demos//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
v_epoch_iter = v_num_demos//batch_size  # number of batches per validation (e.g. 100//32 = 3)
avg_loss_cnmp, avg_loss_cnep = 0, 0

val_per_epoch = 1000
min_vl_cnmp, min_vl_cnep = 1000000, 1000000

mse_loss = torch.nn.MSELoss()

tl_cnmp, tl_cnep = [], []
ve_cnmp, ve_cnep = [], []

cnmp_tl_path, cnep_tl_path = f'{root_folder}cnmp_training_loss.pt', f'{root_folder}cnep_training_loss.pt'
cnmp_ve_path, cnep_ve_path = f'{root_folder}cnmp_validation_error.pt', f'{root_folder}cnep_validation_error.pt'

for epoch in range(epochs):
    epoch_loss_cnmp, epoch_loss_cnep = 0, 0

    traj_ids = torch.randperm(num_demos)[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

    for i in range(epoch_iter):
        prepare_masked_batch(traj_ids[i])

        optimizer_cnmp.zero_grad()
        pred = cnmp(obs, tar_x, obs_mask)
        loss = cnmp.loss(pred, tar_y, tar_mask)
        loss.backward()
        optimizer_cnmp.step()

        epoch_loss_cnmp += loss.item()

        optimizer_cnep.zero_grad()
        pred, gate = cnep(obs, tar_x, obs_mask)
        loss, nll = cnep.loss(pred, gate, tar_y, tar_mask)
        loss.backward()
        optimizer_cnep.step()

        epoch_loss_cnep += nll.item()

    epoch_loss_cnmp = epoch_loss_cnmp/epoch_iter
    tl_cnmp.append(epoch_loss_cnmp)
    epoch_loss_cnep = epoch_loss_cnep/epoch_iter
    tl_cnep.append(epoch_loss_cnep)

    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            v_traj_ids = torch.randperm(v_num_demos)[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
            val_err_cnmp, val_err_cnep = 0, 0

            for j in range(v_epoch_iter):
                prepare_masked_val_batch(v_traj_ids[j])

                p = cnmp.val(val_obs, val_tar_x, val_obs_mask)
                vp_means = p[:, :, :dy]
                val_err_cnmp += mse_loss(vp_means, val_tar_y).item()

                p, g = cnep.val(val_obs, val_tar_x, val_obs_mask)
                dec_id = torch.argmax(g.squeeze(1), dim=-1)
                vp_means = p[dec_id, torch.arange(batch_size), :, :dy]
                val_err_cnep += mse_loss(vp_means, val_tar_y).item()

            val_err_cnmp = val_err_cnmp/v_epoch_iter
            val_err_cnep = val_err_cnep/v_epoch_iter

            if val_err_cnmp < min_vl_cnmp:
                min_vl_cnmp = val_err_cnmp
                print(f'CNMP New best: {min_vl_cnmp}')
                torch.save(cnmp_.state_dict(), f'{root_folder}saved_models/cnmp.pt')

            if val_err_cnep < min_vl_cnep:
                min_vl_cnep = val_err_cnep
                print(f'CNEP New best: {min_vl_cnep}')
                torch.save(cnep_.state_dict(), f'{root_folder}saved_models/cnep.pt')

            ve_cnmp.append(val_err_cnmp)
            ve_cnep.append(val_err_cnep)

    avg_loss_cnmp += epoch_loss_cnmp
    avg_loss_cnep += epoch_loss_cnep

    if epoch % val_per_epoch == 0:
        print("Epoch: {}, Loss: {}, {}, Min Err: {}, {}".format(epoch, avg_loss_cnmp/val_per_epoch, avg_loss_cnep/val_per_epoch, min_vl_cnmp, min_vl_cnep))
        avg_loss_cnmp, avg_loss_cnep = 0, 0

torch.save(torch.Tensor(tl_cnmp), cnmp_tl_path)
torch.save(torch.Tensor(ve_cnmp), cnmp_ve_path)

# %%



