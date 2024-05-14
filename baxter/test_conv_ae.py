# %%
import torch
import matplotlib.pyplot as plt
import os

from PIL import Image
import torchvision.transforms as transforms

from conv_autoenc import ConvAE

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
batch_size = 8
data = []

img_folder = '/home/yigit/projects/cnep/baxter/data/img/'

def crop_left(im): 
    return transforms.functional.crop(im, top=0, left=0, height=480, width=480)

img_tf = transforms.Compose([
    transforms.Lambda(crop_left),  # Crop the left side
    transforms.Lambda(lambda x: x.convert('RGB')),  # Ensure the image is in RGB mode
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Resize((128, 128), antialias=True),  # Downsample to 120x120
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
])

# iterate over all files in the in_folder
for filename in os.listdir(img_folder):
    if filename.endswith('.jpeg'):
        img = img_tf(Image.open(os.path.join(img_folder, filename)))
        data.append(img)

imgs = torch.stack(data, dim=0)

num_train = 24
num_val = 8
epoch_iter = num_train//batch_size
v_epoch_iter = num_val//batch_size
x = imgs[:num_train]
vx = imgs[num_train:]

# %%
model_ = ConvAE(filter_sizes=[1024,768,512,256])
optimizer = torch.optim.Adam(lr=1e-4, params=model_.parameters())

if torch.__version__ >= "2.0":
    model = torch.compile(model_)

# %%
import time
import os
timestamp = int(time.time())
root_folder = f'output/ae/{str(timestamp)}/'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f'{root_folder}saved_model/'):
    os.makedirs(f'{root_folder}saved_model/')


epochs = 5_000_000

val_per_epoch = 200  # validation frequency
min_val_error = 1_000_000

mse_loss = torch.nn.MSELoss()

training_loss, validation_error = [], []
avg_loss_for_n_epochs = 0

tr_loss_path = f'{root_folder}training_loss.pt'
val_err_path = f'{root_folder}validation_error.pt'

for epoch in range(epochs):
    mean_epoch_loss = 0

    img_ids = torch.randperm(num_train)[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

    for i in range(epoch_iter):
        optimizer.zero_grad()
        input = x[img_ids[i]]
        pred = model(input)
        loss = model.loss(pred, input)  # mean loss over the batch
        loss.backward()
        optimizer.step()

        mean_epoch_loss += loss.item()

    mean_epoch_loss /= epoch_iter  # mean loss over the epoch
    training_loss.append(mean_epoch_loss)

    avg_loss_for_n_epochs += mean_epoch_loss

    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            v_traj_ids = torch.randperm(vx.shape[0])[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
            val_epoch_err = 0

            for j in range(v_epoch_iter):
                vinput = vx[v_traj_ids[j]]
                vpred = model(vinput)
                vloss = model.loss(vpred, vinput)
                val_epoch_err += vloss.item()

            if val_epoch_err < min_val_error:
                min_val_error = val_epoch_err
                print(f'New best: {min_val_error}')
                torch.save(model_.state_dict(), f'{root_folder}saved_model/cae.pt')

    if epoch % 100 == 0:
        print("Epoch: {}, Loss: {}".format(epoch, avg_loss_for_n_epochs/100))
        avg_loss_for_n_epochs = 0

# %%



