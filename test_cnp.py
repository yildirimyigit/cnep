# %%
from models.cnp import CNP
from data.data_generators import *
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

# %%
# Data generation
#import matplotlib.pyplot as plt

dx = 1
t_steps = 200
num_train_per_class, num_val_per_class = 32, 8
num_classes = 4

x = torch.linspace(0, 1, t_steps).view(-1, 1)

generator_functions = [generate_sin, generate_cos, generate_cx_sigm, 
    generate_reverse_cx_sigm]
y = []
for i in range(num_train_per_class + num_val_per_class):
    for j in range(num_classes):
        y.append(generator_functions[j](x))

colors = ["b", "r", "g", "y"]
# for i, y_i in enumerate(y):
#     plt.plot(y_i, alpha=0.5, c=colors[i%num_classes])

x = x.unsqueeze(0).repeat(len(y), 1, 1).to(device)
y = torch.stack(y, dim=0).to(device)

vx, vy = x[num_train_per_class*num_classes:], y[num_train_per_class*num_classes:]
x, y = x[:num_train_per_class*num_classes], y[:num_train_per_class*num_classes]

#print(x.shape, y.shape)
#print(vx.shape, vy.shape)

# %%
batch_size = 32

model = CNP(input_dim=1, hidden_dim=287, output_dim=1, n_max_obs=10, n_max_tar=10, num_layers=2, batch_size=batch_size).to(device)
optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

# %%
def get_batch(x, y, traj_ids):
    dx, dy = x.shape[-1], y.shape[-1]
    n = x.shape[1]
    n_t = torch.randint(1, model.n_max_tar, (1,)).item()
    n_o = torch.randint(1, model.n_max_obs, (1,)).item()

    obs = torch.zeros(batch_size, n_o, dx+dy).to(device)
    tar = torch.zeros(batch_size, n_t, dx).to(device)
    tar_val = torch.zeros(batch_size, n_t, dy).to(device)

    for i in range(len(traj_ids)):
        random_query_ids = torch.randperm(n)
        o_ids = random_query_ids[:n_o]
        t_ids = random_query_ids[n_o:n_o+n_t]

        # print(x.shape, traj_ids[i], o_ids, t_ids)

        obs[i, :, :] = torch.cat((x[traj_ids[i], o_ids], y[traj_ids[i], o_ids]), dim=-1)
        tar[i, :, :] = x[traj_ids[i], t_ids]
        tar_val[i, :, :] = y[traj_ids[i], t_ids]

    return obs, tar, tar_val

# %%
def get_validation_batch(vx, vy, o_ids=[0, -1]):
    obs = torch.cat((vx[:, o_ids, :], vy[:, o_ids, :]), dim=-1)
    tar = vx[:, torch.arange(t_steps)]
    tar_val= vy[:, torch.arange(t_steps)]

    return obs, tar, tar_val

# %%
import time

file_name = int(time.time())

epochs = 5000000
epoch_iter = 4

avg_loss = 0

val_per_epoch = 1000
min_val_loss = 1000000

mse_loss = torch.nn.MSELoss()

training_loss, validation_error = [], []

tr_loss_path = f'training_loss_{file_name}.pt'
val_err_path = f'validation_error_{file_name}.pt'

for epoch in range(epochs):
    epoch_loss = 0

    traj_ids = torch.randperm(x.shape[0])[:batch_size*epoch_iter].chunk(epoch_iter)

    for i in range(epoch_iter):
        optimizer.zero_grad()
        obs, tar_x, tar_y = get_batch(x, y, traj_ids[i])
        pred, encoded_rep = model(obs, tar_x)
        loss = model.loss(pred, tar_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    training_loss.append(epoch_loss)

    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            obs, tar_x, tar_y = get_validation_batch(vx, vy)
            pred, encoded_rep = model(obs, tar_x)
            val_loss = mse_loss(pred[:, :, :model.output_dim], tar_y)
            validation_error.append(val_loss.item())
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print(f'New best: {min_val_loss}')
                torch.save(model.state_dict(), f'saved_models/cnp_synth_{file_name}.pt')

    avg_loss += epoch_loss

    if epoch % 100 == 0:
        print("Epoch: {}, Loss: {}".format(epoch, avg_loss/100))
        avg_loss = 0

    if epoch % 100000:
        torch.save(torch.Tensor(training_loss), tr_loss_path)
        torch.save(torch.Tensor(validation_error), val_err_path)

# %%



