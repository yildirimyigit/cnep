# %%
from models.cnp import CNP
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

print("device:", device)

# %%
batch_size = 32
n_max, m_max = 10, 10
t_steps = 200

num_train, num_val = 1024, 128

dx, dy = 20, 20

def continuous_function(x):
    y = torch.sin(x) + torch.cos(x)**2 + torch.tanh(x)
    return y

# %%
x = torch.rand(num_train, t_steps, dx)
y = continuous_function(x)

vx = torch.rand(num_val, t_steps, dx)
vy = continuous_function(vx)

x, y = x.to(device), y.to(device)
vx, vy = vx.to(device), vy.to(device)

print(x.shape, y.shape)
print(vx.shape, vy.shape)

# %%
model = CNP(input_dim=dx, hidden_dim=512, output_dim=dy, n_max_obs=n_max, n_max_tar=m_max, num_layers=3, batch_size=batch_size).to(device)
optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

# if torch.__version__ >= "2.0":
#     model = torch.compile(model_)

# %%
def get_batch(x, y, traj_ids, device=device):
    n_o = torch.randint(1, n_max, (1,)).item()
    n_t = torch.randint(1, m_max, (1,)).item()
    
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
    num_obs = torch.randint(1, n_max, (1,)).item()

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
import time
import os

# torch._dynamo.config.suppress_errors = True

timestamp = int(time.time())
root_folder = f'outputs/test/{str(timestamp)}/'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f'{root_folder}saved_model/'):
    os.makedirs(f'{root_folder}saved_model/')

# if not os.path.exists(f'{root_folder}img/'):
#     os.makedirs(f'{root_folder}img/')

# torch.save(y, f'{root_folder}y.pt')


epochs = 5_000#_000
epoch_iter = num_train//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)

val_per_epoch = 1000  # validation frequency
min_val_loss = 1_000_000

mse_loss = torch.nn.MSELoss()

training_loss, validation_error = [], []
avg_loss = 0

tr_loss_path = f'{root_folder}training_loss.pt'
val_err_path = f'{root_folder}validation_error.pt'

for epoch in range(epochs):
    epoch_loss = 0

    traj_ids = torch.randperm(num_train)[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

    for i in range(epoch_iter):
        optimizer.zero_grad()
        obs, tar_x, tar_y = get_batch(x, y, traj_ids[i], device)

        pred, _ = model(obs, tar_x)

        loss = model.loss(pred, tar_y)  # mean loss over the batch

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= epoch_iter  # mean loss over the epoch
    
    training_loss.append(epoch_loss)

    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            v_traj_ids = torch.randperm(num_val)[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
            val_loss = 0

            for j in range(v_epoch_iter):
                o, t, tr = get_validation_batch(vx, vy, v_traj_ids[j], device)

                p, _ = model(o, t)
                val_loss += mse_loss(p[:, :, :dy], tr).item()

            validation_error.append(val_loss)
            if val_loss < min_val_loss and epoch > 5e3:
                min_val_loss = val_loss
                print(f'New best: {min_val_loss}')
                torch.save(model.state_dict(), f'{root_folder}saved_model/best.pt')

    avg_loss += epoch_loss

    if epoch % val_per_epoch == 0:
        print("Epoch: {}, Loss: {}".format(epoch, avg_loss/val_per_epoch))
        avg_loss = 0

torch.save(torch.Tensor(training_loss), tr_loss_path)
torch.save(torch.Tensor(validation_error), val_err_path)

# %%



