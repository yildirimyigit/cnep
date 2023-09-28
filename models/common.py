import torch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class DataOps():
    def __init__(self, x, y, vx, vy, batch_size, n_max_obs, n_max_tar, device) -> None:
        self.dx, self.dy = x.shape[-1], y.shape[-1]
        self.t_steps = x.shape[1]
        self.n_max_obs, self.n_max_tar = n_max_obs, n_max_tar
        self.batch_size = batch_size

        self.x, self.y, self.vx, self.vy = x, y, vx, vy

        self.obs = torch.zeros(batch_size, n_max_obs, self.dx+self.dy, device=device)  # Initialize with maximum possible size
        self.tar = torch.zeros(batch_size, n_max_tar, self.dx, device=device)  # Initialize with maximum possible size
        self.tar_val = torch.zeros(batch_size, n_max_tar, self.dy, device=device)  # Initialize with maximum possible size

        self.v_tar = torch.zeros(batch_size, self.t_steps, self.dx, device=device)  # Full trajectory x
        self.v_tar_val = torch.zeros(batch_size, self.t_steps, self.dy, device=device)  # Full trajectory y

        # Create mask tensors
        self.obs_mask = torch.zeros((batch_size, n_max_obs, 1), dtype=torch.bool, device=device)
        self.tar_mask = torch.zeros((batch_size, n_max_tar, 1), dtype=torch.bool, device=device)

        self.v_tar_mask = torch.ones((batch_size, self.t_steps, 1), dtype=torch.bool, device=device)

        
    def get_batch(self, traj_ids):
        n_obs = torch.randint(1, self.n_max_obs+1, (1,)).item()  # Step 1: Random n_obs
        n_tar = torch.randint(1, self.n_max_tar+1, (1,)).item()  # Step 2: Random n_tar

        self.update_masks()
        
        for i in range(len(traj_ids)):

            random_query_ids_obs = torch.randperm(self.t_steps)
            random_query_ids_tar = torch.randperm(self.t_steps)
            
            o_ids = random_query_ids_obs[:n_obs]
            t_ids = random_query_ids_tar[:n_tar]

            self.obs[i, :n_obs, :] = torch.cat((self.x[traj_ids[i], o_ids], self.y[traj_ids[i], o_ids]), dim=-1)  # Steps 3 & 4
            self.tar[i, :n_tar, :] = self.x[traj_ids[i], t_ids]
            self.tar_val[i, :n_tar, :] = self.y[traj_ids[i], t_ids]

            self.obs_mask[i, :n_obs, :] = True
            self.tar_mask[i, :n_tar, :] = True

        return self.obs, self.tar, self.tar_val, self.obs_mask, self.tar_mask

    def get_validation_batch(self, traj_ids):
        n_obs = torch.randint(1, self.n_max_obs+1, (1,)).item()

        self.update_masks()

        for i in range(len(traj_ids)):
            random_query_ids = torch.randperm(self.t_steps)
            o_ids = random_query_ids[:n_obs]

            self.obs[i, :n_obs, :] = torch.cat((self.vx[traj_ids[i], o_ids], self.vy[traj_ids[i], o_ids]), dim=-1)
            self.v_tar[i, :, :] = self.vx[traj_ids[i]]
            self.v_tar_val[i, :, :] = self.vy[traj_ids[i]]

            self.obs_mask[i, :n_obs, :] = True

        return self.obs, self.v_tar, self.v_tar_val, self.obs_mask, self.v_tar_mask

    def update_masks(self):
        self.obs_mask.fill_(False)
        self.tar_mask.fill_(False)


    def draw_val_plot(self, root_folder, epoch, model_cnp, model_wta, colors):
        plt_y_lim_up = torch.max(self.vy) + 0.1
        plt_y_lim_low = torch.min(self.vy) - 0.1

        obs = torch.zeros((model_wta.num_decoders, 1, 1, 2)).to(self.device)
        for i in range(self.batch_size):
            obs[i] = torch.Tensor([self.x[i, 80, 0], self.y[i, 80, 0]]).unsqueeze(0).unsqueeze(0).to(self.device)

        tar = torch.linspace(0, 1, 200).unsqueeze(0).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            for i in range(self.batch_size):
                pred_cnp, _ = model_cnp(obs[i], tar)
                pred_wta, gate = model_wta(obs[i], tar)

                plt.ylim((plt_y_lim_low, plt_y_lim_up))
                plt.scatter(obs[i,:,:,0].cpu(), obs[i,:,:,1].cpu(), c='k')

                handles = []
                for j in range(model_wta.num_decoders):
                    plt.plot(torch.linspace(0, 1, 200), pred_wta[j,0,:,0].cpu(), colors[j], alpha=max(0.2, gate[0, 0, j].item()))  # wta pred
                    handles.append(Line2D([0], [0], label=f'gate{j}: {gate[0, 0, j].item():.4f}', color=colors[j]))

                plt.plot(torch.linspace(0, 1, 200), pred_cnp[:, :, :model_cnp.output_dim].squeeze(0).cpu(), 'b')  # cnp pred
                
                for j in range(self.batch_size):
                    plt.plot(torch.linspace(0, 1, 200), self.vy[j].squeeze(-1).cpu(), 'k', alpha=0.05 if j!=i else 0.35)  # data

                plt.legend(handles=handles, loc='upper right')

                plt.savefig(f'{root_folder}img/{i}_{epoch}.png')
                plt.close()