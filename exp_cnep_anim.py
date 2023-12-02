# %%
import os
import torch
from models.wta_cnp import WTA_CNP


root_path = "/home/yigit/projects/mbcnp/outputs/experimental/62D/"
run_id, model_folder = f'1700927469', 'saved_models/'
run_folder = f'{root_path}{run_id}/'
models_path = f'{run_folder}{model_folder}'

wta_model_path = f'{models_path}wta_on_synth.pt'

y = torch.load(f'{run_folder}y.pt').cpu()
num_samples, t_steps, dy = y.shape
dx = 1
batch_size = 1

x = torch.linspace(0, 1, t_steps).repeat(batch_size, 1).view(batch_size, -1, 1)
colors = ['tomato', 'aqua', 'limegreen', 'gold', 'mediumslateblue', 'lightcoral', 'darkorange', 'teal']

wta = WTA_CNP(1, dy, 10, 10, [512, 512, 512], num_decoders=1, decoder_hidden_dims=[512, 512, 512], batch_size=batch_size)

wta.load_state_dict(torch.load(wta_model_path))
wta.eval()

# %%
def get_batch(idx=[0], cond_idx=[0]):
    batch_obs = torch.zeros(batch_size, len(cond_idx), dy+dx, dtype=torch.float32)
    batch_obs[:, :, 0] = x[0, cond_idx, 0]

    batch_obs[:, :, 1:] = y[idx, cond_idx, :].view(batch_size, len(cond_idx), dy)
    
    return batch_obs, y[idx, :, :]

# %%
test_ind = 0
tar = torch.linspace(0, 1, t_steps).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)

obs, real_y = get_batch(list(range(batch_size)), [test_ind])

print(obs.shape, real_y.shape, tar.shape)

with torch.no_grad():
    pred_wta, gate = wta(obs, tar)

out_traj = pred_wta[0, 0, :, :dy].cpu().numpy()

# %%
print(out_traj.shape)

# %%
def replace_amc_values(original_file, out_traj, new_file):
    with open(original_file, 'r') as f, open(new_file, 'w') as new_f:
        lines = f.readlines()
        d_ctr = 0
        for i, line in enumerate(lines):
            # Skip header lines
            if i < 3:
                new_f.write(line)
                continue

            # Split the line into parts
            parts = line.split()
            if len(parts) > 1:
                # Replace the numeric values
                new_parts = [parts[0]]
                for j in range(len(parts)-1):
                    new_parts.append(str(out_traj[frame_id-1, d_ctr]))
                    d_ctr = (d_ctr + 1) % dy
                new_line = ' '.join(new_parts) + '\n'
                new_f.write(new_line)
            else:
                frame_id = int(parts[0])
                new_f.write(line)

replace_amc_values(f'{run_folder}out.amc', out_traj, f'{run_folder}new.amc')

# %%
print(out_traj[-1])

# %%
import time

from absl import app
from absl import flags
from dm_control.suite import humanoid_CMU
from dm_control.suite.utils import parse_amc
import matplotlib.pyplot as plt
import numpy as np

def play():
    env = humanoid_CMU.stand()

    max_num_frames = 90

    # Parse and convert specified clip.
    converted = parse_amc.convert(f'{run_folder}new.amc',
                                  env.physics, env.control_timestep())

    max_frame = max(max_num_frames, converted.qpos.shape[1] - 1)
    print('Plotting {} frames.'.format(max_frame))

    width = 480
    height = 480
    video = np.zeros((max_frame, height, width, 3), dtype=np.uint8)

    for i in range(max_frame):
      p_i = converted.qpos[:, i]
      with env.physics.reset_context():
        env.physics.data.qpos[:] = p_i
      video[i] = env.physics.render(height, width, camera_id=1)

  
    # for i in range(1, max_frame):
    #     img = plt.imshow(video[i])
    #     plt.title(i)
    #     plt.pause(0.01)  # Need min display time > 0.0.
    #     plt.draw()

    return converted, video

converted, video = play()

import matplotlib.animation as animation

def update_frame(i):
    img.set_array(video[i])
    return img,

# Create the initial plot.
fig = plt.figure()
img = plt.imshow(video[0])

ani = animation.FuncAnimation(fig, update_frame, frames=range(1, len(video)), interval=30, blit=True)

plt.show()

# # %%
# # import dm_control.locomotion as locomotion
# # from dm_control import composer

# # walker = locomotion.walkers.cmu_humanoid.CMUHumanoid()
# # arena = locomotion.arenas.Floor()
# # task = locomotion.tasks.stand.Stand(walker=walker, arena=arena)
# # env = composer.Environment(task=task, random_state=None)

# # walker = locomotion.walkers.cmu_humanoid.CMUHumanoid()
# # arena = locomotion.arenas.Floor()
# # task = locomotion.tasks.stand.Stand(walker=walker, arena=arena)
# # env = composer.Environment(task=task, random_state=None)


# # qpos_init = env.physics.bind(walker.qpos).data
# # qvel_init = env.physics.bind(walker.qvel).data

# # # Set the initial joint positions and velocities based on the first (qpos, qvel) tuple from the list
# # qpos_init[:] = converted.qpos[0][0]
# # qvel_init[:] = converted.qvel[0][1]

# # viewer = locomotion.viewers.MujocoViewer(env)
# # viewer.launch()

# # for i, data in enumerate(converted):
# #     qpos_target = data[0]
# #     qvel_target = data[1]

# #     # Set the target joint positions and velocities
# #     qpos_init[:] = qpos_target
# #     qvel_init[:] = qvel_target

# #     # Step the environment and record the next state
# #     for _ in range(100):
# #         env.step(env.random.uniform(env.action_spec().minimum, env.action_spec().maximum))

# from dm_control import mjcf
# import mujoco
# import numpy as np

# # Load the MuJoCo model
# model = mjcf.from_path("/home/yigit/.local/lib/python3.8/site-packages/dm_control/suite/humanoid_CMU.xml")
# d = mujoco.MjData(model)
# sim = mujoco.MjSim(model)

# with mujoco.viewer.launch_passive(model, d) as viewer:
#   start = time.time()
#   while viewer.is_running() and time.time() - start < 30:
#     step_start = time.time()

#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#     mujoco.mj_step(model, d)

#     # Example modification of a viewer option: toggle contact points every two seconds.
#     with viewer.lock():
#       viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
#     viewer.sync()

#     # Rudimentary time keeping, will drift relative to wall clock.
#     time_until_next_step = model.opt.timestep - (time.time() - step_start)
#     if time_until_next_step > 0:
#       time.sleep(time_until_next_step)

# # # Simulate the motion
# # for qpos, qvel, time in motion_data:
# #     sim.data.qpos[:len(qpos)] = qpos
# #     sim.data.qvel[:len(qvel)] = qvel
# #     sim.forward()
    
# #     # Simulate for the specified time duration
# #     num_steps = int(time / sim.model.opt.timestep)
# #     for _ in range(num_steps):
# #         sim.step()

# #     # Render or perform other actions as needed
# #     sim.render()

# # # Close the simulation
# # sim.close()



# # %%
# # from dm_control.locomotion.walkers.cmu_humanoid import _POSITION_ACTUATORS

# # #dm_control_dim_id: mocap_dim_id
# # indices = {0:16,1:17,2:18,3:50,4:51,5:52,6:47,7:48,8:49,9:19,
# #            10:20,11:21,12:53,13:54,14:55,15:0,16:1,17:2,18:28,19:29,
# #            20:38,21:39,22:40,23:41,24:46,25:36,26:37,27:33,28:42,29:43,
# #            30:3,31:4,32:13,33:14,34:15,35:22,36:27,37:11,38:12,39:8,
# #            40:23,41:24,42:30,43:31,44:32,45:44,46:34,47:35,48:45,49:5,
# #            50:6,51:7,52:25,53:9,54:10,55:26}

# # dm_traj = torch.zeros((t_steps, 56))
# # for i in range(t_steps):
# #     for j in range(56):
# #         dm_traj[i, j] = torch.tensor(y[0, i, indices[j]]) # / _POSITION_ACTUATORS[indices[j]][1][1]


# # %%
# # dm_traj = torch.zeros((t_steps, 56))
# # s_traj = y[0, :, :]
# # for i in range(t_steps):
# #     for j in range(56):
# #         dm_traj[i, j] = s_traj[i, indices[j]] / _POSITION_ACTUATORS[indices[j]][1][1]

# # %%


# # %%
# from dm_control import suite
# from dm_control.locomotion.examples import basic_cmu_2019
# import numpy as np
# from matplotlib import pyplot as plt
# from dm_control import viewer

# env = basic_cmu_2019.cmu_humanoid_go_to_target()
# time_step = env.reset()

# action_spec = env.action_spec()

# print('action_spec:', action_spec)

# width = 480
# height = 480
# video = np.zeros((698, height, 2 * width, 3), dtype=np.uint8)

# # # Step through an episode and print out reward, discount and observation.
# # action_spec = env.action_spec()
# # time_step = env.reset()

# viewer.launch(env)

# # # for i in range(698):  # Run for 1000 steps as an example
# # #     action = out_traj[i, 6:]
# # #     print(action)
# # #     break
# # #     env.step(action)

# for i in range(t_steps):
#     action = actions[i, :]
#     time_step = env.step(action)
#     video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
#                           env.physics.render(height, width, camera_id=1)])

# # for i in range(t_steps):
# #     img = plt.imshow(video[i])
# #     plt.pause(0.01)  # Need min display time > 0.0.
# #     plt.draw()




# %%
