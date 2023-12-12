from isaac_env_make import env
import torch
import numpy as np
import matplotlib.pyplot as plt
from isaacgym import gymutil, gymapi
from scipy import signal

if __name__ == '__main__':
    tracked_dofs_vel=[0,1,2,3,4]
    tracked_dofs_pos=[4]
    tracked_root=[6,5,7,8]
    envs=env(tracked_dofs_pos=tracked_dofs_pos,tracked_dofs_vel=tracked_dofs_vel,tracked_root=tracked_root,viewer_flag=True)
    envs._setup_env()
    envs.reset()
    pos_rec=np.zeros((10000,1))
    vel_rec=np.zeros((10000,4))
    state_rec=np.zeros((10000,5))
    i=0
    a=1.1*torch.ones((128,5),dtype=torch.float32,device='cuda:0')
    a[:,-1]=0.
    for i in range(10000):
        
        # a=torch.tensor([[5.,5.,5.,5.,0]],dtype=torch.float32,device='cuda:0')
        # a=1*torch.ones((4,1),dtype=torch.float32,device='cuda:0')
        next_o, r, d, _, _ = envs.step(a)
        # next_o, r, d, _, _ = envs.step(a)
        # state_rec[i,:]=next_o.to(device="cpu").detach().numpy().flatten()
    print("stop")
    # for i in range(4):
    #     plt.plot(vel_rec[:,i])
sos = signal.butter(10, 10, 'lp', fs=1/0.02, output='sos')
filtered = signal.sosfilt(sos,vel_rec[:i-1,1])
plt.plot(filtered)