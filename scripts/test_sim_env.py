from isaac_env_make import env
import torch
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    tracked_dofs=[3]
    tracked_root=[0,1,7]
    
    envs=env(tracked_dofs=tracked_dofs,tracked_root=tracked_root,viewer_flag=True)
    envs._setup_env()
    envs.reset()
    vel_rec=np.zeros((1000,4))
    for i in range(1000):
        a=6*torch.ones((1,4),dtype=torch.float32,device='cuda:0')
        next_o, r, d, _, vel_rec[i,:] = envs.step(a)
    print("stop")
    for i in range(4):
        plt.plot(vel_rec[:,i])
