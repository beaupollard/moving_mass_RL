from isaac_env_make import env
import torch
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    tracked_dofs_vel=[0,1,2,3]
    tracked_dofs_pos=[]
    tracked_root=[7]    
    envs=env(tracked_dofs_pos=tracked_dofs_pos,tracked_dofs_vel=tracked_dofs_vel,tracked_root=tracked_root,viewer_flag=True)
    envs._setup_env()
    envs.reset()
    vel_rec=np.zeros((1000,4))
    state_rec=np.zeros((1000,5))
    i=0
    for i in range(1000):
        # a=6*torch.ones((1,4),dtype=torch.float32,device='cuda:0')
        a=torch.tensor([[5.,5.,5.,5.]],dtype=torch.float32,device='cuda:0')
        next_o, r, d, _, vel_rec[i,:] = envs.step(a)
        # next_o, r, d, _, _ = envs.step(a)
        # state_rec[i,:]=next_o.to(device="cpu").detach().numpy().flatten()
    print("stop")
    # for i in range(4):
    #     plt.plot(vel_rec[:,i])
