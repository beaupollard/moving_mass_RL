from isaac_env_make import env
import ppo

if __name__ == '__main__':
    # tracked_dofs_vel=[0,1,2,3,4]
    # tracked_dofs_pos=[4]
    # tracked_root=[0,5,7]
    tracked_dofs_vel=[0,1,2,3,4]
    tracked_dofs_pos=[4]
    tracked_root=[6,5,7,8]
    envs=env(tracked_dofs_pos=tracked_dofs_pos,tracked_dofs_vel=tracked_dofs_vel,tracked_root=tracked_root,viewer_flag=True)
    envs._setup_env()
    ppo_net=ppo.ppo(envs)
    
    # for i in range(100):
    #     next_obs, reward, done, info=envs.step()
    #     rec_rew.append(reward)