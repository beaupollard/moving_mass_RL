from isaac_env_make import env
import ppo

if __name__ == '__main__':
    tracked_dofs=[4]
    tracked_root=[0,1,7]
    
    envs=env(tracked_dofs=tracked_dofs,tracked_root=tracked_root,viewer_flag=False)
    envs._setup_env()
    ppo_net=ppo.ppo(envs)
    
    # for i in range(100):
    #     next_obs, reward, done, info=envs.step()
    #     rec_rew.append(reward)