import numpy as np
import torch
from torch.optim import Adam
# import gym
import time
import core
import matplotlib.pyplot as plt
# from logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, [obs_dim[0],obs_dim[1]]), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, [act_dim[0],act_dim[1]]), dtype=np.float32)
        self.adv_buf = np.zeros(core.combined_shape(size, act_dim[0]), dtype=np.float32)
        self.rew_buf = np.zeros(core.combined_shape(size, act_dim[0]), dtype=np.float32)
        self.ret_buf = np.zeros(core.combined_shape(size, act_dim[0]), dtype=np.float32)
        self.val_buf = np.zeros(core.combined_shape(size, act_dim[0]), dtype=np.float32)
        self.logp_buf = np.zeros(core.combined_shape(size, act_dim[0]), dtype=np.float32)
        # self.adv_buf = np.zeros(size, dtype=np.float32)
        # self.rew_buf = np.zeros(size, dtype=np.float32)
        # self.ret_buf = np.zeros(size, dtype=np.float32)
        # self.val_buf = np.zeros(size, dtype=np.float32)
        # self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = np.zeros(act_dim[0],dtype=np.int32), np.zeros(act_dim[0],dtype=np.int32), size
        self.act_dim=act_dim[0]
    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        store_idx=np.where(self.ptr < self.max_size)[0]
        assert self.ptr.all() < self.max_size     # buffer has to have room so you can store
        # self.obs_buf[self.ptr] = obs[:,:,0].cpu().numpy()
        # self.act_buf[self.ptr] = act#.cpu().numpy()
        # self.rew_buf[self.ptr] = rew#.cpu().numpy()
        # self.val_buf[self.ptr] = val#.cpu().numpy()
        # self.logp_buf[self.ptr] = logp#.cpu().numpy()
        self.obs_buf[self.ptr[store_idx]]=obs[:,:,0].cpu().numpy()
        self.act_buf[self.ptr[store_idx]] = act#.cpu().numpy()
        self.rew_buf[self.ptr[store_idx],:] = rew[:,0].cpu().numpy()
        self.val_buf[self.ptr[store_idx],:] = val#.cpu().numpy()
        self.logp_buf[self.ptr[store_idx],:] = logp#.cpu().numpy()        
        self.ptr[store_idx] += 1

    def finish_path(self, last_val=0, fin_ids=[]):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        for i in fin_ids:
            path_slice = slice(self.path_start_idx[i], self.ptr[i])
            rews = np.append(self.rew_buf[path_slice,i], last_val[i])
            vals = np.append(self.val_buf[path_slice,i], last_val[i])
            
            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice,i] = core.discount_cumsum(deltas, self.gamma * self.lam)
            
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice,i] = core.discount_cumsum(rews, self.gamma)[:-1]
            
            self.path_start_idx[i] = self.ptr[i]

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        bs=self.obs_buf.shape[0]

        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx= np.zeros(self.act_dim,dtype=np.int32), np.zeros(self.act_dim,dtype=np.int32)
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = core.normalization_trick(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf.swapaxes(0, 1).reshape((self.obs_buf.shape[0]*self.obs_buf.shape[1],self.obs_buf.shape[2])), act=self.act_buf.swapaxes(0, 1).reshape((self.act_buf.shape[0]*self.act_buf.shape[1],self.act_buf.shape[2])), ret=self.ret_buf.swapaxes(0, 1).reshape((self.ret_buf.shape[0]*self.ret_buf.shape[1])),
                    adv=self.adv_buf.swapaxes(0, 1).reshape((self.adv_buf.shape[0]*self.adv_buf.shape[1])), logp=self.logp_buf.swapaxes(0, 1).reshape((self.logp_buf.shape[0]*self.logp_buf.shape[1])))
  
        # data = dict(obs=self.obs_buf.reshape((self.obs_buf.shape[0]*self.obs_buf.shape[1],self.obs_buf.shape[2])), act=self.act_buf.reshape((self.act_buf.shape[0]*self.act_buf.shape[1],self.act_buf.shape[2])), ret=self.ret_buf.reshape((self.ret_buf.shape[0]*self.ret_buf.shape[1])),
        #             adv=self.adv_buf.reshape((self.adv_buf.shape[0]*self.adv_buf.shape[1])), logp=self.logp_buf.reshape((self.logp_buf.shape[0]*self.logp_buf.shape[1])))
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppo(env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=10, 
        steps_per_epoch=512, epochs=10000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=3e-3, train_pi_iters=80, train_v_iters=80, lam=0.98, max_ep_len=1000,
        target_kl=0.05, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 
5e-3 1e-2
    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Set up logger and save configuration
    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    # Random seed
    seed += 10000 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Instantiate environment
    # env = env_fn()
    obs_dim = env.observation_space.size()
    # act_dim = env.action_space.size()
    act_dim = torch.ones((obs_dim[0],1)).size()#env.action_space.size()

    # Create actor-critic module
    # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac = actor_critic(env.observation_space, torch.tensor([[1]]), **ac_kwargs)
    device = torch.device("cuda:0")    #Save the model to the CPU
    ac.to(device)
    # Sync params across processes
    # sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    # local_steps_per_epoch = int(steps_per_epoch / num_procs())
    local_steps_per_epoch = int(4*steps_per_epoch/env.num_envs)
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    ent_weight=0.0
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data,ind=[]):
        if len(ind)==0:
            obs, act, adv, logp_old = torch.tensor(data['obs'],dtype=torch.float,device='cuda:0'),torch.tensor(data['act'],dtype=torch.float,device='cuda:0'), torch.tensor(data['adv'],dtype=torch.float,device='cuda:0'), torch.tensor(data['logp'],dtype=torch.float,device='cuda:0')
        else:
            obs, act, adv, logp_old = torch.tensor(data['obs'][ind],dtype=torch.float,device='cuda:0'),torch.tensor(data['act'][ind],dtype=torch.float,device='cuda:0'), torch.tensor(data['adv'][ind],dtype=torch.float,device='cuda:0'), torch.tensor(data['logp'][ind],dtype=torch.float,device='cuda:0')
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi-ent_weight*pi.entropy().mean(), pi_info #

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = torch.tensor(data['obs'],dtype=torch.float,device='cuda:0'), torch.tensor(data['ret'],dtype=torch.float,device='cuda:0')
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    pi_lr_decay=torch.optim.lr_scheduler.ExponentialLR(pi_optimizer, gamma=0.99)
    vf_lr_decay=torch.optim.lr_scheduler.ExponentialLR(vf_optimizer, gamma=0.99)
    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()
        # obs_np=data['obs'].to(device="cpu").detach().numpy()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            # kl = mpi_avg(pi_info['kl'])
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                # logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            # if i==1:
            #     print("step")
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), 0.5)
            pi_optimizer.step()        


        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            torch.nn.utils.clip_grad_norm_(ac.v.parameters(), 0.5)
            vf_optimizer.step()

        # Log changes from update
        return pi_info['kl'], pi_info_old['ent'], pi_info['cf'], loss_v.item()-v_l_old, loss_pi.item()-pi_l_old, data['obs'][-1,1].item()


    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        ep_ret_rec=[]
        ep_len_rec=[]
        buff_count=0
        for t in range(steps_per_epoch):

            a, v, logp = ac.step(torch.as_tensor(o[:,:,0], dtype=torch.float32))
            
            next_o, r, d, _, _ = env.step(a)
            ep_ret += r
            ep_len += 1
                        
            
            # save and log
            buf.store(o, a.cpu().numpy(), r, v, logp)
            
            # logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            # timeout = ep_len == max_ep_len
            terminal = d #or timeout
            # epoch_ended = t==steps_per_epoch-1
            
            if terminal.size()[0]>0:

                v = v*0
                # v=-10000*np.ones_like(v)
                buf.finish_path(v,fin_ids=terminal.cpu().numpy())
                # if terminal:
                #     # only save EpRet / EpLen if trajectory finished
                #     logger.store(EpRet=ep_ret, EpLen=ep_len)
                ep_len_rec.append(ep_len)
                ep_ret_rec.append(ep_ret)
                o, ep_ret, ep_len = env.reset(terminal.cpu().numpy()), 0, 0 

            if t==steps_per_epoch-1:
                _, v, _ = ac.step(torch.as_tensor(o[:,:,0], dtype=torch.float32))
                buf.finish_path(v,fin_ids=np.arange(0,len(o)))
                try:
                    ep_ret_print=torch.clone(ep_ret)
                except:
                    pass
                o, ep_ret, ep_len = env.reset(), 0, 0

            elif buff_count==local_steps_per_epoch-1:
                _, v, _ = ac.step(torch.as_tensor(o[:,:,0], dtype=torch.float32))
                buf.finish_path(v,fin_ids=np.arange(0,len(o)))

                # ep_ret_print=torch.clone(ep_ret)
                _,_,_,delta_v,delta_pi,cos_p = update()
                buff_count=0
            else:
                buff_count+=1

                
            


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            print('save state here')
            # logger.save_state({'env': env}, None)

        # Perform PPO update!
        _,_,_,delta_v,delta_pi,cos_p = update()
        # max_len=0
        # for ii in range(len(buf.obs_buf[0,:,0])):
        #     diff_obs=buf.obs_buf[1:t,ii,3]-buf.obs_buf[:t-1,ii,3]
        #     idx=np.where(diff_obs>0.5)[0]
        #     idx2=idx[1:]-idx[:-1]
        #     max_val=np.max(idx2)
        #     if max_val>max_len:
        #         max_len=max_val  
        # pi_lr_decay.step()
        # if epoch>100:
        #     vf_lr_decay.step()
        #     pi_lr_decay.step()
        print(epoch, ep_ret_print.mean().item(), cos_p, delta_v, delta_pi)
        # print(epoch, sum(ep_len_rec)/len(ep_len_rec), delta_v, delta_pi)    

        # Log info about epoch
        # logger.log_tabular('Epoch', epoch)
        # logger.log_tabular('EpRet', with_min_and_max=True)
        # logger.log_tabular('EpLen', average_only=True)
        # logger.log_tabular('VVals', with_min_and_max=True)
        # logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('ClipFrac', average_only=True)
        # logger.log_tabular('StopIter', average_only=True)
        # logger.log_tabular('Time', time.time()-start_time)
        # logger.dump_tabular()

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=64)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--cpu', type=int, default=4)
#     parser.add_argument('--steps', type=int, default=4000)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='ppo')
#     args = parser.parse_args()

#     mpi_fork(args.cpu)  # run parallel code with mpi

#     from spinup.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
#         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
#         logger_kwargs=logger_kwargs)