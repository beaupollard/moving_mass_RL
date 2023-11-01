from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np
# from mppi_ctrl import MPPI

class env():
    def __init__(self,tracked_dofs_pos=[],tracked_dofs_vel=[],tracked_root=[],viewer_flag=False):
        
        self.tracked_dofs_pos=tracked_dofs_pos  # Dof's to track with costs
        self.tracked_dofs_vel=tracked_dofs_vel  # Dof's to track with costs
        self.tracked_root=tracked_root  # Root Dof's to track with costs
        self.viewer_flag = viewer_flag


    def _setup_env(self):
       # initialize gym
        gym = gymapi.acquire_gym()

        sim_parms = gymapi.SimParams()
        sim_parms.dt = 0.02
        # gymapi.FlexParams.static_friction=1.0
        # gymapi.FlexParams.deterministic_mode=True 
        # sim_parms.physx.num_velocity_iterations=5
        sim_parms.physx.use_gpu = True
        sim_parms.use_gpu_pipeline = True
        if sim_parms.use_gpu_pipeline == True:
            self.device='cuda:0'
        else:
            self.device='cpu'
        sim = gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_parms)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.distance = -0.95
        gym.add_ground(sim, plane_params)

        # create viewer
        if self.viewer_flag == True:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                print("*** Failed to create viewer")
                quit()
            # position the camera
            cam_pos = gymapi.Vec3(17.2, 2.0, 16)
            cam_target = gymapi.Vec3(5, -2.5, 13)
            gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
            self.viewer = viewer

        # load asset
        asset_root = "../assets"
        asset_file = 'robot2.urdf'

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link =False
        asset_options.use_mesh_materials = False

        print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        # get array of DOF names
        dof_names = gym.get_asset_dof_names(asset)

        # get array of DOF properties
        dof_props = gym.get_asset_dof_properties(asset)

        # create an array of DOF states that will be used to update the actors
        num_dofs = gym.get_asset_dof_count(asset)
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

        # get list of DOF types
        dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

        # get the position slice of the DOF state array
        dof_positions = dof_states['pos']

        # set up the env grid
        num_envs = int(2**5)
        actors_per_env = 1
        dofs_per_actor = 11
        num_per_row = 4
        spacing = 2.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # cache useful handles
        self.envs = []
        self.actor_handles = []

        print("Creating %d environments" % num_envs)
        for i in range(num_envs):
            # create env
            self.env = gym.create_env(sim, env_lower, env_upper, num_per_row)
            self.envs.append(self.env)

            # add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
            pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

            self.actor_handle = gym.create_actor(self.env, asset, pose, "actor", i, 1)
            self.actor_handles.append(self.actor_handle)

            # set default DOF positions
            gym.set_actor_dof_states(self.env, self.actor_handle, dof_states, gymapi.STATE_ALL)

        gym.prepare_sim(sim)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)    

        # ## Aquire tensor descriptors #
        self.root_states_desc = gym.acquire_actor_root_state_tensor(sim)
        self.dof_states_desc = gym.acquire_dof_state_tensor(sim)
        self.force_buff=gym.acquire_dof_force_tensor(sim)

        # ## Pytorch interop ##
        root_states = gymtorch.wrap_tensor(self.root_states_desc)
        dof_states = gymtorch.wrap_tensor(self.dof_states_desc)
        self.dof_force=gymtorch.wrap_tensor(self.force_buff).reshape(num_envs,num_dofs)
        
        self.prev_root_state = torch.clone(root_states)
        self.prev_dof_state = torch.clone(dof_states)

        # ## View information as a vector of envs ##
        self.root_states_vec = root_states.view(num_envs,1,13)
        self.dof_states_vec = dof_states.view(num_envs,int(dof_states.size()[0]/num_envs),2)
        # self.rec_state=torch.zeros((num_envs,int(dof_states.size()[0]/num_envs),self.T),dtype=torch.float32, device=self.device)
        # self.new_state=torch.zeros((int(dof_states.size()[0]/num_envs),self.T),dtype=torch.float32, device=self.device)

        self.gym = gym
        self.sim = sim
        self.num_envs=num_envs
        self.num_dofs = num_dofs
        self.weights=torch.zeros(self.num_envs,len(self.tracked_root)+len(self.tracked_dofs_vel)+len(self.tracked_dofs_pos),len(self.tracked_root)+len(self.tracked_dofs_pos)+len(self.tracked_dofs_vel)).to(self.device)
        # self.weights[:]=torch.diag(torch.tensor([-0.0,-0.1,-.1,1.0,-0.],dtype=torch.float))
        # self.weights[:]=torch.diag(torch.tensor([-0.01,-0.01,-.01,1.0,-0.001,-0.001,-0.001,-0.001,-0.01],dtype=torch.float))
        # self.weights[:]=torch.diag(torch.tensor([-0.5,1.0,-0.01,-0.1],dtype=torch.float))
        self.weights[:]=torch.diag(torch.tensor([-5.0,1.0,-0.00,-0.0],dtype=torch.float))
        self.step()
        self.reset()
        self.observation_space = self.tracked_states_vec
        self.action_space = torch.zeros((self.num_envs,4),dtype=torch.float,device=self.device) 
        self.prev_vel = torch.zeros((self.num_envs,4),dtype=torch.float,device=self.device) 
        self.prev_action = torch.zeros((self.num_envs,4),dtype=torch.float,device=self.device) 

    def _render(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        self.gym.sync_frame_time(self.sim)

    def _compute_reward(self):
        # pendulum position is 3
        return torch.sum(torch.bmm(self.weights,self.tracked_states_vec),dim=1)#-0.1*torch.norm(self.dof_force[:,:4],dim=1).reshape((self.dof_force.size()[0]),1)
        
    def _terminal_flag(self):
        # return torch.where(self.tracked_states_vec[:,3,0]<0.5)[0]
        return torch.where(self.tracked_states_vec[:,0,0]>0.65)[0]
        # print('teop')
        # if self.tracked_states_vec[:,3,0]:
        #     return True
        # else:
        #     return False
    def pd_ctrl(self,des_action):
        kp=5.
        kd=0.
        # des_action.clip(min=-5.,max=5.)
        # action=1*des_action*torch.ones((des_action.size()[0],4),device="cuda:0",dtype=torch.float32)
        des_action=des_action*torch.ones((des_action.size()[0],4),device="cuda:0",dtype=torch.float32)
        action=kp*(des_action*torch.ones((des_action.size()[0],4),device="cuda:0",dtype=torch.float32)-self.dof_states_vec[:,:4,1])-kd*(self.dof_states_vec[:,:4,1]-self.prev_vel)
        
        # action=kp*(des_action-self.dof_states_vec[:,:4,1])-kd*(self.dof_states_vec[:,:4,1]-self.prev_vel)
        self.prev_vel=torch.clone(self.dof_states_vec[:,:4,1])
        return action

    def step(self,action=[]):
        # forces_desc = gymtorch.unwrap_tensor((self.U[:,:,i]+self.delta_U[:,:,i]).contiguous())
        if len(action)>0:
            action=self.pd_ctrl(action)
            self.dof_force[:,:4]=action
            forces_desc = gymtorch.unwrap_tensor(self.dof_force)
            self.gym.set_dof_actuation_force_tensor(self.sim,forces_desc)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self._refresh_state()
        next_obs=self.tracked_states_vec
        reward = self._compute_reward()
        done = self._terminal_flag()
        info=[]

        if self.viewer_flag==True:
            self._render()
        return next_obs, reward, done, info, self.dof_states_vec[0,:4,1].to("cpu").detach().numpy()

    def reset(self,idx=[]):
        ## Need to randomize base ##
        if len(idx)<1:
            idx=np.arange(0,self.num_envs)
        current_root_state=torch.clone(self.root_states_vec[:,0,:])
        current_dof_state=torch.clone(self.dof_states_vec)
        for i in idx:
            current_root_state[i,:]=torch.clone(self.prev_root_state[i])
            
            # current_dof_state[i,:,:]=torch.clone(self.prev_dof_state.view(self.num_envs,self.num_dofs,2)[i]+3.14)
            # current_dof_state[i,:,:]=torch.clone(self.prev_dof_state.view(self.num_envs,self.num_dofs,2)[i]-1.5*(torch.rand((self.num_dofs,2),dtype=torch.float,device=self.device)-0.5))
            current_dof_state[i,-1,0]=3.14+0.5*(torch.rand((1,1),dtype=torch.float,device=self.device)-0.5)
            current_dof_state[i,-1,1]=1.*(torch.rand((1,1),dtype=torch.float,device=self.device)-0.5)
            # current_dof_state[i,-1,0]=3.14*(torch.rand((1,1),dtype=torch.float,device=self.device)-0.5)
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(current_root_state))
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(current_dof_state))       
        # self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.prev_root_state.view(self.num_envs,13)))
        # self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.prev_dof_state.view(self.num_envs,self.num_dofs,2)+0.5*torch.rand((self.num_envs,self.num_dofs,2),dtype=torch.float,device=self.device)))
        self._refresh_state()
        next_obs=self.tracked_states_vec
        return next_obs
    
    def _refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.tracked_states_vec=torch.cat((torch.abs(self.root_states_vec[:,0,self.tracked_root]),self.dof_states_vec[:,self.tracked_dofs_pos,0],torch.abs(self.dof_states_vec[:,self.tracked_dofs_vel,1])),1).view(self.num_envs,len(self.tracked_dofs_vel)+len(self.tracked_dofs_pos)+len(self.tracked_root),1)
        # self.tracked_states_vec[:,3,0]=1+torch.cos(self.tracked_states_vec[:,3,0])
        self.tracked_states_vec[:,1,0]=1+torch.cos(self.tracked_states_vec[:,1,0])
# rec_rew=[]
# if __name__ == '__main__':
#     tracked_dofs=[4]
#     tracked_root=[0,1,7]
    
#     envs=env(tracked_dofs=tracked_dofs,tracked_root=tracked_root)
#     envs._setup_env()
    
#     for i in range(100):
#         next_obs, reward, done, info=envs.step()
#         rec_rew.append(reward)
    # envs.reset()
    