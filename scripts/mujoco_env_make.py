import mujoco as mj
import numpy as np
import copy 
from scipy.spatial.transform import Rotation as R

class sim_runner():
    def __init__(self,filename,active_names=[],tracked_bodies=[],render=False):
        self.tracked_dofs=tracked_dofs  # Dof's to track with costs
        self.tracked_root=tracked_root  # Root Dof's to track with costs
        self.viewer_flag = render  

        self.model=mj.MjModel.from_xml_path(filename)
        self.data=mj.MjData(self.model)
        self.init_data=copy.deepcopy(self.data)
        self.nq=len(self.data.qpos)
        self.nv=len(self.data.qvel)
        self.na=len(self.data.qacc)
        self.njnt=self.model.njnt
        self.nact=len(active_names)
        self.dt = self.model.opt.timestep

        ## Setup a list of active joint information ##
        self.active_joints=[]
        for i in active_names:
            self.active_joints.append({"name": i, "qpos_id": self.model.joint(i).qposadr.item(), \
                                       "qvel_id": self.model.joint(i).dofadr.item(),\
                                        "ctrllim":self.model.actuator(i).ctrllimited[0],\
                                            "ctrlrange":self.model.actuator(i).ctrlrange})

        ## Used to track pose of tracked bodies
        self.body_data=tracked_bodies
        for i in range(len(tracked_bodies)):
            if tracked_bodies[i]["type"]=="body":
                self.body_data[i]["data"]=self.data.body(tracked_bodies[i]["name"])
                self.body_data[i]["model"]=self.model.body(tracked_bodies[i]["name"])
            else:
                self.body_data[i]["data"]=self.data.joint(tracked_bodies[i]["name"])
                self.body_data[i]["model"]=self.model.joint(tracked_bodies[i]["name"])

        mj.mj_forward(self.model,self.data)
        while self.data.time<0.75:
            mj.mj_step(self.model,self.data)
        self.init_data=copy.deepcopy(self.data)

        self.qpos_ind=[]
        self.qvel_ind=[]
        for i in self.body_data:
            for j in i['qpos']:
                self.qpos_ind.append(i['model'].qposadr.item()+j)
            for j in i['qvel']:
                self.qvel_ind.append(i['model'].dofadr.item()+j)

        self.nstate=len(self.qvel_ind)+len(self.qpos_ind)
        self.forward_difference()
        self.render=render
        if render==True:
            self.setup_render()  

    def get_state(self,data):
        return np.concatenate((data.qpos[self.qpos_ind],data.qvel[self.qvel_ind]))
    
    def rec_state(self,data):
        return np.concatenate((data.qpos[self.qpos_ind],data.qvel[self.qvel_ind],data.qacc[self.qvel_ind]))

    def reset_data(self):
        self.data=copy.deepcopy(self.init_data)

    def set_configuration(self,xdes,vdes):
        for i, id in enumerate(self.qpos_ind):
            self.data.qpos[id]=xdes[i]
        for i, id in enumerate(self.qvel_ind): 
            self.data.qvel[id]=vdes[i]
        mj.mj_forward(self.model,self.data)

    def random_configuration(self,init=[]):
        self.reset_data()
        for i in range(len(self.data.qpos)):
            self.data.qpos[i]+=np.random.normal(0.,0.4)
        for i in range(len(self.data.qvel)):
            self.data.qvel[i]+=np.random.normal(0.,0.4)
        if len(init)>0:
            self.data.qpos[-1]=init[0]
            self.data.qvel[-1]=init[1]

        mj.mj_forward(self.model,self.data)

    def setup_render(self):
        ''' Setup visualization'''
        max_height = 100

        self.cam = mj.MjvCamera()
        
        self.opt = mj.MjvOption()

        mj.glfw.glfw.init()
        self.window = mj.glfw.glfw.create_window(1200, 900, "Demo", None, None)
        mj.glfw.glfw.make_context_current(self.window)
        mj.glfw.glfw.swap_interval(1)

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.cam.distance=5.
        self.cam.type=mj.mjtCamera.mjCAMERA_TRACKING
        self.cam.elevation=-15.
        self.cam.trackbodyid=1
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)   

    def render_image(self):
        ''' Render the current scene'''
        mj.glfw.glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, 1200, 900)

        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        mj.glfw.glfw.swap_buffers(self.window)
        mj.glfw.glfw.poll_events()

    def forward_difference(self,eps=1e-7):
        
        A = np.zeros((self.nstate,self.nstate))
        B = np.zeros((self.nstate,self.nact))
        Acount=0
        for i in self.qpos_ind:
            current_state_pos=copy.copy(self.data)
            current_state_neg=copy.copy(self.data)
            current_state_pos.qpos[i]+=eps
            current_state_neg.qpos[i]-=eps
            mj.mj_step(self.model,current_state_pos)
            mj.mj_step(self.model,current_state_neg)
            xpos=np.concatenate((current_state_pos.qpos[self.qpos_ind],current_state_pos.qvel[self.qvel_ind]))
            xneg=np.concatenate((current_state_neg.qpos[self.qpos_ind],current_state_neg.qvel[self.qvel_ind]))            
            # xpos=np.concatenate((current_state_pos.qvel[self.qvel_ind],current_state_pos.qacc[self.qvel_ind]))
            # xneg=np.concatenate((current_state_neg.qvel[self.qvel_ind],current_state_neg.qacc[self.qvel_ind]))
            A[:,Acount] = (xpos-xneg)/(2*eps)
            Acount+=1

        for i in self.qvel_ind:
            current_state_pos=copy.copy(self.data)
            current_state_neg=copy.copy(self.data)
            current_state_pos.qvel[i]+=eps
            current_state_neg.qvel[i]-=eps
            mj.mj_step(self.model,current_state_pos)
            mj.mj_step(self.model,current_state_neg)
            xpos=np.concatenate((current_state_pos.qpos[self.qpos_ind],current_state_pos.qvel[self.qvel_ind]))
            xneg=np.concatenate((current_state_neg.qpos[self.qpos_ind],current_state_neg.qvel[self.qvel_ind]))            
            # xpos=np.concatenate((current_state_pos.qvel[self.qvel_ind],current_state_pos.qacc[self.qvel_ind]))
            # xneg=np.concatenate((current_state_neg.qvel[self.qvel_ind],current_state_neg.qacc[self.qvel_ind]))
            A[:,Acount] = (xpos-xneg)/(2*eps)  
            Acount+=1      

        
        for ii in range(self.nact):
            current_state_pos=copy.copy(self.data)
            current_state_neg=copy.copy(self.data)
            current_state_pos.ctrl[ii]=current_state_pos.ctrl[ii]+eps
            current_state_neg.ctrl[ii]=current_state_neg.ctrl[ii]-eps
            mj.mj_step(self.model,current_state_pos)
            mj.mj_step(self.model,current_state_neg) 
            xpos=np.concatenate((current_state_pos.qpos[self.qpos_ind],current_state_pos.qvel[self.qvel_ind]))
            xneg=np.concatenate((current_state_neg.qpos[self.qpos_ind],current_state_neg.qvel[self.qvel_ind]))               
            # xpos=np.concatenate((current_state_pos.qvel[self.qvel_ind],current_state_pos.qacc[self.qvel_ind]))
            # xneg=np.concatenate((current_state_neg.qvel[self.qvel_ind],current_state_neg.qacc[self.qvel_ind]))           
            B[:,ii] = (xpos-xneg)/(2*eps)

        return A, B

    def forward_difference_total(self,eps=1e-7):
        
        A = np.zeros((self.nv*2,self.nv*2))
        B = np.zeros((self.nv*2,self.nact))
        Acount=0
        for ii in range(self.njnt):
            dofs=len(self.model.jnt(ii).qpos0)
            adr=self.model.jnt(ii).qposadr.item()
            jj=0
            while jj<dofs:
                
                if dofs==7 and jj==3:
                    jj+=1
                else:
                    current_state_pos=copy.copy(self.data)
                    current_state_neg=copy.copy(self.data)
                    current_state_pos.qpos[adr+jj]=current_state_pos.qpos[adr+jj]+eps
                    mj.mj_step(self.model,current_state_pos)
                    current_state_neg.qpos[adr+jj]=current_state_neg.qpos[adr+jj]-eps
                    mj.mj_step(self.model,current_state_neg)
                    A[:,Acount] = np.concatenate(((current_state_pos.qvel-current_state_neg.qvel)/(2*eps),(current_state_pos.qacc-current_state_neg.qacc)/(2*eps)))
                    Acount+=1
                    jj+=1
        
        for ii in range(self.nv):
            current_state=copy.copy(self.data)
            current_state.qvel[ii]=current_state.qvel[ii]+eps
            mj.mj_step(self.model,current_state)
            A[:,Acount] = np.concatenate(((current_state.qvel-self.data.qvel)/eps,(current_state.qacc-self.data.qacc)/eps))
            Acount+=1
        
        for ii in range(self.nact):
            current_state_pos=copy.copy(self.data)
            current_state_pos.ctrl[ii]=current_state_pos.ctrl[ii]+eps
            mj.mj_step(self.model,current_state_pos)
            current_state_neg=copy.copy(self.data)
            current_state_neg.ctrl[ii]=current_state_neg.ctrl[ii]-eps
            mj.mj_step(self.model,current_state_neg)            
            B[:,ii] = np.concatenate(((current_state_pos.qvel-current_state_neg.qvel)/(2*eps),(current_state_pos.qacc-current_state_neg.qacc)/(2*eps)))

        return A, B

    def step_forward(self,u=[],fext=[]):
        if len(fext)>0:
            for ii in range(len(fext[0,:])):
                self.data.xfrc_applied[ii,:]=fext[ii,:]
        for ii in range(len(u)):
            self.data.ctrl[ii]=u[ii]

        mj.mj_step(self.model,self.data)
        if self.render==True:
            self.render_image()  
        # return self.data

# filename='../urdf/cart_pend_wheel.xml'
# active_names=['right_front','left_front','right_back','left_back']
# body_names=[{"name": 'frame:base',"type": "joint", "qpos": [0], "qvel": [0]},{"name": 'pend',"type": "joint", "qpos": [0], "qvel": [0]}]

# sm=sim_runner(filename=filename,active_names=active_names,tracked_bodies=body_names,render=False)