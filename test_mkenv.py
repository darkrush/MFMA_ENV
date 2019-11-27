from MF_env import MultiFidelityEnv
from MSE import MSE_backend
from MF_env import basic,policy
from scenario.paser import  parse_senario
import time
import numpy as np
import math

class naive_policy(object):
    def __init__(self,policy_args):
        self.max_phi = policy_args['max_phi']
        self.l = policy_args['l']
        self.dist = policy_args['dist']
        self.min_r = self.l/np.tan(self.max_phi)
        self.right_o = np.array([self.min_r,0.0])
        self.left_o = np.array([-self.min_r,0.0])
    
    def inference(self,obs_list):
        action_list = []
        for obs in obs_list:
            obs = obs.pos
            theta = obs[2]
            xt = obs[3] - obs[0]
            yt = obs[4] - obs[1]
            xt,yt = (xt*np.cos(theta)+yt*np.sin(theta),yt*np.cos(theta)-xt*np.sin(theta))
    
            if abs(yt) < self.dist:
                vel = np.sign(xt)
                phi = 0
            else:
                in_min_r = (xt**2+(abs(yt)-self.min_r)**2)< self.min_r**2
                vel = -1 if np.bitwise_xor(in_min_r,xt<0) else 1
                phi = -1 if np.bitwise_xor(in_min_r,yt<0) else 1
            a = basic.Action()
            a.ctrl_vel = vel
            a.ctrl_phi = phi
            action_list.append(a)
        return action_list

def check_done(state_list):

    for state in state_list:
        if state.movable :
            return False
    return True


policy_args={}
policy_args['max_phi'] = math.pi/6.0
policy_args['l'] = 0.3
policy_args['dist'] = 0.1

r_policy = policy.RVO_policy()
n_policy = naive_policy(policy_args)

dt = 0.001
num_step = 100
repeat= 2
slow = 2.0
try:
    scenario = parse_senario('./scenario/scenario.yaml')
    back_end = MSE_backend.MSE_backend(scenario,dt,1.0)
    env = MultiFidelityEnv.MultiFidelityEnv(scenario,back_end)
    env.reset_rollout()
    env.rollout_sync(r_policy.inference,num_step,finish_call_back=check_done,delay = 0.0)
    print(env.get_result())
    #env.close()
except KeyboardInterrupt:
    env.close()  # inform the child thread that it should exit