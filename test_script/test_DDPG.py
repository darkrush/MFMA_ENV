from MF_env import MultiFidelityEnv
from MSE import MSE_backend
from MF_env import basic,policy
from MFMA_DDPG.trainer import DDPG_trainer
from MFMA_DDPG.memory import Memory
from MFMA_DDPG.ddpg import DDPG
from MFMA_DDPG.arguments import Singleton_arger
from scenario.paser import  parse_senario
import torch

import time
import numpy as np
import math
import signal

class data_generator(object):
    def __init__(self,max_phi,l,dist,R_laser,N_laser):
        self.policy = policy.naive_policy(max_phi,l,dist)
        self.R_laser = R_laser
        self.N_laser = N_laser
    def gen(self,batch_size = 20):
        feild_size = 5
        pos_data_list1 = np.random.rand(batch_size,2)*[feild_size,feild_size]-[feild_size/2,feild_size/2]
        feild_size = 0.5
        pos_data_list2 = np.random.rand(batch_size*2,2)*[feild_size,feild_size]-[feild_size/2,feild_size/2]
        pos_data_list = np.vstack([pos_data_list1,pos_data_list2])
        laser_data_list = np.array([[self.R_laser]*self.N_laser]*(batch_size*3))
        action_data_list = self.policy.inference([pos_data_list,laser_data_list])
        pos_data_list = torch.tensor(pos_data_list,dtype=torch.float32).cuda()
        laser_data_list = torch.tensor(laser_data_list,dtype=torch.float32).cuda()
        action_data_list = torch.tensor(action_data_list,dtype=torch.float32).cuda()
        return pos_data_list,laser_data_list,action_data_list


dt = 0.0025

ctrl_fpT = 5.0
ctrl_fps = int(1/ctrl_fpT/dt)

scenario = parse_senario('./scenario/scenario.yaml')
eval_scenario = parse_senario('./scenario/scenario_eval.yaml')
agent_prop = scenario['default_agent']

eval_back_end = MSE_backend.MSE_backend(eval_scenario,dt)
eval_env = MultiFidelityEnv.MultiFidelityEnv(eval_scenario,eval_back_end)

back_end = MSE_backend.MSE_backend(scenario,dt)
env = MultiFidelityEnv.MultiFidelityEnv(scenario,back_end)
memory = Memory(int(1e6),(2,),[(2,),(agent_prop['N_laser'],)])
agent = DDPG(Singleton_arger()['agent'])
agent.setup(2,agent_prop['N_laser'],2,Singleton_arger()['model'])
trainer = DDPG_trainer()
trainer.setup(env,eval_env,agent,memory,ctrl_fps)

gener = data_generator(agent_prop['deg_K_phi'],agent_prop['L_axis'],agent_prop['R_reach'],agent_prop['R_laser'],agent_prop['N_laser'])
trainer.agent.load_weights('./')
#trainer.pretrain(gener.gen)
trainer.train()
eval_env.close()
env.close()