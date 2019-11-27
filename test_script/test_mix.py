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


train_dt = 0.01
search_dt = 1.0
eval_dt = 0.01

ctrl_fpT = 1.0
train_ctrl_fps = int(1/train_dt/ctrl_fpT)
search_ctrl_fps = int(1/search_dt/ctrl_fpT)
eval_ctrl_fps = int(1/eval_dt/ctrl_fpT)

scenario = parse_senario('./scenario/scenario_train.yaml')
eval_scenario = parse_senario('./scenario/scenario_6.yaml')
agent_prop = scenario['default_agent']

search_backend = MSE_backend.MSE_backend(scenario,search_dt)
search_backend.use_gui = False
search_env = MultiFidelityEnv.MultiFidelityEnv(scenario,search_backend)

eval_backend = MSE_backend.MSE_backend(eval_scenario,eval_dt)
eval_backend.use_gui = False
eval_env = MultiFidelityEnv.MultiFidelityEnv(eval_scenario,eval_backend)

backend = MSE_backend.MSE_backend(scenario,train_dt)
backend.use_gui = False
env = MultiFidelityEnv.MultiFidelityEnv(scenario,backend)
memory = Memory(int(4.8e6),(2,),[(2,),(agent_prop['N_laser'],)])
agent = DDPG(Singleton_arger()['agent'])
agent.setup(2,agent_prop['N_laser'],2,Singleton_arger()['model'])
trainer = DDPG_trainer()
trainer.setup(env,eval_env,search_env,agent,memory,train_ctrl_fps,search_ctrl_fps,eval_ctrl_fps)

#trainer.agent.load_weights('./results/MAC/3/314')
trainer.train()
eval_env.close()
env.close()
