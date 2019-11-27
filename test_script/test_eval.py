from MF_env import MultiFidelityEnv,policy
from MSE import MSE_backend
from scenario.paser import  parse_senario
import torch
import argparse
import time
import numpy as np
import math
import signal

parser = argparse.ArgumentParser(description='DDPG on pytorch')
parser.add_argument('--test-case',default='./scenario/scenario_eval.yaml', type=str, help='curve smooth coef')

args = parser.parse_args()


dt = 0.01

ctrl_fpT = 1.0
ctrl_fps = int(1/dt/ctrl_fpT)


scenario = parse_senario(args.test_case)
#'./scenario/scenario_eval6.yaml'
agent_prop = scenario['default_agent']

eval_back_end = MSE_backend.MSE_backend(scenario,dt,1.0)
eval_back_end.use_gui = True

eval_env = MultiFidelityEnv.MultiFidelityEnv(scenario,eval_back_end)
#actor = torch.load('results/MAC/mix/15/actor.pkl')
actor = torch.load('results/MAC/rw/1/actor.pkl')
nnp = policy.NN_policy(actor,0,0.0)
rvop = policy.RVO_policy()
result_list = []
for test_idx in range(1):
    eval_env.reset_rollout()
    eval_env.rollout_sync(nnp.inference,ctrl_fps,delay=0.0)
    result = eval_env.get_result()
    #print(result)
    result_list.append(result)
item_list = ['mean_vel','reach_time','crash_time','total_reward']
for item_name in item_list:
    result = np.array([result[item_name] for result in result_list])
    print(item_name, np.mean(result),np.std(result))
#traj =  eval_env.get_trajectoy()
#traj = traj[0]
#key_list = ['obs','reward','done','obs_next']
#for data in zip(*[[tran[key] for tran in traj] for key in key_list]):
#    print(data)
eval_env.close()
