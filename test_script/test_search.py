from MF_env import MultiFidelityEnv
from MSE import MSE_backend
from MF_env import policy
from scenario.paser import  parse_senario

def check_done(state_list):
    for state in state_list:
        if state.movable :
            return False
    return True

dt = 0.01
num_step = 10
repeat= 4
slow = 2.0

scenario = parse_senario('./scenario/scenario_eval.yaml')
back_end = MSE_backend.MSE_backend(scenario,dt,1.0)
back_end.use_gui = True
env = MultiFidelityEnv.MultiFidelityEnv(scenario,back_end)
env.reset_rollout()
search_back_end = MSE_backend.MSE_backend(scenario,dt,1.0)
search_back_end.use_gui = True
search_env = MultiFidelityEnv.MultiFidelityEnv(scenario,search_back_end)
start_time,state = env.get_state()
search_env.set_state(state,total_time = start_time)
[trace,index] = search_env.search_policy(num_step,repeat,2)
if trace is None :
    print("failed!!!")
    exit()
print(zip(*index))
t_policy = policy.trace_policy(trace,1)
env.rollout_sync(t_policy.inference,num_step,finish_call_back=check_done,delay = 0.0)
print(env.get_result())
env.close()