import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from flowtaxis import FlowtaxisEnv
import numpy as np

import os

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
def custom_action(obs):
    if obs > 0.0:
        return 1.0
    elif obs < 0.0:
        return -1.0
    else:
        return 0.0

def main(sensor_l=0.25):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    flow=["freq=1.0_amp=0.24"]
    #######logging configuration#####

    ############
    ############## Hyperparameters ##############
    env_name = "Flowtaxis-v0"
    sensor=["delta_umag"]
    

    max_timesteps = 1900        # max timesteps in one episode
 
   
    #############################################
    
    # creating environment
    env = FlowtaxisEnv(sensor_l=sensor_l,MAX_ANGULARVELOCITY=5,velocity=2)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
 
    # ppo.policy_old.load_state_dict(torch.load('./PPO_continuous_Flowtaxis-v0_delta_umag_head_and_tail_42_190000.pth',map_location=device))
    # ppo.policy.load_state_dict(torch.load('./PPO_continuous_Flowtaxis-v0_delta_umag_head_and_tail_42_190000.pth',map_location=device))
    
    # logging variables


    obs = env.reset()
    for t in range(max_timesteps):

        # Running policy_old:
        action = custom_action(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            print(t)
            break
        # Saving reward and is_terminals:

        

     
if __name__ == '__main__':
    # from mpi4py import MPI

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    main(sensor_l=-0.25)


       
    
