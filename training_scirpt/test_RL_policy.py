
from flowtaxis import FlowtaxisEnv
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import os
import matplotlib.pyplot as plt
from PPO_continuous import PPO, Memory
device=torch.device("cpu")
def custom_action(obs):
    if np.abs( obs)<0.0:
        return 0.0
    if obs > 0.0:
        return 1.0
    elif obs < 0.0:
        return -1.0

def main(sensor_l=0.25):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    flow_index=0
    sensor_index=0
    flow=["Re=5000_freq=1.25_amp=0.2","freq=0.5_amp=0.2","Re=5000_freq=0.5_amp=0.2","freq=1.75_amp=0.2"]
    flow_freq=[1.25,0.5,0.5,1.75]
    env_name = "Flowtaxis-v0"
    sensor=["delta_umag_ht","delta_pressure","delta_ub1","delta_ub2","delta_concentration","delta_vor_b1","delta_vor_b2"]
    max_timesteps = 4070        # max timesteps in one episode
    
    ICs=[[6.5,1.5,-np.pi/3,0],[6.5,-1.5,np.pi/2,0],[6.5,0.5,-np.pi/3,0],[6.5,0.02,0.02,0],[6.5,-0.25,-np.pi*2/3,0]]
    # ICs=[[4.5,0.5,np.pi/6,0]]
    # ICs=[]
    # for i in range(9):
    #     high=np.array([6.5,1.5,np.pi,0])
    #     low=np.array([6.5,-1.5,-np.pi,0])
    #     ICs.append(np.random.uniform(low,high))
    
    env = FlowtaxisEnv(FlowFiled=flow[flow_index],SensorMode=sensor[sensor_index],sensor_l=sensor_l,MAX_ANGULARVELOCITY=3,velocity=0.25,decision_timestep=0.05,flow_freq=flow_freq[flow_index])


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_std = 0.001            # constant std for action distribution (Multivariate Normal)
   
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.999                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps=max_timesteps)
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    polic_name="../../../paper/figure_preparation_new_new_20221028/policy/umag/excitatory/PPO_continuous_Flowtaxis-v0_delta_umag_ht_Re=5000_freq=1.25_amp=0.2_3700.pth"
    ppo.policy_old.load_state_dict(torch.load(polic_name,map_location=device))
    ppo.policy.load_state_dict(torch.load(polic_name,map_location=device))
    
    for III, IC in enumerate(ICs):
        env.reset()
        obs=env.reset_test(IC[0],IC[1],IC[2],IC[3],III)
        # print(III,obs)
        # input()
        total_cost=0
        for t in range(max_timesteps):

            # Running policy_old:
            action = ppo.select_action(obs, memory)
            obs, reward, done, _ = env.step(action)
            # total_cost+=cost
            if done:
                print(t)

                break
        a=env.get_arr()
        a=np.array(a)
        # print(a)
        # print(np.shape(a))
        plt.plot(a[:,0],a[:,1],'r')
        
        np.savetxt("./trajectory_ex_{}.txt".format(III),a)
    polic_name="../../../paper/figure_preparation_new_new_20221028/policy/umag/inhibitory/PPO_continuous_Flowtaxis-v0_delta_umag_ht_Re=5000_freq=1.25_amp=0.2_2800.pth"
    ppo.policy_old.load_state_dict(torch.load(polic_name,map_location=device))
    ppo.policy.load_state_dict(torch.load(polic_name,map_location=device))
    
    for III, IC in enumerate(ICs):
        env.reset()
        obs=env.reset_test(IC[0],IC[1],IC[2],IC[3],III)
        # print(III,obs)
        # input()
        total_cost=0
        for t in range(max_timesteps):

            # Running policy_old:
            action = ppo.select_action(obs, memory)
            obs, reward, done, _ = env.step(action)
            # total_cost+=cost
            if done:
                print(t)

                break
        a=env.get_arr()
        a=np.array(a)
        # print(a)
        # print(np.shape(a))
        plt.plot(a[:,0],a[:,1],'b')
        
        np.savetxt("./trajectory_in_{}.txt".format(III),a)
    
    plt.axis('equal')
    plt.xlim([-1,9])
    plt.ylim([-2,2])
    plt.show()
     
if __name__ == '__main__':
    # from mpi4py import MPI

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    main(sensor_l=0.25)


       
    
