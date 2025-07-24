
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
    if obs > 0.0:
        return 1.0
    elif obs < 0.0:
        return -1.0
    else:
        return 0.0

def main(sensor_l=0.25,angularvelocity=3,sensor_velocity=0.25,flow_index=0,time=0.0,sensor_index=0,excitatory=True):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    env_name = "Flowtaxis-v0"

    sensor=["delta_umag_ht","delta_ub1","delta_ub2","delta_vor_b2","delta_pressure","delta_vor_b1","delta_concentration"]
    flow=["Re=5000_freq=1.25_amp=0.2","Re=5000_freq=0.5_amp=0.2","Re=5000_freq=1.8_amp=0.2"]
    flow_freq=[1.2,0.5,1.8,1.25,0.5,1.8]
    max_timesteps = 4070        # max timesteps in one episode
    if excitatory:
        outputfile="pitching_airfoil_{}_{}_sensor_l_{}_sensor_velocity_{}_angular_velocity_{}_RL_excitatory_time={}.txt".format(flow[flow_index],sensor[sensor_index],format(sensor_l,'.3f'),format(sensor_velocity,'.3f'),format(angularvelocity,'.3f'),format(time,'.1f'))
    else:
        outputfile="pitching_airfoil_{}_{}_sensor_l_{}_sensor_velocity_{}_angular_velocity_{}_RL_inhibitory_time={}.txt".format(flow[flow_index],sensor[sensor_index],format(sensor_l,'.3f'),format(sensor_velocity,'.3f'),format(angularvelocity,'.3f'),format(time,'.1f'))
    f = open(outputfile, "a")

    report_energy=True

    env = FlowtaxisEnv(FlowFiled=flow[flow_index],sensor_l=sensor_l,MAX_ANGULARVELOCITY=angularvelocity,velocity=sensor_velocity,flow_freq=flow_freq[flow_index],SensorMode=sensor[sensor_index],report_energy=report_energy,decision_timestep=0.05)
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps=max_timesteps)
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
    if excitatory:
        polic_name="../../../paper/figure_preparation_new_new_20221028/policy/umag/excitatory/PPO_continuous_Flowtaxis-v0_delta_umag_ht_Re=5000_freq=1.25_amp=0.2_3700.pth"
    else:
        polic_name="../../../paper/figure_preparation_new_new_20221028/policy/umag/inhibitory/PPO_continuous_Flowtaxis-v0_delta_umag_ht_Re=5000_freq=1.25_amp=0.2_2800.pth"

    ppo.policy_old.load_state_dict(torch.load(polic_name,map_location=device))
    ppo.policy.load_state_dict(torch.load(polic_name,map_location=device))

    n_episode=20000

    for ep in range(n_episode):
              
                    
            ep_reward = 0
            # env = wrappers.Monitor(env, './results/movies/' + str(time.time()) + '/')
            obs = env.reset()
            agreement_x=0
            agreement_y=0
            agreement_rotation=0
            thrust_parameter_x=0
            thrust_parameter_y=0
            thrust_parameter_rotation=0
            agreement_rotation_norm=0
            thrust_parameter_y_norm=0
            x,y,th,t=env.state
            for t in range(max_timesteps):

                # Running policy_old:
                action = ppo.select_action(obs,memory)
                if report_energy:
                    obs, reward, done, data = env.step(action)
                    a1,a2,a3,a4,a5,a6,a7,a8=data["data"]
                    agreement_x+=a1
                    agreement_y+=a2
                    agreement_rotation+=a3
                    thrust_parameter_x+=a4
                    thrust_parameter_y+=a5
                    thrust_parameter_rotation+=a6
                    agreement_rotation_norm+=a7
                    thrust_parameter_y_norm+=a8
                else:
                    obs, reward, done, _ = env.step(action)
                ep_reward += reward
                if done:
                    # print(t)
                    break
            memory.clear_memory()
            a=env.get_arr()
            
            endx=a[-1][0]
            endy=a[-1][1]
            endth=a[-1][2]
            endt=a[-1][3]
            print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
            if report_energy:
                if (env.get_success()):
                    f.write(str(x)+"\t"+str(y)+"\t"+str(th)+"\t"+str(time)+"\t"+str(endx)+"\t"+str(endy)+"\t"+str(endth)+"\t"+str(endt)+"\t"+str(env.get_success())+"\t"+str(agreement_x)+"\t"+str(agreement_y)+"\t"+str(agreement_rotation)+"\t"+str(thrust_parameter_x)+"\t"+str(thrust_parameter_y)+"\t"+str(thrust_parameter_rotation)+"\t"+str(agreement_rotation_norm)+"\t"+str(thrust_parameter_y_norm)+"\n")
                else:
                    f.write(str(x)+"\t"+str(y)+"\t"+str(th)+"\t"+str(time)+"\t"+str(endx)+"\t"+str(endy)+"\t"+str(endth)+"\t"+str(endt)+"\t"+str(0)+"\t"+str(agreement_x)+"\t"+str(agreement_y)+"\t"+str(agreement_rotation)+"\t"+str(thrust_parameter_x)+"\t"+str(thrust_parameter_y)+"\t"+str(thrust_parameter_rotation)+"\t"+str(agreement_rotation_norm)+"\t"+str(thrust_parameter_y_norm)+"\n")
            else:
                if (env.get_success()):
                    f.write(str(x)+"\t"+str(y)+"\t"+str(th)+"\t"+str(time)+"\t"+str(endx)+"\t"+str(endy)+"\t"+str(endth)+"\t"+str(endt)+"\t"+str(env.get_success())+"\n")
                else:
                    f.write(str(x)+"\t"+str(y)+"\t"+str(th)+"\t"+str(time)+"\t"+str(endx)+"\t"+str(endy)+"\t"+str(endth)+"\t"+str(endt)+"\t"+str(0)+"\n")
    f.close()

     
if __name__ == '__main__':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    flow_index=3
    sensor_ls=[-0.25,-0.75,-0.5,0.0,0.25,0.5,0.75]
    angularvelocitys=[5,4,3,2,1,0,-5,-4,-3,-2,-1]
    sensor_velocitys=[0.25]
    # for i,angularvelocity in enumerate(angularvelocitys):
    #     if i == rank:
    #         for sensorl in sensor_ls:
    #             for sensor_velocity in sensor_velocitys:
    #                 main(sensor_l=sensorl,sensor_velocity=sensor_velocity,angularvelocity=angularvelocity,flow_index=flow_index,sensor_index=0)
    # flow_index=4
    # main(sensor_l=-0.25,sensor_velocity=0.25,angularvelocity=3,flow_index=flow_index)
    # if rank == 0:
    #     main(sensor_l=-0.25,sensor_velocity=0.25,angularvelocity=3,flow_index=3,sensor_index=0)
    # if rank == 1:
    #     main(sensor_l=-0.25,sensor_velocity=0.25,angularvelocity=-3,flow_index=3,sensor_index=0)
    if rank == 0:
        main(sensor_l=0.25,sensor_velocity=0.25,angularvelocity=3,flow_index=0,sensor_index=0,excitatory=True)
    if rank == 1:
        main(sensor_l=0.25,sensor_velocity=0.25,angularvelocity=3,flow_index=0,sensor_index=0,excitatory=False)
       
    