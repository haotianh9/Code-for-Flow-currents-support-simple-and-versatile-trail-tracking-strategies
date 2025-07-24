import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from flowtaxis import FlowtaxisEnv
import numpy as np
import logging
import os
from PPO_continuous import PPO, Memory
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录,创建目录操作函数
        '''
        os.mkdir(path)与os.makedirs(path)的区别是,当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录
        '''
        #此处路径最好使用utf-8解码，否则在磁盘中可能会出现乱码的情况
        os.makedirs(path) 
        print (path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
        return False

def main(num,flow_index=0,sensor_index=0,sensor_l=0.25,sensor_velocity=0.25):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    flow=["freq=1.0_amp=0.24"]
    #######logging configuration#####
    sensor=["delta_umag_ht","delta_ub1_ht","delta_ub2_ht","delta_pressure_ht","delta_vor_b2_ht","delta_concentration_ht","delta_umag","delta_ub1","delta_ub2","delta_pressure","delta_vor_b2","delta_concentration"]
    output_dir='./training_5_two_obs_{}_{}_{}_{}_{}/'.format(flow[flow_index],sensor[sensor_index],str(sensor_l),str(sensor_velocity),num)
    mkdir(output_dir)
    if os.path.exists(output_dir+'train.log'):
        os.remove(output_dir+'train.log')
    logger = logging.getLogger('trainlog')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(output_dir+'train.log')
    file_handler.filemode='w'
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    ############
    ############## Hyperparameters ##############
    env_name = "Flowtaxis-v0"
    
    
  
    render = False              #Ture for showing video
    solved_reward = 3000000000000         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 1900        # max timesteps in one episode
    
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.4            # constant std for action distribution (Multivariate Normal)
    reward_threshold=20
    std_threshold=0.1
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.999                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None
    logger.info('Basic information and hyperparameters')
    logger.info('env_name: {}'.format(env_name))
    logger.info('sensor: {}'.format(sensor[sensor_index]))
    logger.info('flow: {}'.format(flow[flow_index]))

    logger.info('render: {}'.format(render))

    logger.info('log_interval: {}'.format(log_interval))
    logger.info('max_episodes: {}'.format(max_episodes))
    logger.info('max_timesteps: {}'.format(max_timesteps))
    logger.info('update_timestep: {}'.format(update_timestep))
    logger.info('action_std: {}'.format(action_std))
    logger.info('K_epochs: {}'.format(K_epochs))
    logger.info('eps_clip: {}'.format(eps_clip))
    logger.info('lr: {}'.format(lr))
    logger.info('betas: {}'.format(betas))
    #############################################
    
    # creating environment
    env = FlowtaxisEnv(SensorMode=sensor[sensor_index],FlowFiled=flow[flow_index],sensor_l=sensor_l,MAX_ANGULARVELOCITY=5,velocity=sensor_velocity)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps=max_timesteps)
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    # print(lr,betas)
    # ppo.policy_old.load_state_dict(torch.load('./PPO_continuous_Flowtaxis-v0_delta_umag_head_and_tail_42_190000.pth',map_location=device))
    # ppo.policy.load_state_dict(torch.load('./PPO_continuous_Flowtaxis-v0_delta_umag_head_and_tail_42_190000.pth',map_location=device))
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    running_reward_2=0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            
            running_reward += reward
            running_reward_2 += reward
            if render:
                env.render()
            if done:
                break
        if time_step > update_timestep:
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     logger.info("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}_2.pth'.format(env_name))
        #     break
        
        # save every 500 episodes
        
        if i_episode % 100 == 0:
            torch.save(ppo.policy.state_dict(), output_dir+'PPO_continuous_{}_{}_{}_{}.pth'.format(env_name,sensor[sensor_index],flow[flow_index],i_episode))
            if  running_reward_2 >= 100*reward_threshold and action_std > std_threshold:
                action_std -= 0.01
                reward_threshold += 0.5
                ppo.policy_old.reset_action_std(action_std)
                ppo.policy.reset_action_std(action_std)
            if  running_reward_2 < 0 :
                action_std =0.4
                ppo.policy_old.reset_action_std(action_std)
                ppo.policy.reset_action_std(action_std)
            print("Episode: {} \t action_std: {}".format(i_episode,action_std))
            running_reward_2=0
        


        # # logging
        # if i_episode % log_interval == 0:
        #     avg_length = float(avg_length / log_interval)
        #     running_reward = float((running_reward / log_interval))
        #     logger.info('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        #     running_reward = 0
        #     avg_length = 0
        avg_length = float(avg_length )
        running_reward = float(running_reward )
        logger.info('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
        # if i_episode % log_interval == 0:
        #     avg_length = (avg_length / log_interval)
        #     running_reward = ((running_reward / log_interval))
        #     print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        #     f = open("learningrate_new_{}_{}.csv".format(sensor, sim_num), "a")
        #     f.write('Episode {} \t avg length: {} \t reward: {}\n'.format(i_episode, avg_length, running_reward))
        #     f.close()
        #     running_reward = 0
        #     avg_length = 0
            
if __name__ == '__main__':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for i in range(19):
        if i % 6 == rank:
            main(sensor_l=-0.25,num=i)


       
    
