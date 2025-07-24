
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
# import h5py
# import os
import matplotlib.pyplot as plt
from scipy import integrate
from CFDfunctions import CFDfunctions

class FlowtaxisEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, FlowFiled='freq=1.0_amp=0.24', SensorMode="delta_umag",sensor_l=0.25,Mode="train",MAX_ANGULARVELOCITY=1,decision_timestep=0.05,velocity=0.25,flow_freq=1.0):
        self.flow=FlowFiled
        
        path="/data/haotian/"+FlowFiled+"/"
        self.freq=flow_freq
        self.amp=0.160690653
        self.period=1.0/self.freq



        
        self.intial_time=19
        time_span=self.intial_time+self.period+0.2
        level_limit=4
        self.sensor = SensorMode
        if  self.sensor.casefold() == "delta_umag".casefold():
            high = np.array([1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=True,load_pressure=False,load_vorticity=False)
        elif self.sensor.casefold() == "delta_ub1".casefold():
            high = np.array([1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=True,load_pressure=False,load_vorticity=False)
        elif self.sensor.casefold() == "delta_ub2".casefold():
            high = np.array([1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=True,load_pressure=False,load_vorticity=False)
        elif self.sensor.casefold() == "delta_pressure".casefold():
            high = np.array([1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=False,load_pressure=True,load_vorticity=False)
        elif self.sensor.casefold() == "delta_vor_b2".casefold():
            high = np.array([1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=False,load_pressure=False,load_vorticity=True)
        elif self.sensor.casefold() == "delta_concentration".casefold():
            high = np.array([1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=False,load_pressure=False,load_vorticity=False,load_concentration=True)
        elif  self.sensor.casefold() == "delta_umag_ht".casefold():
            high = np.array([1.0,1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=True,load_pressure=False,load_vorticity=False)
        elif self.sensor.casefold() == "delta_ub1_ht".casefold():
            high = np.array([1.0,1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=True,load_pressure=False,load_vorticity=False)
        elif self.sensor.casefold() == "delta_ub2_ht".casefold():
            high = np.array([1.0,1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=True,load_pressure=False,load_vorticity=False)
        elif self.sensor.casefold() == "delta_pressure_ht".casefold():
            high = np.array([1.0,1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=False,load_pressure=True,load_vorticity=False)
        elif self.sensor.casefold() == "delta_vor_b2_ht".casefold():
            high = np.array([1.0,1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=False,load_pressure=False,load_vorticity=True)
        elif self.sensor.casefold() == "delta_concentration_ht".casefold():
            high = np.array([1.0,1.0])
            low = -high
            self.CFD_data=CFDfunctions(init_time=self.intial_time,time_span=time_span,source_path=path,read_interval=1 ,level_limit=level_limit,load_velocity=False,load_pressure=False,load_vorticity=False,load_concentration=True)
        else:
            raise NotImplementedError
        # high = np.array([self.max_concentration])
        # low = -high
        # high = np.array([self.max_concentration,self.max_concentration])
        # low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


        

        
       
        self.decision_timestep=decision_timestep
        # self.framestep=int(self.decision_timestep/self.CFD_timestep)
        self.max_angularvelocity = MAX_ANGULARVELOCITY
        # self.max_velocity=2

        self.viewer = None
        self.rho=1
        self.thetasave=[]
        self.statesave=[]
        # self.success=[]
        self.episode=0
        self.obs_limit=0.000
        self.outside_action_std=0.0
    

        #sensor related parameter 

        self.V = velocity
        
        
        self.dsensor = 0.01
        self.lsensor = sensor_l
        # self.dt=1/self.periodlength
        # if  self.action.casefold() == "Angular_velocity".casefold():
        #     self.action_space = spaces.Box(low=-self.max_angularvelocity, high=self.max_angularvelocity , shape=(1,),
        #                                dtype=np.float32)
        # elif self.action.casefold() == "Orientation".casefold():
        #     self.action_space = spaces.Box(low=-np.pi, high=np.pi , shape=(1,),
        #                                dtype=np.float32)
        # elif self.action.casefold() == "Angular_and_translational_velocity".casefold():
        #     lb=np.array([0, -self.max_angularvelocity])
        #     hb=np.array([self.max_velocity,self.max_angularvelocity])
        #     self.action_space = spaces.Box(low=lb, high=hb , dtype=np.float32)
        # else:
        #     raise NotImplementedError
        self.action_space = spaces.Box(low=-self.max_angularvelocity, high=self.max_angularvelocity , shape=(1,),
                                       dtype=np.float32)
        
        self.seed()

    def seed(self, seed=None):
        # randomizing
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, thdot):

        x, y, th, time = self.state  # th := theta
        #print(time)        
        x=float(x)
        y=float(y)
        th=float(th)
        time=float(time)
  
        self.last_state=self.state

        V = self.V
        obs=self._get_obs()
        a=0
        for i in range(len(obs)):
            if obs[i] == 0:
                a+=1
        if a ==len(obs):
            thdot=np.random.normal(0, self.outside_action_std)

        thdot = np.clip(thdot, -1, 1)  
       
        thdot=thdot*self.max_angularvelocity
        
        self.last_u = thdot
        # print(step)
        # cost = -reward
        # reward= difference in concentration in b1 direction
        options = {'rtol': 1e-4, 'atol': 1e-8, 'max_step': 1e-2}
        body=np.array([x,y,th])
        sol = integrate.solve_ivp(self._go, (0, self.decision_timestep), body, method='RK45', t_eval=None, dense_output=False,
                                    events=None, vectorized=False, **options)

        body = sol.y[:, -1]
        newx=body[0]
        newy=body[1]
        newth=body[2]

        if newth > np.pi*2:
            newth = -2 * np.pi + newth
        if newth < 0:
            newth = 2 * np.pi + newth
        position = np.array([newx, newy, newth])
        terminal,reward = self._terminal(position)
        # omega,ux,uy,px,py,dudx,dvdy=self.interpolation(x,y,step)
        # pb2_old=-px*np.sin(th)+py*np.cos(th)
        # if step == self.periodlength-1:
        #     step=-1
        # newomega,newux,newuy,newpx,newpy,newdudx,newdvdy=self.interpolation(newx,newy,step+1)
        # pb2_new=-newpx*np.sin(newth)+newpy*np.cos(newth)
        pitching_angle=self.amp*np.sin(2*np.pi*self.freq*(time+self.intial_time))
        TE_x=0.75*np.cos(pitching_angle)
        TE_y=0.75*np.sin(pitching_angle)
        pitching_angle_new=self.amp*np.sin(2*np.pi*self.freq*(time+self.intial_time+ self.decision_timestep))
        TE_x_new=0.75*np.cos(pitching_angle_new)
        TE_y_new=0.75*np.sin(pitching_angle_new)
        reward +=(-np.sqrt((newx-TE_x_new)**2+(newy-TE_y_new)**2)+np.sqrt((x-TE_x)**2+(y-TE_y)**2))
    
        reward=float(reward)
        # print(reward)
        time = time + self.decision_timestep
        # self.thetasave.append(th)
        self.state = np.array([newx, newy, newth, time])
        obs=self._get_obs()
        
        obs_origin=(obs+1)/2*(self.observation_space.high-self.observation_space.low)+(self.observation_space.low)
        self.statesave.append(np.append(self.state,obs_origin))
        return obs, reward, terminal, {}
    def _go(self, t, X):
        x = X[0]
        y = X[1]
        th= X[2]
        dx=self.V*np.cos(th)
        dy=self.V*np.sin(th)
        dth=self.last_u

        # print(type(dth))
        if type(dth) == np.ndarray:
            dth=dth[0]
        # print(dx,dy,dth)
        return np.array([dx,dy,dth])
    def _terminal(self, body):


        if (body[0] <= 0.75)  :
            print('episode terminate: move upstream')
            self.success=2
            return True, float(50)*np.exp(-np.square(body[1]))
 
        if ((body[1] > 6) or (body[1] < -6) or (body[0] < -2.5) or (body[0] > 20.5)):
            print('episode terminate: out of boundary')
            # self.success=False
            return True, float(0)
        return False, float(0)

 
    def reset(self):

        high = np.array([11.0, 1.5, np.pi,1 ])
        low=     np.array([2.0, -1.5, -np.pi,0 ])
        a = self.np_random.uniform(low=low, high=high)
        

        # self.thetasave=[]
        self.state = a
        self.statesave=[]
        
        self.last_u = None
        self.last_state=self.state
        self.success=False


        
        obs=self._get_obs()
        
        obs_origin=(obs+1)/2*(self.observation_space.high-self.observation_space.low)+(self.observation_space.low)
        self.statesave.append(np.append(self.state,obs_origin))
        return obs

   
    def reset_test(self,x,y,th,time,ep):
       

        # self.thetasave=[]
        self.statesave=[]
        self.state = np.array([x,y,th,time])
    
        self.last_u = None
        self.episode=ep
        self.last_state=self.state
        # self.initialstate=np.array(b)
        obs=self._get_obs()
        
        obs_origin=(obs+1)/2*(self.observation_space.high-self.observation_space.low)+(self.observation_space.low)
        self.statesave.append(np.append(self.state,obs_origin))
        return obs
    def _get_obs(self):
        # print("get_observation")
        # get new observation
        # get new observation
        x, y, th, time = self.state
        # print(x,y,th,time)
        # step=int(time)
        # if step >= self.periodlength:
        #     step=step %  self.periodlength
        # # print(step,self.periodlength)
        # if step == self.periodlength:
        #     step=0


        # print(self.state)
        time=time % self.period
        # print(time)
        if  self.sensor.casefold() == "delta_umag".casefold():

            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            ux1,uy1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)


            ux2,uy2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)

            umag_1=np.sqrt(ux1**2+uy1**2)

            umag_2=np.sqrt(ux2**2+uy2**2)

            d1=(umag_1-umag_2)/self.dsensor

            obs=np.array([d1])
        
        elif  self.sensor.casefold() == "delta_ub1".casefold():
            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            ux1,uy1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)


            ux2,uy2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)

            ub1_1=ux1*np.cos(th)+uy1*np.sin(th)
            ub1_2=ux2*np.cos(th)+uy2*np.sin(th)

            d1=(ub1_1-ub1_2)/self.dsensor

            obs=np.array([d1])
        elif  self.sensor.casefold() == "delta_ub2".casefold():
            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            ux1,uy1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)


            ux2,uy2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)

            ub1_1=-ux1*np.sin(th)+uy1*np.cos(th)
            ub1_2=-ux2*np.sin(th)+uy2*np.cos(th)

            d1=(ub1_1-ub1_2)/self.dsensor

            obs=np.array([d1])


        elif  self.sensor.casefold() == "delta_vor_b2".casefold():
            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            vor1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=True,interp_pressure=False,interp_concentration=False)


            vor2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=True,interp_pressure=False,interp_concentration=False)
            d1=(vor1-vor2)/self.dsensor

            obs=np.array([d1])
      
        elif  self.sensor.casefold() == "delta_pressure".casefold():

            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            p1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=True,interp_concentration=False)


            p2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=True,interp_concentration=False)
            d1=(p1-p2)/self.dsensor

            obs=np.array([d1])
        elif  self.sensor.casefold() == "delta_concentration".casefold():

            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            c1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=False,interp_concentration=True)


            c2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=False,interp_concentration=True)
            d1=(c1-c2)/self.dsensor

            obs=np.array([d1])
        elif  self.sensor.casefold() == "delta_umag_ht".casefold():
            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
        
            ux1,uy1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)


            ux2,uy2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)

            umag_1=np.sqrt(ux1**2+uy1**2)

            umag_2=np.sqrt(ux2**2+uy2**2)

            d1=(umag_1-umag_2)/self.dsensor


            sensor_x=x-self.lsensor*np.cos(th)
            sensor_y=y-self.lsensor*np.sin(th)
   
          
            ux1,uy1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)


            ux2,uy2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)

            umag_1=np.sqrt(ux1**2+uy1**2)

            umag_2=np.sqrt(ux2**2+uy2**2)

            d2=(umag_1-umag_2)/self.dsensor


            obs=np.array([d1,d2])
        elif  self.sensor.casefold() == "delta_ub1_ht".casefold():
            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            ux1,uy1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)


            ux2,uy2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)

            ub1_1=ux1*np.cos(th)+uy1*np.sin(th)
            ub1_2=ux2*np.cos(th)+uy2*np.sin(th)

            d1=(ub1_1-ub1_2)/self.dsensor

            sensor_x=x-self.lsensor*np.cos(th)
            sensor_y=y-self.lsensor*np.sin(th)
   
          
            ux1,uy1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)


            ux2,uy2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)

            ub1_1=ux1*np.cos(th)+uy1*np.sin(th)
            ub1_2=ux2*np.cos(th)+uy2*np.sin(th)

            d2=(ub1_1-ub1_2)/self.dsensor
            obs=np.array([d1,d2])
        elif  self.sensor.casefold() == "delta_ub2_ht".casefold():
            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            ux1,uy1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)


            ux2,uy2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)

            ub1_1=-ux1*np.sin(th)+uy1*np.cos(th)
            ub1_2=-ux2*np.sin(th)+uy2*np.cos(th)

            d1=(ub1_1-ub1_2)/self.dsensor

            sensor_x=x-self.lsensor*np.cos(th)
            sensor_y=y-self.lsensor*np.sin(th)
   
          
            ux1,uy1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)


            ux2,uy2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False)

            ub1_1=-ux1*np.sin(th)+uy1*np.cos(th)
            ub1_2=-ux2*np.sin(th)+uy2*np.cos(th)

            d2=(ub1_1-ub1_2)/self.dsensor
            obs=np.array([d1,d2])
        elif  self.sensor.casefold() == "delta_vor_b2_ht".casefold():
            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            vor1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=True,interp_pressure=False,interp_concentration=False)


            vor2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=True,interp_pressure=False,interp_concentration=False)
            d1=(vor1-vor2)/self.dsensor

            sensor_x=x-self.lsensor*np.cos(th)
            sensor_y=y-self.lsensor*np.sin(th)
   
          
            vor1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=True,interp_pressure=False,interp_concentration=False)


            vor2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=True,interp_pressure=False,interp_concentration=False)
            d2=(vor1-vor2)/self.dsensor

            obs=np.array([d1,d2])
      
        elif  self.sensor.casefold() == "delta_pressure_ht".casefold():

            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            p1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=True,interp_concentration=False)


            p2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=True,interp_concentration=False)


            
            d1=(p1-p2)/self.dsensor



            sensor_x=x-self.lsensor*np.cos(th)
            sensor_y=y-self.lsensor*np.sin(th)
   
          
            p1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=True,interp_concentration=False)


            p2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=True,interp_concentration=False)


            
            d2=(p1-p2)/self.dsensor

            obs=np.array([d1,d2])
        elif  self.sensor.casefold() == "delta_concentration_ht".casefold():

            sensor_x=x+self.lsensor*np.cos(th)
            sensor_y=y+self.lsensor*np.sin(th)
   
          
            c1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=False,interp_concentration=True)


            c2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=False,interp_concentration=True)
            d1=(c1-c2)/self.dsensor

            sensor_x=x-self.lsensor*np.cos(th)
            sensor_y=y-self.lsensor*np.sin(th)
   
          
            c1 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x-self.dsensor*np.sin(th)/2,posY =sensor_y +self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=False,interp_concentration=True)


            c2 =  self.CFD_data.adapt_time_interp(time=time+self.intial_time,posX = sensor_x+self.dsensor*np.sin(th)/2,posY =sensor_y -self.dsensor*np.cos(th)/2,interp_velocity=False,interp_vorticity=False,interp_pressure=False,interp_concentration=True)
            d2=(c1-c2)/self.dsensor

            obs=np.array([d1,d2])
        else:
            raise NotImplementedError
       
        # obs=2*(obs-self.observation_space.low)/(self.observation_space.high-self.observation_space.low)+self.observation_space.low
        

        return obs


        
   
    def render(self, mode='human'):

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 1000*(self.ymax-self.ymin)/(self.xmax-self.xmin))
            self.viewer.set_bounds(self.xmin, self.xmax, self.ymin, self.ymax)

 

            source = self.viewer.draw_polygon([(2,-1), (2, 1), (0, 0.5), (0, -0.5)])
            source.set_color(0.7, 0.7, 0.7)
            self.viewer.add_geom(source)
    
        
        x, y, th, time = self.state
        xold,yold,thold,timeold=self.last_state
        self.last_state=self.state
        trajectory = self.viewer.draw_line((xold,yold),
                                           (x, y))
        trajectory.set_color(.9, .1, .1)
  
        self.viewer.add_geom(trajectory)

        l, b, t, r = -self.lsensor/2, -self.dsensor/2, self.dsensor/2, self.lsensor/2
        sensor = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        sensorTrans = rendering.Transform(rotation=th, translation=(x, y))

        sensor.add_attr(sensorTrans)
        sensor.set_color(0, 0, .3)


        xold,yold,thold,timeold=x, y, th, time
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def get_arr(self):
        return self.statesave
    def get_success(self):
        return self.success


def interp(x,y,U1,U2,U3,U4):
    #x,y are in [0,1]
    # U2    U3
    # U1    U4
    U14=x*(U4-U1)+U1
    U23=x*(U3-U2)+U2
    U=y*(U23-U14)+U14
    return U

