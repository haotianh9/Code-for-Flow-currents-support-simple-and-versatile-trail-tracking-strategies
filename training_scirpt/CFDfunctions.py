#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import os
import numpy as np
# import pandas as pd
from tqdm import tqdm
import re

import math

class CFDfunctions():
    def __init__(self,init_time,time_span,source_path,read_interval ,level_limit,viz_dir='viz_cylinder2d/',load_velocity=True,load_vorticity=True,load_pressure=True,multiprocess=False,load_concentration=False,concentration_path="concentration_D=0.01/"):
        self.load_velocity=load_velocity
        self.load_vorticity=load_vorticity
        self.load_pressure=load_pressure
        self.load_concentration=load_concentration
        filename=os.path.join(source_path+viz_dir+"dumps.visit")
        indexList=[]
        f = open(filename)               
        line = f.readline()            
        while line : 
            indexList.append(re.findall('\d+', line)[0])
            line = f.readline()
        f.close()
            
        name = viz_dir+"visit_dump."+indexList[0]+"/summary.samrai"
        filename = os.path.join(source_path + name)
        ff0 =  h5py.File(filename,'r')
        name = viz_dir+"visit_dump."+indexList[1]+"/summary.samrai"
        filename = os.path.join(source_path + name)
        ff1 =  h5py.File(filename,'r')
        self.frame_rate = round(1/(ff1['BASIC_INFO']['time'][0]-ff0['BASIC_INFO']['time'][0]))
        initframe = int(init_time*self.frame_rate)
        self.initframe=initframe
        self.frame_rate = self.frame_rate/read_interval
        # print(frame_rate)
        # print(read_interval)
        # print(time_span)
        self.time_span=time_span
        self.tn = int(time_span*self.frame_rate)+1
        
        if self.load_concentration:
            self.concentration=[]
            concentration_path=os.path.join(source_path+concentration_path)
            self.con_xmax=16.00
            self.con_xmin=-1.0
            self.con_ymax=3.50
            self.con_ymin=-3.50
            self.con_dx=0.015
            self.con_dy=0.015

            self.timestep_concentration=0.005
            self.interval_concentration=int(1/(self.timestep_concentration)/self.frame_rate )
            # print((self.timestep_concentration*self.interval_concentration,self.tn*self.interval_concentration-1,self.interval_concentration))
            for i in range(initframe*self.interval_concentration,self.tn*self.interval_concentration-1,self.interval_concentration):
                name='concentration_'+str(i).zfill(5)+'.npy'
                filename=os.path.join(concentration_path + name)
                con=np.load(filename)
                self.concentration.append(con)


        # with np.load(os.path.join(source_path,"Info.npz")) as data:
        #     init_time = data["init_time"]
        #     self.time_span = min(data["time_span"],time_span)
        #     self.frame_rate = data["frame_rate"]
        # self.tn = int(self.time_span*self.frame_rate)+1
        self.UUU = [[] for i in range(self.tn)]
        self.VVV = [[] for i in range(self.tn)]
        self.OOO = [[] for i in range(self.tn)]
        self.PPP = [[] for i in range(self.tn)]
        # IMIN = [[] for i in range(self.tn)]
        # IMAX = [[] for i in range(self.tn)]
        # JMIN = [[] for i in range(self.tn)]
        # JMAX = [[] for i in range(self.tn)]
        self.XMIN = [[] for i in range(self.tn)]
        self.XMAX = [[] for i in range(self.tn)]
        self.YMIN = [[] for i in range(self.tn)]
        self.YMAX = [[] for i in range(self.tn)]

        # print(initframe,self.tn)
        for n in tqdm(range(initframe,self.tn)):
            index = indexList[n*read_interval]
            # print(index)
            
            if not multiprocess:
                name=viz_dir+"/visit_dump."+index+"/processor_cluster.00000.samrai"
                filename=os.path.join(source_path + name)
                f1 =  h5py.File(filename,'r')
                name =viz_dir+"/visit_dump."+index+"/summary.samrai"
                filename = os.path.join(source_path + name)
                f2 =  h5py.File(filename,'r')
                time = (f2['BASIC_INFO']['time'][0])
                MAX_LEVELS = (f2['BASIC_INFO']['number_levels'][0])
                dx = np.array(f2['BASIC_INFO']['dx'])
                num_patches = np.array(f2['BASIC_INFO']['number_patches_at_level'],dtype = np.int32)
                x_min = np.array(f2['BASIC_INFO']['XLO'])
                patch_tot = 0
                nLevel = min(MAX_LEVELS,level_limit)

                for level in (range(min(MAX_LEVELS,level_limit))):
                    for patch in range(num_patches[level]):
                        patch_extents = f2['extents']['patch_extents'][patch_tot]
                        
                        xn, yn = (patch_extents[1]-patch_extents[0])[0:2]+1
                        
                        if self.load_pressure:
                            if 'P' in f1['processor.00000'][f'level.{level:05n}'][f'patch.{patch:05n}'] :
                                pressure=np.array(f1['processor.00000'][f'level.{level:05n}'][f'patch.{patch:05n}']['P']).reshape(yn,xn)
                                self.PPP[n].append(pressure)
                            else:
                                print("Warning: no pressure data to read")
                                self.load_pressure=False
                        if self.load_vorticity:
                            if 'Omega' in f1['processor.00000'][f'level.{level:05n}'][f'patch.{patch:05n}'] :
                                omega = np.array(f1['processor.00000'][f'level.{level:05n}'][f'patch.{patch:05n}']['Omega']).reshape(yn,xn)
                                self.OOO[n].append(omega)
                            else:
                                print("Warning: no vorticity data to read")
                                self.load_vorticity=False
                        if self.load_velocity:
                            U = np.array(f1['processor.00000'][f'level.{level:05n}'][f'patch.{patch:05n}']['U_x']).reshape(yn,xn)
                            V = np.array(f1['processor.00000'][f'level.{level:05n}'][f'patch.{patch:05n}']['U_y']).reshape(yn,xn)
                            self.UUU[n].append(U)
                            self.VVV[n].append(V)
                            
                        
                        # IMIN[n].append(data["imin"])
                        # JMIN[n].append(data["jmin"])
                        # IMAX[n].append(data["imax"])
                        # JMAX[n].append(data["jmax"])
                        xmin = float(patch_extents[2][0])
                        xmax = float(patch_extents[3][0])
                        ymin = float(patch_extents[2][1])
                        ymax = float(patch_extents[3][1])
                        self.XMIN[n].append(xmin+dx[level][0]/2)
                        self.XMAX[n].append(xmax-dx[level][0]/2)
                        self.YMIN[n].append(ymin+dx[level][1]/2)
                        self.YMAX[n].append(ymax-dx[level][1]/2)

                        # np.savez(filename,U=U,V=V,Omega=omega,
                        #         imin = int(patch_extents[0][0]),
                        #         jmin = int(patch_extents[0][1]),
                        #         imax = int(patch_extents[1][0]),
                        #         jmax = int(patch_extents[1][1]),
                        #         xmin = float(patch_extents[2][0]),
                        #         xmax = float(patch_extents[3][0]),
                        #         ymin = float(patch_extents[2][1]),
                        #         ymax = float(patch_extents[3][1]))
                        patch_tot += 1
            else:
                file_list=[]
                key_list=[]
                
                for i in range(multiprocess):
                    name=viz_dir+"/visit_dump."+index+"/processor_cluster.{}.samrai".format(str(i).zfill(5))
                    filename=os.path.join(source_path + name)
                    f10 =  h5py.File(filename,'r')
                    file_list.append(f10)
                    key_list.append('processor.{}'.format(str(i).zfill(5)))
                name =viz_dir+"/visit_dump."+index+"/summary.samrai"
                filename = os.path.join(source_path + name)
                f2 =  h5py.File(filename,'r')
                time = (f2['BASIC_INFO']['time'][0])
                MAX_LEVELS = (f2['BASIC_INFO']['number_levels'][0])
                dx = np.array(f2['BASIC_INFO']['dx'])
                num_patches = np.array(f2['BASIC_INFO']['number_patches_at_level'],dtype = np.int32)
                x_min = np.array(f2['BASIC_INFO']['XLO'])
                patch_tot = 0
                nLevel = min(MAX_LEVELS,level_limit)

                for level in (range(min(MAX_LEVELS,level_limit))):
                    for patch in range(num_patches[level]):
                        patch_extents = f2['extents']['patch_extents'][patch_tot]
                        xn, yn = (patch_extents[1]-patch_extents[0])[0:2]+1

                        if self.load_velocity:
                            finished=False
                            for i,f1 in enumerate(file_list):
                                try:
                                    U = np.array(f1[key_list[i]][f'level.{level:05n}'][f'patch.{patch:05n}']['U_x']).reshape(yn,xn)
                                    V = np.array(f1[key_list[i]][f'level.{level:05n}'][f'patch.{patch:05n}']['U_y']).reshape(yn,xn)
                                    self.UUU[n].append(U)
                                    self.VVV[n].append(V)
                                    finished=True
                                    break
                                except KeyError:
                                    continue
                            if finished == False:
                                print("Warning: no velocity data to read")
                                self.load_velocity=False
                        if self.load_pressure:
                            finished=False
                            for i,f1 in enumerate(file_list):
                                try:
                                    pressure=np.array(f1[key_list[i]][f'level.{level:05n}'][f'patch.{patch:05n}']['P']).reshape(yn,xn)
                                    self.PPP[n].append(pressure)
                                    finished=True
                                    break
                                except KeyError:
                                    continue
                            if finished == False:
                                print("Warning: no pressure data to read")
                                self.load_pressure=False
                        if self.load_vorticity:
                            finished=False
                            for i,f1 in enumerate(file_list):
                                try:
                                    omega = np.array(f1[key_list[i]][f'level.{level:05n}'][f'patch.{patch:05n}']['Omega']).reshape(yn,xn)
                                    self.OOO[n].append(omega)
                                    finished=True
                                    break
                                except KeyError:
                                    continue
                            if finished == False:
                                print("Warning: no vorticity data to read")
                                self.load_vorticity=False
                        
                            
                        
                        # IMIN[n].append(data["imin"])
                        # JMIN[n].append(data["jmin"])
                        # IMAX[n].append(data["imax"])
                        # JMAX[n].append(data["jmax"])
                        xmin = float(patch_extents[2][0])
                        xmax = float(patch_extents[3][0])
                        ymin = float(patch_extents[2][1])
                        ymax = float(patch_extents[3][1])
                        self.XMIN[n].append(xmin+dx[level][0]/2)
                        self.XMAX[n].append(xmax-dx[level][0]/2)
                        self.YMIN[n].append(ymin+dx[level][1]/2)
                        self.YMAX[n].append(ymax-dx[level][1]/2)

                        # np.savez(filename,U=U,V=V,Omega=omega,
                        #         imin = int(patch_extents[0][0]),
                        #         jmin = int(patch_extents[0][1]),
                        #         imax = int(patch_extents[1][0]),
                        #         jmax = int(patch_extents[1][1]),
                        #         xmin = float(patch_extents[2][0]),
                        #         xmax = float(patch_extents[3][0]),
                        #         ymin = float(patch_extents[2][1]),
                        #         ymax = float(patch_extents[3][1]))
                        patch_tot += 1
            if self.load_velocity:
                self.UUU[n].reverse()
                self.VVV[n].reverse()
            if self.load_vorticity:
                self.OOO[n].reverse()
            if self.load_pressure:
                self.PPP[n].reverse()
            self.XMIN[n].reverse()
            self.XMAX[n].reverse()
            self.YMIN[n].reverse()
            self.YMAX[n].reverse()



    def adapt_time_interp(self,time = 0.634,posX = -0.883,posY = 0.43,interp_velocity=True,interp_vorticity=False,interp_pressure=False,interp_concentration=False):
        # print(posX,posY,time)
        if interp_velocity and not self.load_velocity:
            print("Warning: no velocity data is given")
            interp_velocity=False
        if interp_vorticity and not self.load_vorticity:
            print("Warning: no vorticity data is given")
            interp_vorticity=False
        if interp_pressure and not self.load_pressure:
            print("Warning: no pressure data is given")
            interp_pressure=False
        if interp_concentration and not self.load_concentration:
            print("Warning: no concentration data is given")
            interp_concentration=False
        
        if (interp_velocity or interp_vorticity or interp_pressure or interp_concentration) == False:
            print("你在逗我吗？")
            raise NotImplementedError
        
        flag= [interp_velocity,interp_vorticity,interp_pressure,interp_concentration]
        # if interp_velocity:
        #     self.tn = len(self.UUU)
        # elif interp_vorticity:
        #     self.tn = len(self.OOO)
        # elif interp_pressure:
        #     self.tn = len(self.PPP)
        
        frame = time*self.frame_rate
        
        if frame>self.tn-1:
            # print(frame)
            frameDown = self.tn-1
            frameUp = frameDown
        else:
            frameUp = np.int64(np.ceil(frame))
            frameDown = np.int64(np.floor(frame))
        weightDown = (frameUp-frame)
        weightUp = (frame-frameDown)
        """velocity for frameDown"""
        # print(posX,posY,frameDown,flag)
        Down = self.adapt_space_interp(posX,posY,frameDown,flag)
        # #########################################################
        if frameDown != frameUp:
            """velocity for frameUp"""
            Up = self.adapt_space_interp(posX,posY,frameUp,flag)
            # UUp, VUp = read_vel(posX, posY, frameUp)
            interp = Up*weightUp+Down*weightDown
        else:
            interp = Down+0

        return interp


    def adapt_space_interp(self,posX,posY,frame,flag):
        # print(frame)
        # print(len(self.UUU))
        
        if  (flag[0] or flag[1] or flag[2]):
            if flag[0]:
                num_patch=len(self.UUU[frame])-1
            elif flag[1]:
                num_patch=len(self.OOO[frame])-1
            elif flag[2]:
                num_patch=len(self.PPP[frame])-1
            # print(posX,posY)
            select=False
            for i in range(num_patch):
                # print(i,self.XMIN[frame][i],self.XMAX[frame][i],self.YMIN[frame][i],self.YMAX[frame][i])
                if posX>self.XMIN[frame][i] and posX<self.XMAX[frame][i] and posY>self.YMIN[frame][i] and posY<self.YMAX[frame][i]:
                    # print('patch',i)
                    # print('left',XMIN[i])
                    # print('right',XMAX[i])
                    # print('bottom',YMIN[i])
                    # print('top',YMAX[i])
                    # print('level',l[i])
                    # print("flag: ",flag)
                    if flag[0]:
                        Uf = self.UUU[frame][i]
                        Vf = self.VVV[frame][i]
                    if flag[1]:
                        # print("aaaa")
                        Of = self.OOO[frame][i]
                    if flag[2]:
                        Pf =self.PPP[frame][i]
                    # dx, dy = dx_list[l[i]][0:2]
                    if flag[0]:
                    
                        nx=(Uf.shape[1]-1)
                        ny=(Uf.shape[0]-1)
                    elif flag[1]:
                    
                        nx=(Of.shape[1]-1)
                        ny=(Of.shape[0]-1)
                    elif flag[2]:
                    
                        nx=(Pf.shape[1]-1)
                        ny=(Pf.shape[0]-1)
                    dx = (self.XMAX[frame][i]-self.XMIN[frame][i])/nx
                    dy = (self.YMAX[frame][i]-self.YMIN[frame][i])/ny
                    indexX = math.ceil((posX-self.XMIN[frame][i])/dx)
                    rx = (posX-self.XMIN[frame][i])%dx
                    indexY = math.ceil((posY-self.YMIN[frame][i])/dy)
                    ry = (posY-self.YMIN[frame][i])%dy
                    # xf = np.linspace(XMIN[i],XMAX[i],Uf.shape[1])
                    # yf = np.linspace(YMIN[i],YMAX[i],Uf.shape[0])
                    select=True
                    break
            # if not found in higher level data, then use the coarsest level data (in lcuding some thing out of boundary)
            if select==False:
                i=num_patch
                # print(i)
                if flag[0]:
                    # print(len(self.UUU[frame]))
                    Uf = self.UUU[frame][i]
                    Vf = self.VVV[frame][i]
                    nx=(Uf.shape[1]-1)
                    ny=(Uf.shape[0]-1)
                if flag[1]:
                    Of = self.OOO[frame][i]
                    nx=(Of.shape[1]-1)
                    ny=(Of.shape[0]-1)
                if flag[2]:
                    Pf =self.PPP[frame][i]
                    nx=(Pf.shape[1]-1)
                    ny=(Pf.shape[0]-1)
            
            # dx, dy = dx_list[l[i]][0:2]
    
            dx = (self.XMAX[frame][i]-self.XMIN[frame][i])/nx
            dy = (self.YMAX[frame][i]-self.YMIN[frame][i])/ny
        
            indexX = math.ceil((posX-self.XMIN[frame][i])/dx)
            indexY = math.ceil((posY-self.YMIN[frame][i])/dy)
            rx = (posX-self.XMIN[frame][i])%dx
            ry = (posY-self.YMIN[frame][i])%dy
            # deal with out of boundary
            if indexX <= 1:
                indexX=1
            if indexX >= nx:
                indexX= nx
            if indexY <= 1:
                indexY=1
            if indexY >= ny:
                indexY= ny
            while (rx < 0) :
                rx=rx+dx
            while (rx > dx) :
                rx=rx-dx
            while (ry < 0) :
                ry=ry+dy
            while (ry > dy) :
                ry=ry-dy
            xa = np.array([[1-rx/dx, rx/dx]])
            ya = np.array([[1-ry/dy] , [ry/dy]])


        if flag[3]:
            # print(len(self.concentration))
            # print(frame)
            Cf=self.concentration[frame-self.initframe]
            con_indexX = math.ceil((posX-self.con_xmin)/self.con_dx)
            con_indexY = math.ceil((posY-self.con_ymin)/self.con_dy)
            con_rx = (posX-self.con_xmin)%self.con_dx
            con_ry = (posY-self.con_ymin)%self.con_dy
            if con_indexX <= 1:
                con_indexX=1
            if con_indexX >= Cf.shape[1]-1:
                con_indexX= Cf.shape[1]-1
            if con_indexY <= 1:
                con_indexY=1
            if con_indexY >= Cf.shape[0]-1:
                con_indexY= Cf.shape[0]-1
            while (con_rx < 0) :
                con_rx=con_rx+self.con_dx
            while (con_rx > self.con_dx) :
                con_rx=con_rx-self.con_dx
            while (con_ry < 0) :
                con_ry=con_ry+self.con_dy
            while (con_ry > self.con_dy) :
                con_ry=con_ry-self.con_dy
            con_xa = np.array([[1-con_rx/self.con_dx, con_rx/self.con_dx]])
            con_ya = np.array([[1-con_ry/self.con_dy] , [con_ry/self.con_dy]])
        # indexX = np.searchsorted(xf,posX)
        # indexY = np.searchsorted(yf,posY)
        # print(indexX,indexY)
        # print(posX,posY)
        # dx = xf[1]-xf[0]
        # dy = yf[1]-yf[0]
        # xa = np.array([[xf[indexX] - posX, posX - xf[indexX-1]]])/dx
        # ya = np.array([[yf[indexY] - posY] , [posY - yf[indexY-1]]])/dy
        



        
        #########################################################
        """interpolate velocity"""
        
        res=[]
        if flag[0]:
            QU = Uf[indexY-1:indexY+1,indexX-1:indexX+1].T
            QV = Vf[indexY-1:indexY+1,indexX-1:indexX+1].T
            UInterp = (xa @ QU @ ya).squeeze()
            VInterp = (xa @ QV @ ya).squeeze()
            res.append(UInterp)
            res.append(VInterp)
        if flag[1]:
            QO = Of[indexY-1:indexY+1,indexX-1:indexX+1].T
            OInterp = (xa @ QO @ ya).squeeze()
            res.append(OInterp)
        if flag[2]:
            QP = Pf[indexY-1:indexY+1,indexX-1:indexX+1].T
            PInterp = (xa @ QP @ ya).squeeze()
            res.append(PInterp)
        if flag[3]:
            QC=Cf[con_indexY-1:con_indexY+1,con_indexX-1:con_indexX+1].T
            CInterp = (con_xa @ QC @ con_ya).squeeze()
            res.append(CInterp)
        return np.array(res)



    def read_image(self,time,rootpath,dump_interval):
        path = rootpath+f"/movie{int(np.floor(time*self.frame_rate/dump_interval)):04n}.png"
        return path
