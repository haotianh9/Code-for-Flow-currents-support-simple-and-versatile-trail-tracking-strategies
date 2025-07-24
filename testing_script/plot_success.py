import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
# St=np.array([0.38,0.40,0.42,0.44,0.46,0.63,0.64,0.65,0.66,0.67,0.71,0.73,0.75,0.77,0.79])



angular_velocity=[-3.0]
#sensor_l=[-3.45,-2.3,-1.15,-0.9,-0.5,0.0,1.15,2.3,3.45]
sensor_l=[-0.25]
# sensor_l=[0.5]
#time111=[0.0,0.2,0.4,0.6,0.8]
time111=[0.0]

sensor="delta_umag"
oasr=np.zeros((len(sensor_l),len(angular_velocity)))
qua_0=np.zeros((len(sensor_l),len(angular_velocity)))
qua_25=np.zeros((len(sensor_l),len(angular_velocity)))
qua_75=np.zeros((len(sensor_l),len(angular_velocity)))
qua_50=np.zeros((len(sensor_l),len(angular_velocity)))
qua_100=np.zeros((len(sensor_l),len(angular_velocity)))

length_th=20
    
test_xmin=11.0
test_xmax= 2
test_ymin=-1.6
test_ymax=1.6
test_nx=45
test_ny=17
test_nth=20
test_x=np.linspace(test_xmin,test_xmin,test_nx)
test_y=np.linspace(test_ymin,test_ymin,test_ny)
[test_x,test_y]=np.meshgrid(test_x,test_y)
for iii,sensorl in enumerate(sensor_l):
    for jjj,angularvelocity in enumerate(angular_velocity):
        success_s=[]
        print(sensorl,angularvelocity)
        for kkk,time in enumerate(time111):
            filename="pitching_airfoil_Re=5000_freq=1.25_amp=0.2_delta_umag_sensor_l_{}_sensor_velocity_0.250_angular_velocity_{}_time=0.0.txt".format(format(sensorl,'.3f'),format(angularvelocity,'.3f'))

            data=np.loadtxt('./'+filename)

            xxx=data[:,0]
            yyy=data[:,1]
            theta=data[:,2]
            time=data[:,3]
            end_x=data[:,4]
            end_y=data[:,5]
            end_th=data[:,6]
            end_t=data[:,7]
            # success=data[:,8]/2
            success=np.logical_and(data[:,8]==2 , np.abs(end_y) <=0.3)
            # for i in range(len(end_y)):
            #     if end_x[i] <= 1.5 and np.abs(end_y) <= 0.75:
            #         success[i]=1
            #     else:
            #         success[i]=0
            success_s.append(success)
        success_s=np.mean(np.array(success_s),axis=0)


        x1=np.zeros((int(len(xxx)/length_th),1))
        y1=np.zeros((int(len(xxx)/length_th),1))
        successrate=np.zeros((int(len(xxx)/length_th),1))
        successrate_2=np.zeros(np.shape(test_x))
        avg_angle=np.zeros((int(len(xxx)/length_th),2))
        for i in range(int(len(xxx)/length_th)):
            x1[i]=xxx[i*length_th]
            y1[i]=yyy[i*length_th]
            successrate[i]=0
            avg_angle[i,0]=0
            avg_angle[i,1]=0
            for j in range(length_th):
                # if abs(end_y[i*length1+j]) <= 0.5:
                #     success[i*length1+j]=1
                # else:
                #     success[i*length1+j]=0
                    
                successrate[i]=successrate[i]+success_s[i*length_th+j]
                if success_s[i*length_th+j] > 0.5:
                    avg_angle[i,0]=avg_angle[i,0]+np.cos(theta[i*length_th+j])
                    avg_angle[i,1]=avg_angle[i,1]+np.sin(theta[i*length_th+j])
            avg_angle[i,0]=avg_angle[i,0]/successrate[i]
            avg_angle[i,1]=avg_angle[i,1]/successrate[i]
            # if successrate[i]== 0:
            #     avg_angle[i,:]=0
            successrate[i]=successrate[i]/length_th
        ep=0
        for p in range(test_nx):
            for j in range(test_ny):
                successrate_2[j,p]=0

                for k in range(length_th):
                    successrate_2[j,p]=successrate_2[j,p]+success_s[ep]
                    ep+=1

                successrate_2[j,p]=successrate_2[j,p]/length_th
                # print(successrate[j,p])

        print(np.sum(success_s)/np.size(success_s))
        oasr[iii,jjj]=np.sum(success_s)/np.size(success_s)
        qua_0[iii,jjj]=np.percentile(successrate,0)
        qua_25[iii,jjj]=np.percentile(successrate,25)
        qua_50[iii,jjj]=np.percentile(successrate,50)
        qua_75[iii,jjj]=np.percentile(successrate,75)
        qua_100[iii,jjj]=np.percentile(successrate,100)

        plt.figure(figsize=(8, 3))
        matplotlib.rcParams['xtick.direction'] = 'in' 
        matplotlib.rcParams['ytick.direction'] = 'in' 
        bwith = 0.25 #边框宽度设置为2
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.tick_params(which='major',width=0.1)
        plt.quiver(x1,y1,avg_angle[:,0],avg_angle[:,1],color="black",scale=54,width=0.002)
        
        # plt.clim(0,1)
        # plt.colorbar()

        plt.axis('equal')
        plt.xlim([11,0])
        plt.ylim([-2,2])
        plt.savefig('success_quiver_flowtaxis_no_color_{}_{}_{}.eps'.format(sensor,format(sensorl,'.3f'),format(angularvelocity,'.3f')), format='eps')
        # plt.savefig('success_quiver_no_color_{}_{}_{}.eps'.format(sensor,format(sensorl,'.3f'),format(angularvelocity,'.3f')), format='eps')
        # plt.show()
        plt.close()


        plt.figure(figsize=(8, 3))
        matplotlib.rcParams['xtick.direction'] = 'in' 
        matplotlib.rcParams['ytick.direction'] = 'in' 
        bwith = 0.25 #边框宽度设置为2
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.tick_params(which='major',width=0.1)
        
        map=np.loadtxt("./colormap.txt")
        
        cmap = matplotlib.colors.ListedColormap(map, name='myColorMap', N=map.shape[0])
        plt.imshow(successrate_2,vmin=0.0, vmax=1.0, cmap=cmap,origin='lower', 
            extent=[test_xmin, test_xmax, test_ymin, test_ymax],interpolation="bicubic")
        # plt.imshow(avg_angle,vmin=np.pi/2, vmax=np.pi*3/2, cmap=plt.cm.RdBu,origin='lower', 
        #         extent=[test_xmin, test_xmax, test_ymin, test_ymax],interpolation="bicubic")
        # plt.imshow(std_angle,vmin=0.0, vmax=2.0, cmap=plt.cm.plasma_r,origin='lower', 
        #         extent=[test_xmin, test_xmax, test_ymin, test_ymax],interpolation="bicubic")

        # plt.clim(0,1)
        plt.colorbar()
        plt.axis('equal')
        plt.xlim([11,0])
        plt.ylim([-2,2])
        # plt.ylim([-2,2])
        # plt.xlim([-0.5,7])
        plt.savefig('success_plot_flowtaxis_{}_{}_{}_2.eps'.format(sensor,sensorl,angularvelocity), format='eps')
        # plt.savefig('success_plot_{}_{}_{}_2.eps'.format(sensor,sensorl,angularvelocity), format='eps')
        # plt.show()
        plt.close()







b=[]
for iii,sensorl in enumerate(sensor_l):
    for jjj,angularvelocity in enumerate(angular_velocity):
        a=[sensorl,angularvelocity, oasr[iii,jjj], qua_0[iii,jjj], qua_25[iii,jjj],qua_50[iii,jjj],qua_75[iii,jjj],qua_100[iii,jjj]]
        b.append(a)
b=np.array(b)
# np.savetxt("79_umag_search.txt",b)
# np.savetxt("success_rate.txt",(St,St_successrate))
# np.savetxt("42_qua.txt",(St,St_qua_0,St_qua_25,St_qua_50,St_qua_75,St_qua_100))
# print(np.shape(aaa))

# plt.figure()
# matplotlib.rcParams['xtick.direction'] = 'in' 
# matplotlib.rcParams['ytick.direction'] = 'in' 
# bwith = 0.25 #边框宽度设置为2
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)
# ax.spines['left'].set_linewidth(bwith)
# ax.spines['top'].set_linewidth(bwith)
# ax.spines['right'].set_linewidth(bwith)
# plt.tick_params(which='major',width=0.1)
# plt.plot(St,St_successrate,'k')
# plt.errorbar(St,St_successrate,St_sr_std,fmt='o',ms=5,ecolor='k',color='k',elinewidth=0.5,capsize=0.0)
# # plt.errorbar(St,St_successrate,St_sr_std,ecolor='k',color='k',elinewidth=2)
# plt.ylim([0,1])
# plt.xticks(St)
# plt.show()