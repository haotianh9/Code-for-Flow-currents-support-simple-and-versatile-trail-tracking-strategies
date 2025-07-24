from flowtaxis import FlowtaxisEnv
import numpy as np

import os
import matplotlib.pyplot as plt


def custom_action(obs):
    if obs > 0.0:
        return 1.0
    elif obs < 0.0:
        return -1.0
    else:
        return 0.0


def main(sensor_l=0.25, flow_index=0, sensor_index=0):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    flow = [
        # "flapping_Re=0.1",
        "Re=5000_freq=1.25_amp=0.2",
        "freq=0.5_amp=0.2",
        "Re=5000_freq=0.5_amp=0.2",
        "freq=1.25_amp=0.2",
    ]
    flow_freq = [1.25, 0.5, 0.5, 1.25]
    env_name = "Flowtaxis-v0"
    sensor = [
        "delta_umag",
        "delta_pressure",
        "delta_ub1",
        "delta_ub2",
        "delta_concentration",
        "delta_vor_b1",
        "delta_vor_b2",
        "delta_umag_b1",
        "delta_pressure_b1",
        "delta_concentration_b1",
        "delta_ub1_b1",
        "delta_ub2_b1",
    ]
    max_timesteps = 9000  # max timesteps in one episode

    ICs = [
        # [6.5, 1.5, -np.pi / 3, 0],
        # [6.5, -1.5, np.pi / 2, 0],
        # [6.5, 0.5, -np.pi / 3, 0],
        [6.5, 0.02, 0.05, 0],
        # [6.5, 0.02, 0.04, 0],
        [6.5, 0.02, -0.02, 0],
        [6.5, 0.02, -0.05, 0],
        [6.5, 0.02, np.pi / 3, 0],
        # [6.5, -0.02, 0.05, 0],
        # [6.5, -0.02, 0.04, 0],
        # [6.5, -0.02, -0.02, 0],
        # [6.5, -0.02, -0.05, 0],
        # [6.5, 0.0, 0.05, 0],
        # [6.5, 0.0, 0.04, 0],
        # [6.5, 0.0, -0.02, 0],
        # [6.5, 0.0, -0.05, 0],
        # [6.5, -0.25, -np.pi * 2 / 3, 0],
    ]
    # ICs = [
    #     [9.5, 1.5, -np.pi / 3, 0],
    #     [9.5, -1.5, np.pi / 2, 0],
    #     [9.5, 0.5, -np.pi / 3, 0],
    #     [9.5, 0.02, 0.02, 0],
    #     [9.5, -0.25, -np.pi * 2 / 3, 0],
    # ]
    ntest = len(ICs)
    # ICs=[0]*20
    # ICs=[[4.5,0.5,np.pi/6,0]]
    # ICs=[]
    # for i in range(9):
    #     high=np.array([6.5,1.5,np.pi,0])
    #     low=np.array([6.5,-1.5,-np.pi,0])
    #     ICs.append(np.random.uniform(low,high))
    # Peclet = 100
    env = FlowtaxisEnv(
        FlowFiled=flow[flow_index],
        SensorMode=sensor[sensor_index],
        sensor_l=sensor_l,
        MAX_ANGULARVELOCITY=3,
        velocity=0.25,
        decision_timestep=0.025,
        flow_freq=flow_freq[flow_index],
        SensingLimit=False,
        # LimitThreshold=2.0,
        # Peclet=Peclet,
        return_grad_orientation=False,
    )

    for III in range(ntest):
        obs = env.reset()
        if III < len(ICs):
            IC = ICs[III]
            obs = env.reset_test(IC[0], IC[1], IC[2], IC[3], III)
        # print(III,obs)
        # input()
        total_cost = 0
        for t in range(max_timesteps):
            # Running policy_old:
            action = custom_action(obs)
            # if np.abs(obs) < 0.22530376947534925 * 0.2:
            #     print(obs, action)
            obs, reward, done, _ = env.step(action)
            # print(obs)
            # total_cost+=cost
            if done:
                print(t)

                break
        a = env.get_arr()
        a = np.array(a)
        # print(a)
        # print(np.shape(a))

        plt.plot(a[:, 0], a[:, 1], "r")

        np.savetxt("./trajectory_ex_{}.txt".format(III), a)

    for III in range(ntest):
        obs = env.reset()
        if III < len(ICs):
            IC = ICs[III]
            obs = env.reset_test(IC[0], IC[1], IC[2], IC[3], III)
        # print(III,obs)
        # input()
        total_cost = 0
        for t in range(max_timesteps):
            # Running policy_old:
            action = custom_action(obs)
            obs, reward, done, _ = env.step(-action)
            # total_cost+=cost
            if done:
                print(t)

                break
        a = env.get_arr()
        a = np.array(a)
        # print(a)
        # print(np.shape(a))
        plt.plot(a[:, 0], a[:, 1], "b")
        # if III == 3:
        #     plt.plot(a[:,0],a[:,1],'b')
        np.savetxt("./trajectory_in_{}.txt".format(III), a)
    plt.axis("equal")
    plt.xlim([-1, 11])
    plt.ylim([-2, 2])
    plt.savefig("traj_concentrab1.eps")
    plt.show()


if __name__ == "__main__":
    # from mpi4py import MPI

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    main(sensor_l=0, sensor_index=9, flow_index=0)
