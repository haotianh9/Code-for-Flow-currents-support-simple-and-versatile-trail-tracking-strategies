from flowtaxis import FlowtaxisEnv
import numpy as np

import os
import matplotlib.pyplot as plt
import matplotlib


def custom_action(obs):
    if np.abs(obs) <= 0.0:
        return 0.0
    if obs > 0.0:
        return 1.0
    elif obs < 0.0:
        return -1.0


def main(sensor_l=0.0, sensor_index=0):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    flow_index = 0

    flow = [
        # "freq=1.25_amp=0.2",
        "Re=5000_freq=1.25_amp=0.2",
        "flapping_Re=0.1",
        "freq=0.5_amp=0.2",
        "Re=5000_freq=0.5_amp=0.2",
        "freq=1.75_amp=0.2",
    ]
    flow_freq = [1.25, 0.5, 0.5, 1.75]
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
    ]
    max_timesteps = 19000  # max timesteps in one episode

    # ICs=[0]*20
    # ICs=[[4.5,0.5,np.pi/6,0]]
    num_IC = 9
    print(sensor[sensor_index])

    velocity = 0.25
    decision_timestep = 0.005
    delayed_time = 0.05
    delayed_index = int(delayed_time / decision_timestep)
    print(delayed_index)
    env = FlowtaxisEnv(
        FlowFiled=flow[flow_index],
        SensorMode=sensor[sensor_index],
        sensor_l=sensor_l,
        MAX_ANGULARVELOCITY=2,
        velocity=velocity,
        decision_timestep=decision_timestep,
        flow_freq=flow_freq[flow_index],
        Peclet=1000,
    )
    plt.figure(figsize=(10, 3))
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    bwith = 0.25  # 边框宽度设置为2
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(bwith)
    ax.spines["left"].set_linewidth(bwith)
    ax.spines["top"].set_linewidth(bwith)
    ax.spines["right"].set_linewidth(bwith)
    # set tick width
    ax.xaxis.set_tick_params(width=bwith)
    ax.yaxis.set_tick_params(width=bwith)
    # ICs = []
    ICs = [
        # [6.5, 1.5, -np.pi / 3, 0],
        # [6.5, -1.5, np.pi / 2, 0],
        # [6.5, 0.5, -np.pi / 3, 0],
        [6.5, 0.02, 0.05, 0],
        # [6.5, 0.02, 0.04, 0],
        # [6.5, 0.02, -0.02, 0],
        # [6.5, 0.02, -0.05, 0],
        [6.5, 0.02, np.pi / 3, 0],
        [6.5, 0.02, np.pi, 0],
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
    III = 0
    # while len(ICs) < num_IC:
    for iii in range(len(ICs)):
        IC = ICs[iii]
        # high = np.array([4, 0.5, np.pi, 0])
        # low = np.array([3, -0.5, -np.pi, 0])
        # IC = np.random.uniform(low, high)
        print("excitatory {}".format(III))
        obs = env.reset()
        obs = env.reset_test(IC[0], IC[1], IC[2], IC[3], III)
        # print(III,obs)
        # input()
        obs_list = []
        obs_list.append(obs)
        total_cost = 0
        for t in range(max_timesteps):

            # Running policy_old:
            if t < delayed_index:
                action = 0.0
            else:
                # print(t-delayed_index)
                # print(len(obs_list))
                action = custom_action(obs_list[t - delayed_index])
            obs, reward, done, _ = env.step(action)
            obs_list.append(obs)
            # total_cost+=cost
            if done:
                print(t)

                break
        a1 = env.get_arr()
        a1 = np.array(a1)
        # if not env.get_success():
        #     continue

        print("inhibitory {}".format(III))
        obs = env.reset()
        obs = env.reset_test(IC[0], IC[1], IC[2], IC[3], III)
        # print(III,obs)
        # input()
        obs_list = []
        obs_list.append(obs)
        total_cost = 0
        for t in range(max_timesteps):

            # Running policy_old:
            if t < delayed_index:
                action = 0.0
            else:
                # print(t - delayed_index)
                # print(len(obs_list))
                # print(obs_list[t - delayed_index])
                action = custom_action(obs_list[t - delayed_index])

            obs, reward, done, _ = env.step(-action)
            obs_list.append(obs)
            # total_cost+=cost
            if done:
                print(t)

                break
        a2 = env.get_arr()
        a2 = np.array(a2)

        # if not env.get_success():
        #     continue

        III += 1
        # ICs.append([IC[0], IC[1], IC[2]])
        plt.plot(a1[:, 0], a1[:, 1], "r")
        np.savetxt("./trajectory_ex_{}.txt".format(III), a1)
        plt.plot(a2[:, 0], a2[:, 1], "b")
        np.savetxt("./trajectory_in_{}.txt".format(III), a2)
    # ICs = np.array(ICs)
    # plt.quiver(
    #     ICs[:, 0],
    #     ICs[:, 1],
    #     np.cos(ICs[:, 2]),
    #     np.sin(ICs[:, 2]),
    #     color="k",
    #     scale=32,
    #     headwidth=3,
    #     headlength=4,
    #     headaxislength=3,
    # )
    plt.axis("equal")
    plt.xlim([-1, 9])
    plt.ylim([-2, 2])
    plt.savefig("trajectory_conc.eps", format="eps")
    plt.show()


if __name__ == "__main__":
    # from mpi4py import MPI

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    main(sensor_l=0, sensor_index=7)
