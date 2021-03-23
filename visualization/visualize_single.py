import numpy as np, os
import cv2
from PIL import ImageColor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotation(data, alpha=0, beta=0):
        # rotate the skeleton around x-y axis
        r_alpha = alpha * np.pi / 180
        r_beta = beta * np.pi / 180

        rx = np.array([[1, 0, 0],
                       [0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
                       [0, np.sin(r_alpha), np.cos(r_alpha)]]
                      )

        ry = np.array([
            [np.cos(r_beta), 0, np.sin(r_beta)],
            [0, 1, 0],
            [-1 * np.sin(r_beta), 0, np.cos(r_beta)],
        ])

        r = ry.dot(rx)
        data = data.dot(r)

        return data


def normal_skeleton(data):
    #  use as center joint
    center_joint = data[0, :]

    center_jointx = center_joint[0]
    center_jointy = center_joint[1]
    center_jointz = center_joint[2]

    center = np.array([center_jointx, center_jointy, center_jointz])
    data = data - center

    return data


'''                #   belly   chest    neck     head    lshoulder
neighbor_edge =   [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                # lupperarm lforearm lwrist rshoulder rupperarm
                    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                #   rforearm  rwrist    lhip     lfemur    ltibia 
                    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                #   lfoot     rhip     rfemur    rtibia    rfoot
                    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                #   lfinger   lhand    rfinger   rhand 
                    (22, 23), (23, 8), (24, 25), (25, 12)]

darkred1, darkred2, darkred3 = '#7e0008', '#580005', '#320003'
green, darkgreen, darkgreen1, darkgreen2 = '#17695b', '#125a58', '#103f49', '#183449'
lightblue1, lightblue2, lightblue3, blue = '#b6d5eb', '#7fbadc', '#54a5d5', '#3892c6'
orange, darkorange = '#f79a39', '#e64242'

#honey, darkgreen = '#FCBF5D', '#007e3d', 
#red, blue, orange, green, darkblue = '#FD0010', '#00B2EE', '#FF631C', '#00FD7B', '#0C68F5'


color_edge =   [darkred1, darkred1, darkred2, darkred3, green, 
                darkgreen, darkgreen1, darkgreen2, lightblue1, lightblue2, 
                lightblue3, blue, green, darkgreen, darkgreen1, 
                orange, lightblue1, lightblue2, lightblue3, darkorange, 
                orange, orange, darkorange, darkorange]
'''

if not os.path.isdir('synthetic'):
    os.mkdir('synthetic')


trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]


root_data = '/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_data.npy'
data = np.load(root_data, mmap_mode='r')

print(data.shape)

data_numpy = np.transpose(data[100,:,0,:,0], (1, 0))
data_numpy = rotation(data_numpy, 0,50)
#data_numpy = normal_skeleton(data_numpy)


print(data_numpy.shape)
V, _ = data_numpy.shape
init_horizon=-45
init_vertical=20


fig = plt.figure()
ax = Axes3D(fig)

ax.view_init(init_vertical, init_horizon)

ax.set_xlim3d([-1, 1])
ax.set_ylim3d([-1, 1])
ax.set_zlim3d([0, 1.8])

x = data_numpy[:, 0]
y = data_numpy[:, 1]
z = data_numpy[:, 2]

for part in body:
    x_plot = x[part]
    y_plot = y[part]
    z_plot = z[part]
    ax.plot(x_plot, z_plot, y_plot, color='b', marker='o', markerfacecolor='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


plt.savefig("synthetic/zau.png")