
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def read_trj_data(filename):
    with open(filename,mode='r',encoding='utf-8') as file:
        lines = file.readlines()
    data=[ ]
    cnt=0
    trj_prev=-1
    for line in lines:
        cols = line.split()
        if len(cols)>1: # skip empty line
            trj_id = int(cols[0])
            frame = int(cols[1])
            xpos = float(cols[6])  # depth
            ypos = float(cols[7])
            zpos = float(cols[8])  # height
            xvel = float(cols[12])
            yvel = float(cols[13])
            zvel = float(cols[14])
            n = int(cols[15]) # length of sequence

            if trj_prev != trj_id:
                if trj_prev!=-1: # very first one
                    data.append(trajectory)
                trajectory = []
                trj_prev = trj_id
                cnt += 1
            else:
                trajectory.append([frame,xpos,ypos,zpos,xvel,yvel,zvel])

    return data
###

filename = "20191117-S3F4676E1#1S20.trj"

data = read_trj_data(filename)

ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=-60)
for trj in data:
    if len(trj)<10:
        continue
    X = []
    Y = []
    Z = []
    for x in trj:
        X.append(x[1])
        Y.append(x[2])
        Z.append(x[3])        
    ax.plot3D(X, Y, Z, 'red', linewidth=0.4)
ax.set_title('Trajectory: ' + filename)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.grid(True)
plt.show()
    
