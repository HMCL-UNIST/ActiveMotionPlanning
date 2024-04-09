import math
import numpy as np
import matplotlib.pyplot as plt
from motion_planning.vehicle_model import *


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle

def plot_intersection(rd_width, rd_length, dim):
    plt.xlim([-dim,dim])
    plt.ylim([-dim,dim])
    plt.hlines(y=rd_width, xmin=rd_width/2, xmax=rd_length, color='black', linestyle='solid')
    plt.hlines(y=0.0, xmin=rd_width/2, xmax=rd_length, color='black', linestyle='solid')
    plt.hlines(y=rd_width, xmin=-dim, xmax=-rd_width/2, color='black', linestyle='solid')
    plt.hlines(y=0.0, xmin=-dim, xmax=-rd_width/2, color='black', linestyle='solid')
    plt.vlines(x=-rd_width/2, ymin=-dim, ymax=0.0, color='black', linestyle='solid')
    plt.vlines(x=rd_width/2, ymin=-dim, ymax=0.0, color='black', linestyle='solid')
    plt.vlines(x=-rd_width/2, ymin=rd_width, ymax=rd_length, color='black', linestyle='solid')
    plt.vlines(x=rd_width/2, ymin=rd_width, ymax=rd_length, color='black', linestyle='solid')

def plot_intersection_offline(rd_width, rd_length, traj_r, traj_h, cr, cr2, coll_arr):

    
    for i in range(len(traj_r)-2):
        plt.cla()
        
        plt.xlim([-15,15])
        plt.ylim([-15,15])
        plt.hlines(y=rd_width, xmin=rd_width/2, xmax=rd_length, color='black', linestyle='solid')
        plt.hlines(y=0.0, xmin=rd_width/2, xmax=rd_length, color='black', linestyle='solid')
        plt.hlines(y=rd_width, xmin=-15, xmax=-rd_width/2, color='black', linestyle='solid')
        plt.hlines(y=0.0, xmin=-15, xmax=-rd_width/2, color='black', linestyle='solid')

        plt.vlines(x=-rd_width/2, ymin=-15.0, ymax=0.0, color='black', linestyle='solid')
        plt.vlines(x=rd_width/2, ymin=-15.0, ymax=0.0, color='black', linestyle='solid')
        plt.vlines(x=-rd_width/2, ymin=rd_width, ymax=rd_length, color='black', linestyle='solid')
        plt.vlines(x=rd_width/2, ymin=rd_width, ymax=rd_length, color='black', linestyle='solid')

        dy = (traj_r[i+2,2] - traj_r[i,2]) / (traj_r[i,3] * Vehicle.dt)
        steer = pi_2_pi(-math.atan(Vehicle.WB * dy))

        draw.draw_car(traj_r[i,0], traj_r[i,1], traj_r[i,2], steer, Vehicle, False)
        plt.gcf().canvas.mpl_connect('key_release_event',
                                    lambda event:
                                    [exit(0) if event.key == 'escape' else None])

        plt.plot(traj_r[i:,0], traj_r[i:,1], cr)
        plt.plot(traj_r[i,0], traj_r[i,1], '*b')

        dy = (traj_h[i+2,2] - traj_h[i,2]) / (traj_h[i,3] * Vehicle.dt)
        steer = pi_2_pi(-math.atan(Vehicle.WB * dy))

        draw.draw_car(traj_h[i,0], traj_h[i,1], traj_h[i,2], steer, Vehicle, coll_arr[i])
        plt.gcf().canvas.mpl_connect('key_release_event',
                                    lambda event:
                                    [exit(0) if event.key == 'escape' else None])
        plt.plot(traj_h[i:,0], traj_h[i:,1], cr2, alpha=0.7)
        plt.plot(traj_h[i,0], traj_h[i,1], '*r')

        plt.text(5,-3.5, r"$v_R$: {:.2f}".format(float(traj_r[i,3])), color='black', fontsize=10)
        plt.text(5,-2.5, r"$v_H$: {:.2f}".format(float(traj_h[i,3])), color='black', fontsize=10)

        plt.pause(0.001)
        plt.draw()