import math
import numpy as np
import bisect
import torch 
import matplotlib.pyplot as plt
import matplotlib.lines as line
from scipy.stats import truncnorm

import warnings

import utils.draw as draw
from utils.scenario import *
import utils.Reference as rf

from motion_planning.MPPI import *
from motion_planning.inference import *
from motion_planning.dynamics import *
from motion_planning.vehicle_model import *

from human_model.iLQcost import *
from human_model.solveiLQgame import *
from human_model.PlayerCost import *


plot_offline = True
plot_trajectory = False
warnings.filterwarnings(action='ignore')


for trial in range(0,1):
    
    ''' 
        Initilize the paramters 
    '''
    ## Parameters for ILQR
    num_player = 2
    alpha_scale = 0.3
    max_iteration= 200
    tolerence = 5e-3
   
    ## Parameters for reciding horizon
    dt = 0.1
    N_ilq = 8
    N = 8
    N_sim = 100

    rd_width = 4
    rd_length = 40
    vH_lim = 20/3.6
    vR_lim = 20/3.6

    ## Initial State
    xR_0 = [-30.0, rd_width/2, 0.0, vR_lim]
    xH_0 = [0.0, -30.0, math.pi/2, vH_lim]

    ## Goal State
    xR_f = [0.0, rd_width/2, 0.0, vR_lim]
    xH_f = [0.0, rd_width/2, 0.0, vH_lim]

    ## State and Control limit 
    xR_lim = [0.0, rd_width, 0.0, vR_lim]
    xH_lim = [0.0, rd_width, 0.0, vH_lim]
    uR_lim = [5, 0.5]
    uH_lim = [5, 0.5]
    x_lim = np.array(xH_lim + xR_lim)
    u_lim = np.array(uH_lim + uR_lim)

    ## Generate reference trajectory
    pts_r = np.array([[xR_0[0], rd_width/2], [rd_length+30, rd_width/2]])
    pts_h = np.array([[0.0, xH_0[1]],[0.0, -5], [rd_width/2, rd_width/2-1], [rd_width/2+4, rd_width/2],  [rd_length+30, rd_width/2] ])
    ref_r = rf.reference(dt, 1, 1, pts_r)
    ref_h = rf.reference(dt, 1, 1, pts_h)

    if plot_trajectory:
        plt.plot(ref_h.ref_pts[:,0], ref_h.ref_pts[:,1], "xb", label="input")
        plt.plot(ref_r.ref_pts[:,0], ref_r.ref_pts[:,1], "xr", label="input")
        plt.plot(ref_h.cx, ref_h.cy, "--b")
        plt.plot(ref_r.cx, ref_r.cy, "--r")

        plt.hlines(y=rd_width, xmin=-15, xmax=rd_length, color='black', linestyle='solid')
        plt.hlines(y=0.0, xmin=-15, xmax=-rd_width/2, color='black', linestyle='solid')
        plt.hlines(y=0.0, xmin=rd_width/2, xmax=rd_length, color='black', linestyle='solid')

        plt.vlines(x=-rd_width/2, ymin=-15.0, ymax=0.0, color='black', linestyle='solid')
        plt.vlines(x=rd_width/2, ymin=-15.0, ymax=0.0, color='black', linestyle='solid')
        plt.axes().set_aspect('equal')
        plt.show()
        

    ## Initionalize dynamics and optimization
    L  = 1.5
    Ego = VehicleDyanmics(ref_r, L, dt)
    Human = VehicleDyanmics(ref_h, L, dt)
    
    xH_dim = Human.x_dim
    xR_dim = Ego.x_dim

    xH_dims = list(range(0,xH_dim))
    xR_dims = list(range(xH_dim, xR_dim+xH_dim))

    x_dim = xH_dim + xR_dim
    x_dims = np.array([xH_dims, xR_dims])
    xHdims = np.array(xH_dims)

    uH_dim = Human.u_dim
    uR_dim = Ego.u_dim

    uH_dims = list(range(0,uH_dim))
    uR_dims = list(range(uH_dim,uH_dim+uR_dim))

    u_dim = uH_dim + uR_dim
    u_dims = np.array([uH_dims, uR_dims])

    dynamics_A = InteractionDynamics(dt, x_dim, u_dim, x_dims, u_dims, [Human,Ego])
    dynamics_D = InteractionDynamics(dt, x_dim, u_dim, x_dims, u_dims, [Human,Ego])


    Ps_A = np.zeros((num_player, N_ilq, uH_dim, x_dim))
    alpha_A = np.zeros((num_player, N_ilq, uH_dim, 1))
    Ps_D = np.zeros((num_player, N_ilq, uH_dim, x_dim))
    alpha_D = np.zeros((num_player, N_ilq, uH_dim, 1))


    '''
        Human Model
    '''

    #Attentive Human Cost
    Cost_Ref_A = ReferenceCost(0, rd_width) #0: Human_A, 1:Robot, 2:Human_D

    px_indices = [0,4]
    py_indices = [1,5]
    distance = 7
    Cost_Collision_A = CollisionCost(px_indices, py_indices, x_dim, uH_dim, distance)
    Cost_Goal_A = PreferenceCost(px_indices)
    Cost_RoadBoundary_A = LaneBoundaryCost(0, rd_width)
    Cost_Heaing_A = HeadingCost(0)
    Cost_Vel_A = QuadraticCost(3, vH_lim) #Penalty for acceleration
    Cost_Acc_A = InputPenalty(0) #Penalty for acceleration
    Cost_Delta_A = InputPenalty(1) 
    

    Car_H_cost_A = PlayerCost()
    Car_H_cost_A.add_cost(Cost_Ref_A, 'x',  163)
    Car_H_cost_A.add_cost(Cost_RoadBoundary_A, 'x', 250)
    Car_H_cost_A.add_cost(Cost_Goal_A, 'x', 85)
    Car_H_cost_A.add_cost(Cost_Heaing_A, 'x', 128)
    Car_H_cost_A.add_cost(Cost_Vel_A, 'x', 4.5)
    Car_H_cost_A.add_cost(Cost_Collision_A, 'x', 250)
    Car_H_cost_A.add_cost(Cost_Acc_A, 'u', 1.8)
    Car_H_cost_A.add_cost(Cost_Delta_A, 'u', 7.1)

    RH_A = np.diag([1.8, 7.1])

    #Distracted Human Cost
    Cost_Ref_D = ReferenceCost(0, rd_width) #0: Human_A, 1:Robot, 2:Human_D
    Cost_Collision_D = CollisionCost(px_indices, py_indices, x_dim, uH_dim, distance)
    Cost_RoadBoundary_D = LaneBoundaryCost(0, rd_width)
    Cost_Heading_D = HeadingCost(0)
    Cost_Vel_D = QuadraticCost(3, vH_lim)
    Cost_Acc_D = InputPenalty(0) 
    Cost_Delta_D = InputPenalty(1) 
    

    Car_H_cost_D = PlayerCost()
    Car_H_cost_D.add_cost(Cost_Ref_D, 'x', 163)
    Car_H_cost_D.add_cost(Cost_RoadBoundary_D, 'x', 250)
    Car_H_cost_D.add_cost(Cost_Heading_D, 'x', 128)
    Car_H_cost_D.add_cost(Cost_Vel_D, 'x', 4.5)
    Car_H_cost_D.add_cost(Cost_Collision_D, 'x', 5)
    Car_H_cost_D.add_cost(Cost_Acc_D, 'u', 1.8)
    Car_H_cost_D.add_cost(Cost_Delta_D, 'u', 7.1)
    RH_D = np.diag([1.8, 7.1])

    


    '''
        Robot Model
    '''

    ##Ego Vehicle Cost
    Cost_Ref_R = ReferenceCost(1, rd_width) #0: Human, 1:Robot
    Cost_Collision_R = CollisionCost(px_indices, py_indices, x_dim, uR_dim, distance)
    Cost_RoadBoundary_R = LaneBoundaryCost(1, rd_width)
    Cost_Heading_R = QuadraticCost(6)
    Cost_Vel_R = QuadraticCost(7,des=xR_f[3])
    Cost_Acc_R = InputPenalty(0) #Penalty for acceleration
    Cost_Delta_R = InputPenalty(1) #Penalty for steering angle

    W = 1200 #2000  #weight for collision
    Car_cost_R = PlayerCost()
    Car_cost_R.add_cost(Cost_Ref_R, 'x', 120)
    Car_cost_R.add_cost(Cost_Collision_R, 'x', W)
    Car_cost_R.add_cost(Cost_RoadBoundary_R, 'x', 500) 
    Car_cost_R.add_cost(Cost_Heading_R, 'x', 20)
    Car_cost_R.add_cost(Cost_Vel_R, 'x', 10)
    Car_cost_R.add_cost(Cost_Acc_R, 'u', 1)
    Car_cost_R.add_cost(Cost_Delta_R, 'u', 1)
    
    Costs_A = [Car_H_cost_A, Car_cost_R]
    Costs_D = [Car_H_cost_D, Car_cost_R]



    '''
       Internal State
    '''

    ## Initialize the probability distribution
    mu_beta_init = 0.6
    sigma_beta_init = 0.14

    beta_lim = [0.2, 1.0]
    beta_distr = beta_prob_distr(mu_beta_init, sigma_beta_init, beta_lim)
    theta_prob = [0.5, 0.5]
    

    ## Internal states of true human model
    beta = np.random.normal(mu_beta_init,sigma_beta_init, size =1)
    if beta <= beta_lim[0]:
        beta = beta_lim[0]
    elif beta >= beta_lim[1]:
        beta = beta_lim[1]
    theta = 'a'


    '''
        Initialization of human decision-making and robot motion planning
    '''
    ## Human decision-making
    x0 = np.array(xH_0 + xR_0)
    u0 = np.zeros((4,1))
    ilq_results_A = ilq_results(xHdims, dynamics_A, Ps_A, \
                            alpha_A, Costs_A, uH_dim, u_lim, \
                            alpha_scale, N_ilq, max_iteration, tolerence)
    ilq_results_D = ilq_results(xHdims, dynamics_D, Ps_D,\
                            alpha_D, Costs_D, uH_dim, u_lim, \
                            alpha_scale, N_ilq, max_iteration, tolerence)
    in_inter = False
    pass_inter = False
    

    ## Robot motion planning
    K = 3000  # Number of robot action sample
    K2 = 4    # Number of human action sample
    ca_lat_bd = 1.5 # ellipsoid length for collision avoidance
    ca_lon_bd = 3.5 # ellipsoid length for collision avoidance
   
    Ego_opt = mppi()
    Ego_opt.set_params(N, K, K2, x_lim, u_lim, x_dim, uR_dim, dt, L)
    Ego_opt.set_cost(Car_cost_R.weights, RH_A, RH_D, W, wac = 1, ca_lat_bd=ca_lat_bd, ca_lon_bd=ca_lon_bd)
    active = True # Boolean: Active inference vs. Passive inference
    beta_w = True # Boolean: With rationality vs. Wihtout consideration

    ## Paramters for saving result data
    Ego_traj = []
    Human_traj = []
    THETA = []
    BETA = []
    BETA.append(mu_beta_init)
    Collision = False
    Collision_arr = []



    '''
        Run the algorithm
    '''

    for t in range(N_sim):
        
        print("-----------------------------------------------------")
        print("trial", trial, "simuation time:", t, "true beta", beta)

        '''
            Update robot action
        '''
        ilq_results_A.solveiLQgame(x0)
        ilq_results_D.solveiLQgame(x0)

        uR  = Ego_opt.solve_mppi(x0, ilq_results_A, ilq_results_D, theta_prob, beta_distr, active, beta_w)
        xR = Ego.update(x0[4:],uR)


        '''
            Update human action
        '''
        uH_A = ilq_results_A.ilq_solve.best_operating_point.us[0,:2]
        uH_D = ilq_results_D.ilq_solve.best_operating_point.us[0,:2]

        ## Attentive Human
        u0_A = np.array([uH_A[0,], uH_A[1,],uR[0], uR[1]])
        A, B = dynamics_A.linearizeDiscrete_Interaction(x0, u0_A)
        Sigma_A = get_covariance(RH_A, B[0], ilq_results_A.ilq_solve.best_operating_point.Zs[0,0])
        Sigma_A = np.linalg.inv(Sigma_A)
        Sigma_A = np.abs(Sigma_A)

        ## Distracted Human
        u0_D = np.array([uH_D[0,], uH_D[1,],uR[0], uR[1]])
        A, B = dynamics_D.linearizeDiscrete_Interaction(x0, u0_D)
        Sigma_D = get_covariance(RH_D, B[0], ilq_results_D.ilq_solve.best_operating_point.Zs[0,0])
        Sigma_D = np.linalg.inv(Sigma_D)
        Sigma_D = np.abs(Sigma_D)

        if theta == 'a':
            uH_a = np.random.normal(u0_A[0],  Sigma_A[0,0]/beta, size =1)
            uH_delta = np.random.normal(u0_A[1], Sigma_A[1,1]/beta, size =1)
        
            u0[0] = np.clip(uH_a,-uH_lim[0], uH_lim[0])
            u0[1] = np.clip(uH_delta,-uH_lim[1], uH_lim[1])
            
        elif theta == 'd':
            uH_a = np.random.normal(u0_D[0],  Sigma_D[0,0]/beta, size =1)
            uH_delta = np.random.normal(u0_D[1], Sigma_D[1,1]/beta, size =1)
     
            u0[0] = np.clip(uH_a,-uH_lim[0], uH_lim[0])
            u0[1] = np.clip(uH_delta,-uH_lim[1], uH_lim[1])


        xH = Human.update(x0[:4],u0[:2])
        
        if xH[3] >= xH_f[3]:
            xH[3] = xH_f[3]
  
        


        '''
            Update Belief over Human Internal State
        '''
       
        ##  Rationality
        mu, sigma, trunc_mu  = beta_distr.updateBeta('a', u0[:2].reshape(2,), uH_A.reshape(2,), Sigma_A)
        beta_distr.a.mu = mu
        beta_distr.a.covar = sigma
        beta_distr.a.trunc_mu = trunc_mu 

        mu, sigma, trunc_mu = beta_distr.updateBeta('d', u0[:2].reshape(2,), uH_D.reshape(2,), Sigma_D)

        beta_distr.d.mu = mu
        beta_distr.d.covar = sigma
        beta_distr.d.trunc_mu = trunc_mu

        ##  Characteristic
        theta_ = updateTheta(theta_prob, beta_distr, u0[:2].reshape(2,), uH_A.reshape(2,), uH_D.reshape(2,), Sigma_A, Sigma_D, beta_w)

        theta_prob = theta_


        '''
            Print the results
        '''
        
        if theta == 'a':
            print("uH is ", uH_A.reshape(2,))
            print(np.round(Sigma_A/beta,2))
        elif theta == 'd':
            print("uH is ", uH_D.reshape(2,))
            print(np.round(Sigma_D/beta,2))

  
        if theta == 'a':
            print("truncated beta is: %f, predicted beta is: %f , cov is: %f" %(beta_distr.a.trunc_mu, beta_distr.a.mu, beta_distr.a.covar))
            error = round( math.sqrt((beta - beta_distr.a.trunc_mu)**2),2)
            error2 = round( math.sqrt((beta -beta_distr.a.mu)**2),2)
            print("error truncated beta is: %f, predicted beta is: %f " %(error, error2))
            print("curent_prob: ", theta_prob[0])
            THETA.append(theta_prob[0])
            BETA.append(beta_distr.a.trunc_mu)
        elif theta == 'd':
            print("true truncated is: %f, predicted beta is: %f, cov is: %f " %(beta_distr.d.trunc_mu, beta_distr.d.mu, beta_distr.d.covar))
            error = round( math.sqrt((beta -beta_distr.d.trunc_mu)**2),2)
            error2 = round( math.sqrt((beta-beta_distr.d.mu)**2),2)
            print("error truncated beta is: %f, predicted beta is: %f " %(error, error2))
            print("curent_prob: ", theta_prob[1])
            THETA.append(theta_prob[1])
            BETA.append(beta_distr.s.trunc_mu)

        x0[:4] = xH.reshape(4,)
        x0[4:] = xR.reshape(4,)


        '''
            Check the collision
        '''
        
        psi = xH[2]
        dist_x =  (xH[0] - xR[0])*math.cos(psi) + (xH[1] - xR[1])*math.sin(psi)
        dist_y =  (xH[0] - xR[0])*math.sin(psi) - (xH[1] - xR[1])*math.cos(psi)
        cl = ((dist_x**2)/((ca_lon_bd)**2) + (dist_y**2)/((ca_lat_bd)**2))

        if cl <= 1 or Collision: # Collision with human-driven vehicle
            print("Collision!!!!!!!")
            Collision = True
    
        if xR[1] > rd_width+0.1 or xR[1] < -0.1 or Collision: # Collision with the wall 
            Collision = True
            print("Collision!!!!!!!")
        
        Collision_arr.append(Collision)


        Ego_traj.append([float(xR[0]),float(xR[1]),float(xR[2]),float(xR[3]) ])
        Human_traj.append([float(xH[0]),float(xH[1]),float(xH[2]),float(xH[3]) ])
        
        if xH[1] >= -10.0 and not in_inter:
            ilq_results_A.costs[0].check_intersection()
            ilq_results_A.costs[1].check_intersection()
            ilq_results_D.costs[0].check_intersection()
            ilq_results_D.costs[1].check_intersection()
            in_inter = True
            
        
        if xH[1] >= 0.0 and not pass_inter:
            ilq_results_A.costs[0].pass_intersection()
            ilq_results_A.costs[1].pass_intersection()
            ilq_results_D.costs[0].pass_intersection()
            ilq_results_D.costs[1].pass_intersection()
            pass_inter = True
            
        print("-----------------------------------------------------")

    print("Simulation Done")
    PassInter = False

    if not Collision  and xR[0] >= 0.0:
        PassInter = True
        print("Success the Goal")

    if plot_offline:
        for k in range(0,len(Human_traj),1):
            plt.cla()
            plot_intersection(rd_width, rd_length,30)
                
            Human_traj_ = np.array(Human_traj)
            plt.plot(Human_traj_[:k,0], Human_traj_[:k,1], '-',color='grey', alpha=0.5)
            Ego_traj_ = np.array(Ego_traj)
            plt.plot(Ego_traj_[:k,0], Ego_traj_[:k,1], '--', color='y', alpha=0.5)
            
            if t > 2: 
                dy_ego = (Ego_traj_[k,2] - Ego_traj_[k-2,2]) / (Ego_traj_[k-2,3] * Vehicle.dt)
                steer_ego = pi_2_pi(-math.atan(Vehicle.WB * dy_ego))

                dy_human = (Human_traj_[k,2] - Human_traj_[k,2]) / (Human_traj_[k,3] * Vehicle.dt)
                steer_human = pi_2_pi(-math.atan(Vehicle.WB * dy_ego))
            else:
                steer_ego = 0.0
                steer_human = 0.0

            draw.draw_car(Ego_traj_[k,0], Ego_traj_[k,1], Ego_traj_[k,2], steer_ego, Vehicle, Collision, color='yellow', alpha=1)
            draw.draw_car(Human_traj_[k,0], Human_traj_[k,1], Human_traj_[k,2], steer_human, Vehicle, Collision, color='grey', alpha=1)

            plt.pause(0.0001)
            plt.draw()


    np.savez_compressed('./result/test_'+str(trial), \
                        ego = Ego_traj, \
                        human = Human_traj, \
                        beta = BETA,\
                        t_beta = beta, \
                        theta = THETA, \
                        t_theta = theta,\
                        PassInter = PassInter,\
                        Collision = Collision )  

    
    


