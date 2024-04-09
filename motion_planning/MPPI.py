
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import functools
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))

def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray

def handle_batch_input(func):
    """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
        batch_dims = []
        for arg in args:
            if is_tensor_like(arg) and len(arg.shape) > 2:
                batch_dims = arg.shape[:-1]  # last dimension is type dependent; all previous ones are batches
                break
        # no batches; just return normally
        if not batch_dims:
            return func(*args, **kwargs)

        # reduce all batch dimensions down to the first one
        args = [v.view(-1, v.shape[-1]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
        ret = func(*args, **kwargs)
        # restore original batch dimensions; keep variable dimension (nx)
        if type(ret) is tuple:
            ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                v.view(*batch_dims, v.shape[-1]) if len(v.shape) == 2 else v.view(*batch_dims)) for v in ret]
        else:
            if is_tensor_like(ret):
                if len(ret.shape) == 2:
                    ret = ret.view(*batch_dims, ret.shape[-1])
                else:
                    ret = ret.view(*batch_dims)
        return ret

    return wrapper


class mppi:
    def __init__(self):
        #Prams for MPPI
        self.d="cuda"
        self.dtype = torch.float64 
        self.U = None
        self.U_prev = None
        self.lambda_ = 80

    def set_params(self, N, K, K2, xlim, ulim, Nx, Nu, dt, L):
        
        self.N = N  # Receding horizon
        self.Nx = Nx # Demension of states for two agents
        self.Nx_ = Nx//2 # Demension of states for an agents
        self.Nu = Nu # Demension of inputs for an agent

        self.K = K # Number of sampling of robot actions
        self.K2 = K2 # Number of sampling of human actions
        self.dt = dt

        self.xlim = torch.tensor(xlim).to(device= self.d)
        self.u_min = torch.tensor([-ulim[2], -ulim[3]]).to(device= self.d)
        self.u_max = torch.tensor([ulim[2], ulim[3]]).to(device= self.d)
        self.u_h_min = torch.tensor([-ulim[0], -ulim[1]]).to(device= self.d)
        self.u_h_max = torch.tensor([ulim[0], ulim[1]]).to(device= self.d)

        self.L = L
    
    def set_cost(self, weight, RH_A, RH_D, W, wac = 1, ca_lat_bd=2.5, ca_lon_bd=4):
        self.Qref = weight[0]
        self.Qrd = weight[2]
        self.Qpsi = weight[3]
        self.Qvel = weight[4]

        self.Racc = weight[5]
        self.Rdel = weight[6]

        self.RH_A = RH_A
        self.RH_D = RH_D

        self.w = W
        self.wca = 0.5 

        self.ca_lat_bd = ca_lat_bd
        self.ca_lon_bd = ca_lon_bd

    @handle_batch_input
    def _dynamics(self, state, u):
        return self.dynamics_update(state, u)

    def running_cost(self, xR, uR, uR_prev, xH, uH, xH_sig):
    
       ## Cost
        cost = torch.zeros(self.K*self.K2).to(device=self.d)

        cost = self.Qref*((xR[:,1]- self.xlim[1]/2)**2) # Center line Deviation
        cost += self.Qpsi*((xR[:,2]- 0.0)**2) # Psi Deviation
        cost += self.Qvel*((xR[:,3]- self.xlim[3])**2) # Velocity Deviation

        cost += self.Racc*(uR[:,0]**2) + self.Rdel*(uR[:,1]**2) # Input Penalty

        psi_h = xH[:,2]
        dist_x =  (xH[:,0]-xR[:,0])*torch.cos(psi_h) - (xH[:,1]-xR[:,1])*torch.sin(psi_h)
        dist_y =  (xH[:,0]-xR[:,0])*torch.sin(psi_h) + (xH[:,1]-xR[:,1])*torch.cos(psi_h)

        if self.beta_w:
            var = torch.zeros(self.K*self.K2,2).to(device=self.d)
            var[:,0] = (torch.cos(xH[:,2])**2 * xH_sig[:,0,0] + torch.sin(xH[:,2])**2  * xH_sig[:,1,1])
            var[:,1] = (torch.sin(xH[:,2])**2 * xH_sig[:,0,0] + torch.cos(xH[:,2])**2  * xH_sig[:,1,1])

            mu_a = (var[:,0] + self.ca_lon_bd).to(self.d)
            mu_b = (var[:,1] + self.ca_lat_bd).to(self.d)
        else:
            mu_a = self.ca_lon_bd*torch.ones(self.K*self.K2).to(self.d)
            mu_b = self.ca_lat_bd*torch.ones(self.K*self.K2).to(self.d)


        cost += self.w*torch.exp(-1/2*self.wca*(dist_x**2/(mu_a**2) + dist_y**2/(mu_b**2))) # Collision Avoidance 

        
        ## Constraint  
        idx = (dist_x**2/(mu_a**2) + dist_y**2/(mu_b**2)) <= 1
        if torch.any(idx):
            cost[idx] = torch.tensor(float('nan')) 

        idx = xR[:, 1] >= self.xlim[1]-0.0
        if torch.any(idx):
            cost[idx] = torch.tensor(float('nan')) 

        idx = xR[:, 1] <= 0.0
        if torch.any(idx):
            cost[idx] = torch.tensor(float('nan')) 
        
        idx = xR[:, 3] <= 0.0
        if torch.any(idx):
            cost[idx] =torch.tensor(float('nan')) 

        idx = xR[:, 3] >= self.xlim[7]
        if torch.any(idx):
            cost[idx] = torch.tensor(float('nan'))
        
        return cost

    def _compute_human_action(self, xR):

        xH_A, xH_D, u_A, u_D, Sigma_A, Sigma_D, xH_Sigma_A, xH_Sigma_D = self.update_human_state(xR)
        xH_Sigma = torch.zeros((self.K*self.K2, self.N, self.Nx_, self.Nx_), device=self.d, dtype=self.dtype)

        ## Sample the possible human aharacteristic
        if self.active: #Active Inference
            coin = np.random.uniform(0,1,self.K2)
            N_A = len( np.where(coin <= self.theta_init)[0] ) 
        else:  #PAssive Inference
            if self.theta_init >= 0.5:
                N_A = self.K2
            else:
                N_A = 0

        ## Sample the human action
        uH = torch.zeros((self.K*self.K2, self.N, self.Nu), device=self.d, dtype=self.dtype)
        xH = torch.zeros((self.K*self.K2 , self.N, 4), device=self.d, dtype=self.dtype)
        
        if N_A >= 1: #Number of samples for attentive
            ## Sample for Attentive Human Driver 
            uH_A_dist_ac = Normal(u_A[:,:,0], torch.sqrt(Sigma_A[:,:,0,0]))
            uH_A_dist_del = Normal(u_A[:,:,1], torch.sqrt(Sigma_A[:,:,1,1]))
            perturbed_uH_A_ac = uH_A_dist_ac.sample((N_A,)).transpose(0,1).reshape(-1,self.N) 
            perturbed_uH_A_del = uH_A_dist_del.sample((N_A,)).transpose(0,1).reshape(-1,self.N)
            perturbed_uH_A_ac = torch.max(torch.min(perturbed_uH_A_ac, self.u_h_max[0]), self.u_h_min[0])
            perturbed_uH_A_del = torch.max(torch.min(perturbed_uH_A_del, self.u_h_max[1]), self.u_h_min[1])
            
            uH[:self.K*N_A,:,0] = perturbed_uH_A_ac
            uH[:self.K*N_A,:,1] = perturbed_uH_A_del
            xH[:self.K*N_A,:,:] = self.trajectory_update(self.x0[:4], uH[:self.K*N_A],id=0)
            
            if self.beta_w:
                xH_Sigma[:self.K*N_A] = xH_Sigma_A.repeat(N_A,1,1,1)

        N_D = self.K2-N_A
        if N_D >= 1: #Number of samples for distracted
            ## Sample for Distracted Human Driver 
            uH_D_dist_ac = Normal(u_D[:,:,0], torch.sqrt(Sigma_D[:,:,0,0]))
            uH_D_dist_de1 = Normal(u_D[:,:,1], torch.sqrt(Sigma_D[:,:,1,1]))
            
            perturbed_uH_D_ac = uH_D_dist_ac.sample((N_D,)).transpose(0,1).reshape(-1,self.N) 
            perturbed_uH_D_del = uH_D_dist_de1.sample((N_D,)).transpose(0,1).reshape(-1,self.N)
            perturbed_uH_D_ac = torch.max(torch.min(perturbed_uH_D_ac, self.u_h_max[0]), self.u_h_min[0])
            perturbed_uH_D_del = torch.max(torch.min(perturbed_uH_D_del, self.u_h_max[1]), self.u_h_min[1])
        
            uH[self.K*N_A:,:,0] = perturbed_uH_D_ac
            uH[self.K*N_A:,:,1] = perturbed_uH_D_del

            xH[self.K*N_A:,:,:] = self.trajectory_update(self.x0[:4], uH[self.K*N_A:],id=0)
            if self.beta_w:
                xH_Sigma[self.K*N_A:] = xH_Sigma_D.repeat(N_D,1,1,1)

        return xH, uH, u_A, u_D, Sigma_A, Sigma_D, xH_Sigma, N_A
     

    def solve_mppi(self, state, ilq_results_A, ilq_results_D, theta_prob, beta_distr, active, beta_w):
        
        # Initial State
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        state = state.to(dtype=self.dtype, device=self.d)
        self.xR_0 = state[4:]
        self.x0 = state

        # Initial Human Internal State: characteristic and rationality
        self.theta_init = theta_prob[0]
        self.theta_prob = torch.zeros((self.N,self.K*self.K2), device=self.d, dtype=self.dtype)
        self.theta_prob_sampled = torch.zeros((self.N,self.K*self.K2), device=self.d, dtype=self.dtype)
        self.theta_prob[0,:] = theta_prob[0]

        self.beta = beta_distr

        # Initial Human Decision Model
        self.ilq_results_A = ilq_results_A
        self.ilq_results_D = ilq_results_D
    
        # Paramters for algorithm 
        self.active = active # Boolean: Active inference vs. Passive inference
        self.beta_w = beta_w # Boolean: With rationality vs. Wihtout consideration
        
        # Compute the cost of sampled trajectories
        self.ur_sampled_action = torch.torch.distributions.Uniform(low=self.u_min, high=self.u_max).sample((self.K, self.N))
        xRs, uRs, cost_total  = self._compute_rollout_costs(self.ur_sampled_action)
        

        if self.U_prev is None:
            self.U_prev = torch.zeros((self.N, self.Nu), dtype=self.dtype, device=self.d)
        if self.U is None:
            self.U = torch.zeros((self.N, self.Nu), dtype=self.dtype, device=self.d)

        if torch.all(torch.isnan(cost_total)):
            self.U = self.U_prev
        else:
            idx = ~cost_total.isnan()
            beta = torch.min(cost_total[idx])
            cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

            if self.active:
                theta_prob_prod = torch.prod(self.theta_prob_sampled, 0)
                weight = theta_prob_prod*cost_total_non_zero
                eta = torch.sum(weight[idx])
                weight = weight/eta
                
                for t in range(self.N):                  
                    self.U[t] = torch.sum( weight[idx].view(-1, 1)*uRs[t,idx], dim=0)   
            else:
                eta = torch.sum(cost_total_non_zero[idx])
                omega = (1. / eta) * cost_total_non_zero
                
                for t in range(self.N):  
                    self.U[t] = torch.sum( omega[idx].view(-1, 1)*uRs[t,idx], dim=0)

        self.U_prev = torch.roll(self.U, -1, dims=0)
        self.U_prev[-1] = torch.zeros_like(self.u_min)

        return self.U[0].cpu().numpy()


    def _compute_rollout_costs(self, sampled_actions):
        
        K, T, Nu = sampled_actions.shape

        cost_total = torch.zeros((K,), device=self.d, dtype=self.dtype)
        xR = self.xR_0.view(1, -1).repeat(K, 1)

        ## Propagate the robot trajectory based on sampled actions
        xRs = []
        uRs = []
        for t in range(self.N):
            uR =  sampled_actions[:, t]
            xR = self._dynamics(xR, uR)
            uRs.append(uR)
            xRs.append(xR)
        uRs = torch.stack(uRs) # Dimension of inputs  (N x K x Nu)
        xRs = torch.stack(xRs) # Dimension of states  (N x K x Nx)
        
        ## Propagate the human trajectory based on sampled robots actions
        xH, uH, u_A, u_D, Sigma_A, Sigma_D, xH_Simga, num_sampled_A = self._compute_human_action(xRs)
        
        xRs = xRs.repeat_interleave(self.K2, dim=1)
        uRs = uRs.repeat_interleave(self.K2, dim=1)
        
        uR_prev =None

        c = torch.zeros((self.K*self.K2), device=self.d, dtype=self.dtype)     
        
        self.theta_prob_sampled[0,:self.K*num_sampled_A] = self.theta_init
        self.theta_prob_sampled[0, self.K*num_sampled_A:] = 1- self.theta_init
        for t in range(self.N):
                
            c += self.running_cost(xRs[t], uRs[t], uR_prev, xH[:,t], uH[:,t], xH_Simga[:,t]) 

            if t < self.N-1:
                self.theta_prob[t+1] = self.updateTheta(self.theta_prob[t], uH[:,t], u_A[:,t,:2], u_D[:,t], Sigma_A[:,t], Sigma_D[:,t])
                self.theta_prob_sampled[t+1, :self.K*num_sampled_A] = self.theta_prob[t+1, :self.K*num_sampled_A]
                self.theta_prob_sampled[t+1, self.K*num_sampled_A:] = torch.ones(self.K*(self.K2-num_sampled_A)).to(self.d) - self.theta_prob[t+1, self.K*num_sampled_A:]

            uR_prev = uRs[t]
        

        return xRs, uRs, c 


    def update_human_state(self, xR):
        
        ### Get ILQ results
        x_ilq_ref_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.xs).to(self.d)
        u_ilq_ref_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.us).to(self.d)

        x_ilq_ref_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.xs).to(self.d)
        u_ilq_ref_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.us).to(self.d)

        Ps_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.Ps).to(self.d)
        alphas_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.alphas).to(self.d)
        Z_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.Zs).to(self.d)

        Ps_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.Ps).to(self.d)
        alphas_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.alphas).to(self.d)
        Z_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.Zs).to(self.d)

        alpha_scale = self.ilq_results_A.ilq_solve.alpha_scale
        
        ### For attentive driver
        states = torch.zeros((self.K, self.N + 1,  self.Nx), dtype=self.dtype, device=self.d)
        uH = torch.zeros((self.K, self.N,  self.Nu), dtype=self.dtype, device=self.d)
        states[:,0] = self.x0
        for i in range(self.N):
            uH_ = u_ilq_ref_A[i,:2] - Ps_A[0,i]@(states[:,i]- x_ilq_ref_A[i].T).T - 0.01*alphas_A[0,i]
            uH[:,i] = uH_.T
            states[:, i+1, :4] = self._dynamics(states[:,i,:4], uH_.T )
            states[:, i+1, 4:] = xR[i]

        xH_A = states[:,:,:4]
        u_A =  torch.zeros((self.K, self.N,  self.Nu*2), dtype=self.dtype, device=self.d)
        u_A[:,:,:2] = uH
        u_A[:,:,2:] = self.ur_sampled_action

        u_A[:,:,0] = torch.max(torch.min(u_A[:,:,0], self.u_h_max[0]), self.u_h_min[0])
        u_A[:,:,1] = torch.max(torch.min(u_A[:,:,1], self.u_h_max[1]), self.u_h_min[1])

        B = self.calc_dfdu(states[:,:-1], u_A, 2 ) # Shape: (K x N x Nx_ x Nu) 
        Sigma_A = self.get_covariance(self.RH_A, B, Z_A[:,:self.N] )
        if self.beta_w:
            A = self.calc_dfdx(states[:,:-1], u_A ) # Shape: (K x N x Nx_ x Nx_) 
            xH_Sigma_A = self.compute_state_covaraince(A,B[:,:,:self.Nx_,:],Sigma_A)
        else:
            xH_Sigma_A = None
        
        ### For distracted driver
        states = torch.zeros((self.K, self.N + 1, self.Nx), dtype=self.dtype, device=self.d)
        uH = torch.zeros((self.K, self.N,  self.Nu), dtype=self.dtype, device=self.d)
        states[:,0] = self.x0
        for i in range(self.N):
            uH_ = u_ilq_ref_D[i,:2] - Ps_D[0,i]@(states[:,i]- x_ilq_ref_D[i].T).T - 0.01*alphas_D[0,i]
            uH[:,i] = uH_.T
            states[:,i + 1, :4] = self._dynamics(states[:,i,:4], uH_.T )
            states[:, i+1, 4:] = xR[i]

        xH_D = states[:,:,:4]
        u_D =  torch.zeros((self.K, self.N,  self.Nu*2), dtype=self.dtype, device=self.d)
        u_D[:,:,:2] = uH
        u_D[:,:,2:] = self.ur_sampled_action

        u_D[:,:,0] = torch.max(torch.min(u_D[:,:,0], self.u_h_max[0]), self.u_h_min[0])
        u_D[:,:,1] = torch.max(torch.min(u_D[:,:,1], self.u_h_max[1]), self.u_h_min[1])

        B = self.calc_dfdu(states[:,:-1], u_D, 2 )
        Sigma_D = self.get_covariance(self.RH_D, B, Z_D[:,:self.N] )
        if self.beta_w:
            A = self.calc_dfdx(states[:,:-1], u_D ) # Shape: (K x N x Nx_ x Nx_) 
            xH_Sigma_D = self.compute_state_covaraince(A, B[:,:,:self.Nx_,:],Sigma_D)
        else:
            xH_Sigma_D = None


        return xH_A, xH_D, u_A, u_D, torch.abs(Sigma_A), torch.abs(Sigma_D), xH_Sigma_A, xH_Sigma_D

    def updateTheta(self, theta_now, uH, uH_A, uH_D, u_A_sigma, u_D_sigma):
        if self.beta_w: #Consideration of rationality
            x_next_sig_a = u_A_sigma/(self.beta.a.trunc_mu)
            x_next_sig_d = u_D_sigma/(self.beta.d.trunc_mu)
        else: # No consideration of rationality
            x_next_sig_a = u_A_sigma
            x_next_sig_d = u_D_sigma

        pdf_a = self.calc_pdf(uH, uH_A, x_next_sig_a)
        pdf_d = self.calc_pdf(uH, uH_D, x_next_sig_d)

        denom = theta_now*pdf_a
        denom2 = (torch.ones_like(theta_now)-theta_now)*pdf_d
        theta_ = denom/(denom+denom2)
        theta_ = theta_*0.5 + theta_now*0.5

        return theta_ 

    def calc_pdf(self, u, u_pred, u_sigma):
        u_pred = u_pred.repeat_interleave(self.K2, dim=0)
        u_sigma = u_sigma.repeat_interleave(self.K2, dim=0)

        a = (u_pred[:,0]-u[:,0])**2
        pdf_u_acc = torch.exp(-0.5*( (u_pred[:,0]-u[:,0])**2/u_sigma[:,0,0] ))/torch.sqrt(u_sigma[:,0,0]*2*math.pi)
        pdf_u_del = torch.exp(-0.5*( ((u_pred[:,1]-u[:,1])**2).T@(1/u_sigma[:,1,1]) )) /torch.sqrt(u_sigma[:,1,1]*2*math.pi)

        pdf = (pdf_u_acc + pdf_u_del)/2
        pdf = pdf.to(device =  self.d, dtype= self.dtype)

        return pdf

    def dynamics_update(self,x,u):
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(device=self.d)    
        if not torch.is_tensor(u):
            u = torch.tensor(u).to(device=self.d)    
        
        delta = u[:,1]
        a = u[:,0]

        nx = torch.clone(x).to(device=self.d)  
        v = x[:,3]
        psi = x[:,2]

        nx[:,0] = nx[:,0] + v * torch.cos(psi) * self.dt
        nx[:,1] = nx[:,1] + v * torch.sin(psi) * self.dt
        nx[:,2] = nx[:,2] + v * torch.tan(delta) * self.dt / self.N
        nx[:,3] = nx[:,3] + a * self.dt

        return nx
    

    def trajectory_update(self,x,u, id = 1): # 0: human 1: robot
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(device=self.d)    
        if not torch.is_tensor(u):
            u = torch.tensor(u).to(device=self.d)    
    
        state = torch.zeros((u.shape[0], self.N+1, 4)).to(self.d)
        state[:,0] = x

        for n in range(self.N):
            state[:,n+1,0] = state[:,n,0] + state[:,n,3]* torch.cos(state[:,n,2]) * self.dt
            state[:,n+1,1] = state[:,n,1] + state[:,n,3] * torch.sin(state[:,n,2]) * self.dt
            state[:,n+1,2] = state[:,n,2] + state[:,n,3] * torch.tan(u[:,n,1]) * self.dt /self.L
            state[:,n+1,3] = state[:,n,3] +  u[:,n,0]* self.dt

            if id ==0:
                idx = state[:, n+1, 3] >= self.xlim[7]
                if torch.any(idx):
                    state[idx, n+1, 3] = float(self.xlim[7]) #1e10

        return state[:,1:]

    def calc_dfdx(self, x, u):
        A = torch.zeros((self.K, self.N, 4, 4), dtype=self.dtype, device=self.d)

        A[:,:,0,0] = 1.0 
        A[:,:,0,2] = -x[:,:,3]*torch.sin(x[:,:,2])* self.dt
        A[:,:,0,3] = torch.cos(x[:,:,2])* self.dt
        A[:,:,1,1] = 1.0 
        A[:,:,1,2] = x[:,:,3]*torch.cos(x[:,:,2])* self.dt
        A[:,:,1,3] = torch.sin(x[:,:,2])* self.dt
        A[:,:,2,2] = 1.0
        A[:,:,2,3] = torch.tan(u[:,:,1])* self.dt/self.L
        A[:,:,3,3] = 1.0
        return A

    def calc_dfdu(self, x, u, player_num):
        B = torch.zeros((self.K, self.N, self.Nx_*player_num, 2), dtype=self.dtype, device=self.d)
        B[:,:,2,1] = x[:,:,3]*self.dt/((torch.cos(u[:,:,1])**2) * self.L)
        B[:,:,3,0] = 1.0*self.dt
            
        return B

    def compute_state_covaraince(self, A, B, Sigma_u):
        P = torch.zeros((self.K,self.N, self.Nx_, self.Nx_), dtype=self.dtype, device=self.d)
        for t in range(1, self.N):
            APA = torch.bmm(torch.bmm(A[:,t-1,:,:],P[:,t-1,:,:]), torch.transpose(A[:,t-1,:,:],1,2))
            BSB = torch.bmm(torch.bmm(B[:,t-1,:,:],Sigma_u[:,t-1,:,:]), torch.transpose(B[:,t-1,:,:],1,2))

            P[:,t,:,:] = APA + BSB
        return P

    def get_covariance(self, RH, B, Zs ):
        Z = Zs[0]
        
        RH = torch.tensor(RH).to(self.d)
        Sigma = torch.zeros((self.K, self.N,  self.Nu, self.Nu), dtype=self.dtype, device=self.d)
        
        Bi = B.transpose(2, 3)
        Z = Z.unsqueeze(0)
        Zs = Z.repeat(self.K,1,1,1)
        Sigma = torch.inverse(RH + Bi @ Zs @ B)

        return Sigma
    
