import numpy as np
import math
import torch
from scipy.stats import norm, multivariate_normal

class prob_distr:
    def __init__(self, mu_init, sigma_init):
        self.mu = mu_init
        self.covar = sigma_init
        self.trunc_mu = mu_init

class beta_prob_distr:
    def __init__(self, mu_init, sigma_init, beta_lim):
        self.a = prob_distr(mu_init, sigma_init)
        self.d = prob_distr(mu_init, sigma_init)
        self.beta_lim = beta_lim
        
        
    def updateBeta(self, theta, u, u_pred, u_sigma):
        
        if theta == 'a':
            mu_now = self.a.mu
            sigma_now = self.a.covar
        else:
            mu_now = self.d.mu
            sigma_now = self.d.covar
        
        sigma_now_inv = 1/sigma_now
        u_sigma_inv = np.linalg.inv(u_sigma)

        A = 1/2*sigma_now_inv
        B = mu_now*sigma_now_inv - 1/2*(u-u_pred)@u_sigma_inv@(u-u_pred).T
        C = 1/2 * mu_now**2*sigma_now_inv
        
        mu = (B + math.sqrt(B**2 + 4*A))/(4*A)
        sigma = 1/(2*A + 0.5/(mu**2))

        trunc_mu = self.truncated_normal(mu, sigma)

        return mu, sigma, trunc_mu

    def truncated_normal(self, mu, sigma):
        alpha = (self.beta_lim[0]-mu)/math.sqrt(sigma)
        beta = (self.beta_lim[1]-mu)/math.sqrt(sigma)
   
        pdf_alpha = math.exp(-1/2*(alpha**2))/math.sqrt(2*math.pi)
        pdf_beta = math.exp(-1/2*(beta**2))/math.sqrt(2*math.pi)
        cdf_alpha = 1/2*(1+math.erf(alpha/math.sqrt(2)))
        cdf_beta = 1/2*(1+math.erf(beta/math.sqrt(2)))

        Z = cdf_beta - cdf_alpha
        if Z <= 1e-4:
            Z = 1e-4
        trunc_mu = mu + (pdf_alpha - pdf_beta)*math.sqrt(sigma)/Z
        trunc_sig = sigma*(1-(beta*pdf_beta - alpha*pdf_alpha)/Z-((pdf_alpha - pdf_beta)/Z)**2)

        return trunc_mu


def updateTheta(theta_now, beta_distr_now, uH, uH_A, uH_D, u_A_sigma, u_D_sigma, beta_w):
    
    if beta_w:
        mu_a = beta_distr_now.a.trunc_mu 
        mu_d = beta_distr_now.d.trunc_mu
        x_next_sig_a = u_A_sigma/(mu_a)
        x_next_sig_d = u_D_sigma/(mu_d)
    else:
        x_next_sig_a = u_A_sigma
        x_next_sig_d = u_D_sigma

    pdf_a = calc_pdf(uH, uH_A, x_next_sig_a)
    pdf_d = calc_pdf(uH, uH_D, x_next_sig_d)
    denom = theta_now[0]*pdf_a
    denom2 = theta_now[1]*pdf_d

    theta_ = denom/(denom+denom2)
    theta_ = theta_*0.5 + theta_now[0]*0.5
    theta = [theta_, 1-theta_]

    # if np.isnan(theta[0]) or np.isnan(theta[1]):
    #     print(theta)   

    return theta

def calc_pdf(u, u_pred, u_sigma):
    pdf_u_acc = norm(u_pred[0], u_sigma[0,0]).pdf(u[0])
    pdf_u_del = norm(u_pred[1], u_sigma[1,1]).pdf(u[1])
    pdf = (pdf_u_acc + pdf_u_del)/2

    return pdf