import numpy as np


class ReferenceCost:
    def __init__(self, id, lanewidth):
        self.id = id
        self.lbd = lanewidth
        self.in_inter = False
        self.pass_inter = False

    def calc_r(self, x):
        
        r = 0.0
        if  self.id == 0: # id == 0: human, id == 1: ego
            if not self.in_inter:
                r = x[0]**2
            else:
                r = (x[1]-self.lbd/2)**2
        else:
            r = (x[5]-self.lbd/2)**2
        
        return r

    def calc_dldx(self, x):
        
        drdx = np.zeros((len(x),1))
        if self.id == 0:
            if not self.in_inter:
                drdx[0] = 2*x[0]
            else:
                drdx[1] = 2*(x[1]-self.lbd/2)
        else:
            drdx[5] = 2*(x[5]-self.lbd/2)

        return drdx

    def calc_Hx(self,x):

        d2rd2x = np.zeros((len(x), len(x)))
        if self.id == 0:
            if not self.in_inter:
                d2rd2x[0,0] = 2
            else:
                d2rd2x[1,1] = 2
        else:
            d2rd2x[5,5] = 2

        return d2rd2x

    def calc_Hu(self,u):
        return np.zeros((len(u),len(u)))


class InputPenalty:
    def __init__(self, u_idx):
        self.idx = u_idx
        self.in_inter = False
        self.pass_inter = False

    def calc_r(self, u):
        
        r = u[self.idx]**2 
        return r    

    def calc_dldx(self,x):
        drdx = np.zeros((len(x),1))
        return drdx
    
    def calc_Hx(self, x):
        d2rd2x = np.zeros((len(x), len(x)))
        return d2rd2x
    
    def calc_Hu(self,u):
        Hu = np.zeros((len(u),len(u)))
        Hu[self.idx, self.idx] = 2
        return Hu
    
class HeadingCost:
    def __init__(self, id):
        self.id = id 
        self.in_inter = False
        self.pass_inter = False

    def calc_r(self, x):
        r = 0.0
        if self.id == 0:
            if not self.in_inter:
                r = (x[2]-np.pi/2)**2 
            else:
                r = (x[2])**2 
        else:
            r = (x[6])**2 

        return r

    def calc_dldx(self,x):
        drdx = np.zeros((len(x),1))
        if self.id == 0:
            if not self.in_inter:
                drdx[2] = 2*(x[2]-np.pi/2)
            else:
                drdx[2] = 2*(x[2])
        else:
            drdx[6] = 2*(x[6])

        return drdx
    
    def calc_Hx(self, x):
        d2rd2x = np.zeros((len(x), len(x)))
        if self.id == 0:
            d2rd2x[2,2] = 2 
        else:
            d2rd2x[6,6] = 2
        return d2rd2x

    def calc_Hu(self,u):
        return np.zeros((len(u),len(u)))
    
class QuadraticCost:
    def __init__(self, x_idx, des=0.0):
        self.idx = x_idx
        self.in_inter = False
        self.pass_inter = False
        self.des = des

    def calc_r(self, x):
        r = (x[self.idx]-self.des)**2 
        return r

    def calc_dldx(self,x):
        drdx = np.zeros((len(x),1))
        drdx[self.idx] = 2*(x[self.idx]-self.des)
        return drdx
    
    def calc_Hx(self, x):
        d2rd2x = np.zeros((len(x), len(x)))
        d2rd2x[self.idx, self.idx]=2
        return d2rd2x

    def calc_Hu(self,u):
        return np.zeros((len(u),len(u)))


class LaneBoundaryCost:
    def __init__(self, id, lanewidth):
        self.id = id
        self.lbd = lanewidth
        self.in_inter = False
        self.pass_inter = False
        self.tres= 0.2

    def calc_r(self, x):
        
        r = 0.0

        if self.id == 0: # id == 0: human, id == 1: ego
            if not self.pass_inter:
                if x[0] <= -self.lbd/2+self.tres:
                    r = (x[0]+self.lbd/2)**2
                elif x[0] >= self.lbd/2-self.tres:
                    r = (x[0]-self.lbd/2)**2
            else:
                if x[1] <= 0.0+self.tres:
                    r = (x[1])**2
                elif x[1] >= self.lbd-self.tres:
                    r = (x[1]-self.lbd)**2 
                
                if x[0] <= -self.lbd/2+self.tres:
                    r = (x[0]+self.lbd/2)**2
        else:
            if x[5] <= 0.0+self.tres:
                r = (x[5])**2
            elif x[5] >= self.lbd-0.5:
                r = (x[5]-self.lbd)**2 

        return r
    
    def calc_dldx(self,x):
        
        drdx = np.zeros((len(x),1))

        if self.id == 0: # id == 0: human, id == 1: ego
            if not self.pass_inter:
                if x[0] <= -self.lbd/2+self.tres:
                    drdx[0] = 2*(x[0]+self.lbd/2)
                elif x[0] >= self.lbd/2-self.tres:
                    drdx[0] = 2*(x[0]-self.lbd/2)
            else:
                if x[1] <= 0.0+self.tres:
                    drdx[1] = 2*(x[1])
                elif x[1] >= self.lbd-self.tres:
                    drdx[1] = 2*(x[1]-self.lbd) 
                
                if x[0] <= -self.lbd/2+self.tres:
                    drdx[0] = 2*(x[0]+self.lbd/2)
        else:
            if x[5] <= 0.0+self.tres:
                drdx[5] = 2*(x[5])
            elif x[5] >= self.lbd-self.tres:
                drdx[5] = 2*(x[5]-self.lbd)

        return drdx
    
    def calc_Hx(self, x):

        d2rd2x = np.zeros((len(x), len(x)))
        if self.id == 0: # id == 0: human, id == 1: ego
            if not self.pass_inter:
                if x[0] <= -self.lbd/2+self.tres:
                    d2rd2x[0,0] = 2
                elif x[0] >= self.lbd/2-self.tres:
                    d2rd2x[0,0] = 2
            else:
                if x[1] <= 0.0+self.tres:
                    d2rd2x[1,1] = 2
                elif x[1] >= self.lbd-self.tres:
                    d2rd2x[1,1] = 2
                if x[0] <= -self.lbd/2+self.tres:
                    d2rd2x[0,0] = 2
        else:
            if x[5] <= 0.0+self.tres:
                d2rd2x[5,5] = 2
            elif x[5] >= self.lbd-self.tres:
                d2rd2x[5,5] = 2

        return d2rd2x
    
    def calc_Hu(self, u):
        return np.zeros((len(u),len(u)))

class PreferenceCost:
    def __init__ (self, px_dim):
        self.px_dim = px_dim
        self.in_inter = False
        self.pass_inter = False

    def calc_r(self, x):
        
        r = 0.0
        if self.in_inter and x[self.px_dim[0]]-x[self.px_dim[1]] >= 0: #For attentive
            r = x[0]**2
            r += 5*x[3]**2

        return r

    def calc_dldx(self, x):

        drdx = np.zeros((len(x),1))
        if self.in_inter and x[self.px_dim[0]]-x[self.px_dim[1]] >= 0: #For attentive
            drdx[0] = 2*x[0]
            drdx[3] = 5*2*x[3]
        return drdx
    
    def calc_Hx(self, x):

        d2rd2x = np.zeros((len(x), len(x)))
        if self.in_inter and x[self.px_dim[0]]-x[self.px_dim[1]] >= 0: #For attentive
            d2rd2x[0,0] = 2
            d2rd2x[3,3] = 10*2

        return d2rd2x
        
    def calc_Hu(self, u):
        return np.zeros((len(u),len(u)))


class CollisionCost:
    def __init__(self, px_dim, py_dim, x_dim, u_dim, distance):
        
        self.px_dim = px_dim
        self.py_dim = py_dim
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.distance = distance
        self.in_inter = False
        self.pass_inter = False

    def calc_r(self, x):
        

        dist_x = (x[self.px_dim[0]] - x[self.px_dim[1]])
        dist_y = (x[self.py_dim[0]] - x[self.py_dim[1]])
        dist = np.sqrt(dist_x**2 + dist_y**2)

        cost = ( min( (dist-self.distance), 0.0) )**2

        return cost
        
    def calc_dldx(self, x):
        
        dist_x = (x[self.px_dim[0]] - x[self.px_dim[1]])
        dist_y = (x[self.py_dim[0]] - x[self.py_dim[1]])
        dis = np.sqrt(dist_x**2 + dist_y**2) 

        drdx = np.zeros((self.x_dim,1))
        if dis < self.distance:

            drdx[0] = 2*(dist_x)*(dis - self.distance)/dis
            drdx[1] = 2*(dist_y)*(dis - self.distance)/dis
            drdx[4] = -2*(dist_x)*(dis - self.distance)/dis
            drdx[5] = -2*(dist_y)*(dis - self.distance)/dis
        return drdx

    def calc_Hx(self, x):

        dist_x = (x[self.px_dim[0]] - x[self.px_dim[1]])
        dist_y = (x[self.py_dim[0]] - x[self.py_dim[1]])
        dis = np.sqrt(dist_x**2 + dist_y**2) 

        d2rd2x = np.zeros((self.x_dim, self.x_dim))
        if np.sqrt(dist_x**2 + dist_y**2) < self.distance:
            d2f_dx2 = 2*( (dist_x**2)/((dist_x**2+dist_y**2)**1.5) - (dis-self.distance)*(dist_x**2)/((dist_x**2 + dist_y**2)**1.5) ) #d2f/dx2 for human
            d2f_dy2 = 2*( (dist_y**2)/((dist_x**2+dist_y**2)**1.5) - (dis-self.distance)*(dist_y**2)/((dist_x**2 + dist_y**2)**1.5) )

            d2f_da2 = d2f_dx2  
            d2f_db2 = d2f_dy2 
            
            d2f_dab = 2*( (dist_x*dist_y/((dist_x**2+dist_y**2)**1.5))*(dis-self.distance) + (dis-self.distance)*dist_x*dist_y/((dist_x**2 + dist_y**2)**1.5) )
            d2f_dxy = 2*( dist_x*dist_y/((dist_x**2+dist_y**2)**1.5) - (dis-self.distance)*dist_x*dist_y/((dist_x**2 + dist_y**2)**1.5) ) #d2f/dxdy, d2f/dxdy for human


            d2fdxda = -2*((dist_x**2)/((dist_x**2+dist_y**2)**1.5)-(dis-self.distance)*(dist_x**2)/(dist_x**2 + dist_y**2)**1.5)
            d2fdyda = -2*((dist_x*dist_y)/((dist_x**2+dist_y**2)**1.5)-(dis-self.distance)*(dist_x*dist_y)/(dist_x**2 + dist_y**2)**1.5)
            d2fdxdb = -2*((dist_x*dist_y)/((dist_x**2+dist_y**2)**1.5)-(dis-self.distance)*(dist_x*dist_y)/(dist_x**2 + dist_y**2)**1.5)
            d2fdydb = -2*((dist_y**2)/((dist_x**2+dist_y**2)**1.5)-(dis-self.distance)*(dist_y**2)/(dist_x**2 + dist_y**2)**1.5)
            
            d2rd2x[0,0] = d2f_dx2 
            d2rd2x[0,1] = d2f_dxy
            d2rd2x[0,4] = d2fdxda
            d2rd2x[0,5] = d2fdxdb

            d2rd2x[1,0] = d2f_dxy 
            d2rd2x[1,1] = d2f_dy2
            d2rd2x[1,4] = d2fdyda
            d2rd2x[1,5] = d2fdydb

            d2rd2x[4,0] = d2fdxda 
            d2rd2x[4,1] = d2fdyda
            d2rd2x[4,4] = d2f_da2
            d2rd2x[4,5] = d2f_dab

            d2rd2x[5,0] = d2fdxdb 
            d2rd2x[5,1] = d2fdydb
            d2rd2x[5,4] = d2f_dab
            d2rd2x[5,5] = d2f_db2

        return d2rd2x
    

    def calc_Hu(self, u):
        return np.zeros((self.u_dim,self.u_dim))