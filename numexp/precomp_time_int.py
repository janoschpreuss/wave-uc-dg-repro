from sympy import * 
import numpy as np

def get_elmat_time(q,qstar):

    elmat_time = { } 

    x, y = symbols('x y') 

    basis_in_time = { 0: [1],
                      1: [1-x, x],
                      2: [2*(1-x)*(1/2-x), 4*x*(1-x), 2*x*(x-1/2)],
                      3: [(-9/2)*(x-1)*(x-2/3)*(x-1/3),(27/2)*x*(x-1)*(x-2/3), -(27/2)*x*(x-1)*(x-1/3), (9/2)*x*(x-1/3)*(x-2/3)] 
                    }

    phis = basis_in_time[q]
    chis = basis_in_time[qstar]

    M_time_q_qstar = np.zeros((qstar+1,q+1))
    for i in range(qstar+1):
        for j in range(q+1):
            M_time_q_qstar[i,j] = integrate(phis[j]*chis[i], (x,0,1))
    elmat_time["M_time_q_qstar"] = M_time_q_qstar 

    M_time_q_q = np.zeros((q+1,q+1))
    for i in range(q+1):
        for j in range(q+1):
            M_time_q_q[i,j] = integrate(phis[j]*phis[i], (x,0,1))
    elmat_time["M_time_q_q"] = M_time_q_q

    M_time_qstar_qstar = np.zeros((qstar+1,qstar+1))
    for i in range(qstar+1):
        for j in range(qstar+1):
            M_time_qstar_qstar[i,j] = integrate(chis[j]*chis[i], (x,0,1))
    elmat_time["M_time_qstar_qstar"] = M_time_qstar_qstar 


    DM_time_q_qstar = np.zeros((qstar+1,q+1))
    for i in range(qstar+1):
        for j in range(q+1):
            DM_time_q_qstar[i,j] = integrate(diff(phis[j],x)*chis[i], (x,0,1))
    elmat_time["DM_time_q_qstar"] = DM_time_q_qstar 


    DMT_time_q_qstar = np.zeros((qstar+1,q+1))
    for i in range(qstar+1):
        for j in range(q+1):
            DMT_time_q_qstar[i,j] = integrate( phis[j] * diff(chis[i],x) , (x,0,1))
    elmat_time["DMT_time_q_qstar"] = DMT_time_q_qstar 

    DM_time_q_q = np.zeros((q+1,q+1))
    for i in range(q+1):
        for j in range(q+1):
            DM_time_q_q[i,j] = integrate(diff(phis[j],x)*phis[i], (x,0,1))
    elmat_time["DM_time_q_q"] = DM_time_q_q

    DDM_time_q_q = np.zeros((q+1,q+1))
    for i in range(q+1):
        for j in range(q+1):
            DDM_time_q_q[i,j] = integrate(diff(phis[j],x)*diff(phis[i],x), (x,0,1))
    elmat_time["DDM_time_q_q"] = DDM_time_q_q

    return elmat_time 

#elmat_time = get_elmat_time(2,2)
#print("DM_time_q_q = " , elmat_time["DM_time_q_q"])  


class quad_rule:

    def __init__(self,name,npoints):
        self.name = name
        self.npoints = npoints

        # available quadrature rules
        gauss_radau = {
            3: ([-1, (1-sqrt(6))/5, (1+sqrt(6))/5],
                [2/9, (16+sqrt(6))/18,(16-sqrt(6))/18]),
            4: ([-1, -0.575319, 0.181066, 0.822824],
                [0.125,0.657689,0.776387,0.440924]),
            5: ([-1, -0.72048, -0.167181, 0.446314, 0.885792],
                [0.08, 0.446208, 0.623653, 0.562712, 0.287427])
        }
        gauss = {
            1: ( [ 0 ],
                 [ 2 ] ),
            2: ( [ - 1.0/sqrt(3), 1.0/sqrt(3) ],
                 [ 1, 1 ] ),
            3: ( [ - sqrt(3/5), 0, sqrt(3/5) ],
                 [ 5/9, 8/9, 5/9 ] ),
        }

        gauss_lobatto = {
            3: ( [ -1, 0, 1 ],
                 [ 1/3, 4/3, 1/3 ] ),
            4: ( [ -1, -np.sqrt(1/5), np.sqrt(1/5), 1],
                 [ 1/6, 5/6, 5/6, 1/6 ] ),
            5: ( [ -1, -(1/7)*np.sqrt(21),0.0, (1/7)*np.sqrt(21), 1.0 ],
                 [ 1/10,49/90,32/45, 49/90, 1/10  ] ),
            6: ( [ -1, -np.sqrt((1/21)*(7+2*np.sqrt(7))), -np.sqrt((1/21)*(7-2*np.sqrt(7))), np.sqrt((1/21)*(7-2*np.sqrt(7))), np.sqrt((1/21)*(7+2*np.sqrt(7))), 1.0 ],
                 [ 1/15, (1/30)*(14-np.sqrt(7)), (1/30)*(14+np.sqrt(7)), (1/30)*(14+np.sqrt(7)), (1/30)*(14-np.sqrt(7)),  1/15 ] ),
        }

        newton_cotes = {
            2: [1/2,1/2],
            3: [1/6,4/6,1/6]
        }

        if name == "Gauss-Radau":
            self.points = gauss_radau[npoints][0]
            self.weights = gauss_radau[npoints][1]
        elif name == "Gauss-Lobatto":
            self.points = gauss_lobatto[npoints][0]
            self.weights = gauss_lobatto[npoints][1]
        elif name == "Gauss":
            self.points = gauss[npoints][0]
            self.weights = gauss[npoints][1]
        elif name == "Newton-Cotes":
            self.points = [0 + i/(self.npoints -1) for i in range(self.npoints)]
            self.weights = newton_cotes[npoints]
        elif name == "Midpoint-Rule":
            self.points = [0.5]
            self.weights = [1.0]
        
    def current_pts(self,a,b):
        if self.name == "Gauss-Radau" or self.name == "Gauss" or self.name == "Gauss-Lobatto":
            return [0.5*(b-a) * pt + 0.5*(b+a)  for pt in self.points]
        elif self.name == "Newton-Cotes":
            h = (b-a)/(self.npoints -1)
            return [a + i*h for i in range(self.npoints)]
        elif self.name == "Midpoint-Rule":
            return [0.5*a+0.5*b]
    
    def t_weights(self,delta_t):
        if self.name == "Gauss-Radau" or self.name == "Gauss" or self.name == "Gauss-Lobatto":
            return [0.5*delta_t*w for w in self.weights]
        elif self.name == "Newton-Cotes":
            return [delta_t*w for w in self.weights]
        elif self.name == "Midpoint-Rule":
            return [delta_t]

pts_Raudau_right = [ 0.0, 1.0 ] 
def theta_ref(tau): 
    result = 1 
    for x_mu in pts_Raudau_right: 
        result *= (tau - x_mu ) / ( -1 - x_mu )
    return result 

def d_theta_ref(tau): 
    result = 0
    for j in range(len(pts_Raudau_right)):
        fac_j = 1/(-1-pts_Raudau_right[j])
        for mu in range(len(pts_Raudau_right)):
            if mu != j: 
                fac_j *= (tau - pts_Raudau_right[mu] ) / (-1 - pts_Raudau_right[mu] )
        result += fac_j
    return result


basis_in_time = { 0: ([lambda t: 1],[lambda t: 0]),
                  1: ([lambda t: 1-t, lambda t: t],[ lambda t: -1, lambda t: 1 ]),
                  2: ([lambda t: 2*(1-t)*(1/2-t), lambda t:4*t*(1-t), lambda t:2*t*(t-1/2)], [ lambda t: -3+4*t, lambda t: 4-8*t, lambda t: 4*t-1] ),
                  3 :([lambda t: (-9/2)*(t-1)*(t-2/3)*(t-1/3), lambda t:(27/2)*t*(t-1)*(t-2/3), lambda t:-(27/2)*t*(t-1)*(t-1/3), lambda t: (9/2)*t*(t-1/3)*(t-2/3)],
                      [lambda t: (-9/2)*((t-2/3)*(t-1/3) + (t-1)*(t-1/3) + (t-1)*(t-2/3)), lambda t: (27/2)*((t-1)*(t-2/3) + t*(t-2/3) + t*(t-1)), lambda t: -(27/2)*( (t-1)*(t-1/3) + t*(t-1/3) + t*(t-1)), lambda t: (9/2)*( (t-1/3)*(t-2/3) + t*(t-2/3) + t*(t-1/3)) ])
                }

nodal_points = { 0: [1/2], 
                 1: [0,1],
                 2: [0,1/2,1],
                 3: [0,1/3,2/3,1]
               }
