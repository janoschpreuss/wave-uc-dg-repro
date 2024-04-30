import numpy as np
from math import pi,sqrt  
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, dS,jump,div
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import create_unit_interval
from dolfinx.io import XDMFFile
import sys
sys.setrecursionlimit(10**6)
from GMREs import GMRes
from space_time import * 
from precomp_time_int import theta_ref, d_theta_ref 
from solver_tools import GetLuSolver, PySolver, GetPyPardisoSolver
import pypardiso
import time
import cProfile
import resource

solver_type = "pypardiso" # 
order_strings = ["order1","order2","order3"]
GCC = False
GCC_string = "noGCC-"

t0 = 0
T = 1/2

data_size = 0.25
ref_lvl_to_N = [1,2,4,8,16,32,64]
Nxs = [2,4,8,16,32,64,128]

#kstar = 1
#if order == 1:
#    qstar = 1
#else:
#    qstar = 0

stabs = {"data": 1e4, 
        "dual": 1.0,
        "primal": 1e-3,
        "primal-jump":1.0,
        "primal-jump-displ-grad":1.0
        #"primal-jump-vel":1.0,
        #"primal-jump-displ-grad":1e-5
       } 

# define quantities depending on space

def get_indicator_function(msh,Nx):

    if Nx > 2:
        def omega_Ind(x): 
            values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
            omega_coords =  ( x[0] <= data_size )  
            rest_coords = np.invert(omega_coords)
            values[omega_coords] = np.full(sum(omega_coords), 1.0)
            values[rest_coords] = np.full(sum(rest_coords), 0)
            return values
    else:
        x = ufl.SpatialCoordinate(msh)
        omega_indicator = ufl.And(x[0] <= data_size, x[0] >= 0.0) 
        omega_Ind = ufl.conditional(omega_indicator, 1, 0)
    
    return omega_Ind 


def sample_sol(t,xu):
    return ufl.cos(pi*(t))*ufl.sin(pi*xu[0])

def dt_sample_sol(t,xu):
    return -pi*ufl.sin(pi*(t))*ufl.sin(pi*xu[0])

def SolveProblem(msh,N,order,plot_sol=False):
    
    omega_Ind = get_indicator_function(msh=msh,Nx=Nx)
    k = order
    q = order
    kstar = order
    qstar = order
    st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind,stabs=stabs,sol=sample_sol,dt_sol=dt_sample_sol, data_dom_fitted= Nx > 2)
    st.SetupSpaceTimeFEs()
    st.PreparePrecondGMRes()
    A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
    b_rhs = st.GetSpaceTimeRhs()
     
    genreal_slab_solver = pypardiso.PyPardisoSolver()
    SlabMatSp = GetSpMat(st.GetSlabMat())
    
    genreal_slab_solver.factorize( SlabMatSp)   
    st.SetSolverSlab(PySolver(SlabMatSp, genreal_slab_solver))
    
    initial_slab_solver = pypardiso.PyPardisoSolver()
    SlabMatFirstSlabSp = GetSpMat( st.GetSlabMatFirstSlab())   
    initial_slab_solver.factorize( SlabMatFirstSlabSp  )   
    st.SetSolverFirstSlab(PySolver( SlabMatFirstSlabSp,  initial_slab_solver ))
    
    x_sweep_once  = fem.petsc.create_vector(st.SpaceTimeLfi)
    residual  = fem.petsc.create_vector(st.SpaceTimeLfi)
    diff = fem.petsc.create_vector(st.SpaceTimeLfi)
    st.pre_time_marching_improved(b_rhs, x_sweep_once)
    u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = 100000, tol = 1e-10, startiteration = 0, printrates = True)
    
    if plot_sol:
        st.PlotError(u_sol,N_space=500,N_time_subdiv=20)
   
    print("Errors in whole domain")
    errors = st.MeasureErrors(u_sol)
    print("")
    print("Errors in restricted space-time domain")
    errors_restrict = st.MeasureErrorsRestrict(u_sol)
    return errors, errors_restrict, st.delta_t 


for order in [1,2,3]:
    
    delta_t = []
    Linfty_L2_u = {"all": [],
                   "restrict": [],
                   }
    L2_L2_ut = {"all": [],
                "restrict": [],
                }
    order_string = order_strings[order-1] 
    max_ref_lvl = 8-order
    for ref_lvl in range(0, max_ref_lvl): 
        N = ref_lvl_to_N[ref_lvl]
        Nx = Nxs[ref_lvl]
        msh = create_unit_interval(MPI.COMM_WORLD, Nx)
        errors, errors_restrict, time_step = SolveProblem(msh,N,order,plot_sol= ref_lvl == max_ref_lvl-2 and order == 2 )
        Linfty_L2_u["all"].append(errors["L-infty-L2-error-u"]) 
        Linfty_L2_u["restrict"].append(errors_restrict["L-infty-L2-error-u"]) 
        L2_L2_ut["all"].append( errors ["L2-L2-error-u_t"])
        L2_L2_ut["restrict"].append( errors_restrict ["L2-L2-error-u_t"])
        delta_t.append(time_step)

    fname = GCC_string+"restricted-1d-"+order_string+".dat" 
    results = [ np.array(delta_t,dtype=float) ] 
    header_str = "deltat "  
    for domain_type in ["all","restrict"]:
        for data_type,data_string in zip([Linfty_L2_u, L2_L2_ut], ["LinftyL2u","L2L2ut"] ):
            full_descriptor_str = data_string+"-"+domain_type+" "
            header_str += full_descriptor_str
            results.append( np.array(data_type[domain_type] ,dtype=float) )

    print(results)
    X = np.transpose(results)
    np.savetxt(fname ="../data/{0}".format(fname),
               X = X,
               header = header_str,
               comments = '')

