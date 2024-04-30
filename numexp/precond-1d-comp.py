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
from space_time_fb import space_time as space_time_fb 
from space_time import space_time,GetSpMat 
from precomp_time_int import theta_ref, d_theta_ref 
from solver_tools import GetLuSolver, PySolver, GetPyPardisoSolver
import pypardiso
import time
import cProfile
import resource
from meshes import get_1Dmesh
from decimal import Decimal

# parse terminal input
if ( len(sys.argv) > 1 and int(sys.argv[1]) in [1,2,3]):
    order = int(sys.argv[1])
else:
    raise ValueError('Invalid input!')

order_strings = ["order1","order2","order3"]
order_string = order_strings[order-1] 

well_posed = False
GCC = True
t0 = 0
T = 1/2
tol = 1e-7
maxsteps = 3000

data_size = 0.25
ref_lvl_to_N = [1,2,4,8,16,32,64]
Nxs = [2,4,8,16,32,64,128]

k = order
q = order
kstar = order
qstar = order

stabs_fb = {"data": 1e4, 
        "dual": 1e-2,
        "primal": 1e-3,
        "primal-jump":1.0,
        "primal-jump-displ-grad":1.0
       } 

stabs_m = {"data": 1e4, 
        "dual": 1e0,
        "primal": 1e-3,
        "primal-jump":1e0,
        "primal-jump-displ-grad":1.0
       } 

def get_indicator_function(msh,Nx,GCC=True):

    if GCC:
        if Nx > 2:
            def omega_Ind(x): 
                values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
                omega_coords = np.logical_or( ( x[0] <= data_size ), (x[0] >= 1-data_size ))  
                rest_coords = np.invert(omega_coords)
                values[omega_coords] = np.full(sum(omega_coords), 1.0)
                values[rest_coords] = np.full(sum(rest_coords), 0)
                return values

        else:
            x = ufl.SpatialCoordinate(msh)
            omega_indicator = ufl.Not(
                                ufl.And(x[0] >= data_size, x[0] <= 1.0-data_size) 
                                     )
            omega_Ind = ufl.conditional(omega_indicator, 1, 0)

    else:
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

def pre_Id(b,x_sol):
    x_sol.array[:] = b.array[:]


def SolveProblemFB(msh,N):

    st = space_time_fb(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind,stabs=stabs_fb,sol=sample_sol,dt_sol=dt_sample_sol,well_posed=well_posed,data_dom_fitted= Nx > 2)
    st.SetupSpaceTimeFEs()
    st.PreparePrecondGMRes()
    A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
    b_rhs = st.GetSpaceTimeRhs()
    st.PrepareFBsweep() 
     
    st.SetSolverSlabMat_u(GetPyPardisoSolver(st.GetSlabMat_u()))       
    st.SetSolverFirstSlab_u(GetPyPardisoSolver(st.GetSlabMat_u_FirstSlab()))       
    st.SetSolverSlabMat_z(GetPyPardisoSolver(st.GetSlabMat_z()))       
    st.SetSolverFirstSlab_z(GetPyPardisoSolver(st.GetSlabMat_z_FirstSlab()))       

    A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
    A_space_time = A_space_time_linop  
    u_sol,_ =  A_space_time_linop.createVecs()
    u_sol,res = GMRes(  A_space_time ,b_rhs,pre=st.forward_backward_sweep  ,maxsteps = maxsteps , tol = tol, startiteration = 0, printrates = True)            
    return st.MeasureErrors(u_sol),res

def SolveProblem(msh,N,solver_option="MTM-full"):

    if solver_option == "MTM-lo":
        st = space_time(q=q,qstar=0,k=k,kstar=1,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind,stabs=stabs_m,sol=sample_sol,dt_sol=dt_sample_sol,well_posed=well_posed,data_dom_fitted= Nx > 2)
    else:
        st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind,stabs=stabs_m,sol=sample_sol,dt_sol=dt_sample_sol,well_posed=well_posed,data_dom_fitted= Nx > 2)
    st.SetupSpaceTimeFEs()
    st.PreparePrecondGMRes()
    A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
    b_rhs = st.GetSpaceTimeRhs()
    
    if solver_option in ["MTM-full","MTM-lo"]:
        st.SetSolverSlab(GetPyPardisoSolver(st.GetSlabMat()))
        st.SetSolverFirstSlab(GetPyPardisoSolver(st.GetSlabMatFirstSlab()))
        u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = maxsteps , tol = tol, startiteration = 0, printrates = True)
    if solver_option == "block":
        st.SetSolverSlab(GetPyPardisoSolver(st.GetSlabMat()))
        st.SetSolverFirstSlab(GetPyPardisoSolver(st.GetSlabMatFirstSlab()))
        u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_Block_Jacobi,maxsteps = maxsteps, tol = tol, startiteration = 0, printrates = True)
    if solver_option == "vanilla":
        u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=pre_Id, maxsteps = maxsteps, tol = tol, startiteration = 0, printrates = True)
    
    ndof = len(u_sol.array[:])
    return st.MeasureErrors(u_sol), res, ndof, st.delta_t 

Linfty_L2_u = {"DFB": [],
               "MTM-full": [],
               "MTM-lo": [],
               "block": [],
               "vanilla": []
               }
L2_L2_ut = {"DFB": [],
            "MTM-full": [],
            "MTM-lo": [],
            "block": [],
            "vanilla": []
            }
iters = {"DFB": [],
         "MTM-full": [],
         "MTM-lo": [],
         "block": [],
         "vanilla": []
         }
residuals = {"DFB": [],
             "MTM-full": [],
             "MTM-lo": [],
             "block": [],
             "vanilla": []
             }

ndof_mtm_full = [ ] 
delta_t_mtm_full = [ ] 

if order == 1:
    max_ref_lvl = 7-order
    for ref_lvl in range(0, max_ref_lvl):
     
        N = ref_lvl_to_N[ref_lvl]
        Nx = Nxs[ref_lvl]
        msh = create_unit_interval(MPI.COMM_WORLD, Nx)
        omega_Ind = get_indicator_function(msh=msh,Nx=Nx,GCC=GCC)
        
        # Solve problem using forward backward sweep
        result,res = SolveProblemFB(msh,N) 
        Linfty_L2_u["DFB"].append( result["L-infty-L2-error-u"])
        L2_L2_ut["DFB"].append( result["L2-L2-error-u_t"])
        residuals["DFB"].append(res)
        iters["DFB"].append(len(res))

        # Solve problem using monolithic forward sweep
        for solver_option in ["MTM-full", "MTM-lo","block","vanilla"]:
            result,res,ndof,delta_t = SolveProblem(msh,N,solver_option) 
            Linfty_L2_u[solver_option].append( result["L-infty-L2-error-u"])
            L2_L2_ut[solver_option].append( result["L2-L2-error-u_t"])
            residuals[solver_option].append(res)
            iters[solver_option].append(len(res))
            if solver_option == "MTM-full":
                ndof_mtm_full.append(ndof) 
                delta_t_mtm_full.append(delta_t)

    for data_type,data_string in zip([Linfty_L2_u, L2_L2_ut, iters ], ["LinftyL2u","L2L2ut","iters"] ):
        fname = "precond-1d-"+data_string+"-"+order_string+".dat"
         
        if data_string == "iters":
            results = [ np.array(ref_lvl_to_N[:max_ref_lvl],dtype=int), np.array( ndof_mtm_full,dtype=int) ] 
            header_str = "N ndof DFB MTM-full MTM-lo block vanilla"   
        else:
            results = [ np.array(delta_t_mtm_full[:max_ref_lvl],dtype=float) ] 
            header_str = "deltat DFB MTM-full MTM-lo block vanilla"   
        
        for solver_type in [ "DFB", "MTM-full", "MTM-lo", "block", "vanilla"]: 
            print( solver_type)
            print(data_type[ solver_type])
            if data_string == "iters":
                results.append(np.array(data_type[ solver_type],dtype=int))
            else:
                results.append(np.array(data_type[ solver_type],dtype=float))
        if data_string == "iters":
            X = np.transpose(results).astype(int) 
            np.savetxt(fname ="../data/{0}".format(fname),
                       X = X,
                       header = header_str,
                       comments = '',
                       fmt='%s'
                       )
        else:
            X = np.transpose(results)
            np.savetxt(fname ="../data/{0}".format(fname),
                       X = X,
                       header = header_str,
                       comments = '')


if order == 2:
    
    max_ref_lvl = 7-order
    for ref_lvl in range(0, max_ref_lvl):
     
        N = ref_lvl_to_N[ref_lvl]
        Nx = Nxs[ref_lvl]
        msh = create_unit_interval(MPI.COMM_WORLD, Nx)
        omega_Ind = get_indicator_function(msh=msh,Nx=Nx,GCC=GCC)
        
        # Solve problem using forward backward sweep
        result,res = SolveProblemFB(msh,N) 
        Linfty_L2_u["DFB"].append( result["L-infty-L2-error-u"])
        L2_L2_ut["DFB"].append( result["L2-L2-error-u_t"])
        residuals["DFB"].append(res)
        iters["DFB"].append(len(res))

        # Solve problem using monolithic forward sweep
        for solver_option in ["MTM-full", "MTM-lo"]:
            result,res,ndof,delta_t = SolveProblem(msh,N,solver_option) 
            Linfty_L2_u[solver_option].append( result["L-infty-L2-error-u"])
            L2_L2_ut[solver_option].append( result["L2-L2-error-u_t"])
            residuals[solver_option].append(res)
            iters[solver_option].append(len(res))
            if solver_option == "MTM-full":
                ndof_mtm_full.append(ndof) 
                delta_t_mtm_full.append(delta_t)

    for data_type,data_string in zip([Linfty_L2_u, L2_L2_ut, iters ], ["LinftyL2u","L2L2ut","iters"] ):
        fname = "precond-1d-"+data_string+"-"+order_string+".dat"
         
        if data_string == "iters":
            results = [ np.array(ref_lvl_to_N[:max_ref_lvl],dtype=int), np.array( ndof_mtm_full,dtype=int) ] 
            header_str = "N ndof DFB MTM-full MTM-lo"   
        else:
            results = [ np.array(delta_t_mtm_full[:max_ref_lvl],dtype=float) ] 
            header_str = "deltat DFB MTM-full MTM-lo"   
        
        for solver_type in [ "DFB", "MTM-full", "MTM-lo"]: 
            print( solver_type)
            print(data_type[ solver_type])
            if data_string == "iters":
                results.append(np.array(data_type[ solver_type],dtype=int))
            else:
                results.append(np.array(data_type[ solver_type],dtype=float))
        if data_string == "iters":
            X = np.transpose(results).astype(int) 
            np.savetxt(fname ="../data/{0}".format(fname),
                       X = X,
                       header = header_str,
                       comments = '',
                       fmt='%s'
                       )
        else:
            X = np.transpose(results)
            np.savetxt(fname ="../data/{0}".format(fname),
                       X = X,
                       header = header_str,
                       comments = '')

    for ref_lvl,ref_lvl_str in zip([2,3,4],["ref-lvl2","ref-lvl3","ref-lvl4"]):
        for solver_type in [ "DFB", "MTM-lo"]: 
            fname = "precond-1d-"+solver_type+"-"+order_string+"-residuals-"+ref_lvl_str+".dat"
            res = np.array(residuals[ solver_type][ref_lvl],dtype=float)
            results = [np.arange(1,len(res)+1),res] 
            header_str = "iter res"
            X = np.transpose(results)
            np.savetxt(fname = "../data/{0}".format(fname),
                       X = X,
                       header = header_str,
                       comments = '')
#print(Linfty_L2_u)
#print(L2_L2_ut) 
#print(iters)


