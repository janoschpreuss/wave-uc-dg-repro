import numpy as np
from math import pi,sqrt  
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, dS,jump,div
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import create_box,CellType, GhostMode
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
if ( len(sys.argv) > 1 and int(sys.argv[1]) in [1,2,3] and  int(sys.argv[2]) in [0,1] ):
    order = int(sys.argv[1])
    GCC_num = int(sys.argv[2])
else:
    raise ValueError('Invalid input!')

order_strings = ["order1","order2","order3"]
order_string = order_strings[order-1] 

well_posed = False
if GCC_num == 0:
    GCC = False
else:
    GCC = True


GCC_string = "GCC-"
if not GCC:
    GCC_string = "noGCC-"
t0 = 0
T = 1/2
tol = 1e-5
maxsteps = 3000

data_size = 0.25
ref_lvl_to_N = [1,2,4,8,16,32,64]
Nxs = [2,4,8,16,32,64,128]

k = order
q = order
kstar = order
qstar = order

if GCC:
    stabs_fb = {"data": 1e4, 
            "dual": 1e-2,
            #"dual": 5e-2,
            "primal": 1e-3,
            "primal-jump":1.0,
            "primal-jump-displ-grad":1.0
           } 
else:
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
                omega_coords = np.logical_or( ( x[0] <= data_size  ), 
                               np.logical_or(  ( x[0] >= 1.0-data_size  ),        
                               np.logical_or(   (x[1] >= 1.0-data_size  ), 
                               np.logical_or(    (x[1] <= data_size  ),            
                               np.logical_or(    (x[2] <= data_size  ),(x[2] >= 1.0-data_size  ) )
                                  )
                                 )
                                )
                               )
                rest_coords = np.invert(omega_coords)
                values[omega_coords] = np.full(sum(omega_coords), 1.0)
                values[rest_coords] = np.full(sum(rest_coords), 0)
                return values
        else:
            x = ufl.SpatialCoordinate(msh)
            omega_indicator = ufl.Not(
                               ufl.And(
                                ufl.And(x[0] >= data_size, x[0] <= 1.0-data_size), 
                                       ufl.And(ufl.And(x[1] >= data_size, x[1] <= 1.0-data_size),
                                               ufl.And( x[2] >= data_size, x[2] <= 1.0-data_size)
                                              )
                                      )
                                     )
            omega_Ind = ufl.conditional(omega_indicator, 1, 0)
    else:
        if Nx > 2:
            def omega_Ind(x): 
                values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
                omega_coords = np.logical_and( ( x[0] <= data_size  ), ( x[0] >= 0.0 ))  
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
    return ufl.cos(sqrt(3)*pi*t)*ufl.sin(pi*xu[0])*ufl.sin(pi*xu[1])*ufl.sin(pi*xu[2])

def dt_sample_sol(t,xu):
    return -sqrt(3)*pi*ufl.sin(sqrt(3)*pi*t)*ufl.sin(pi*xu[0])*ufl.sin(pi*xu[1])*ufl.sin(pi*xu[2])


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
    ndof = len(u_sol.array[:])
    return st.MeasureErrors(u_sol),res,ndof,len(st.only_uh_slab.vector.array[:])

def SolveProblem(msh,N,solver_option="MTM-full",plot_sol=False):

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
    
    if plot_sol:
        st.PlotParaview(u_sol,name="abserr-cube-{0}order{1}-MTM-lo".format(GCC_string,order))
        if k == 1:
            with XDMFFile(msh.comm, "measurement-domain-{0}-3d.xdmf".format(GCC_string), "w") as file:
                file.write_mesh(msh)
                file.write_function(st.omega_ind )
    ndof = len(u_sol.array[:])
    return st.MeasureErrors(u_sol), res, ndof, st.delta_t,len(st.uh_pre.x.array[:])

Linfty_L2_u = {"DFB": [],
               "MTM-lo": [],
               }
L2_L2_ut = {"DFB": [],
            "MTM-lo": [],
            }
iters = {"DFB": [],
         "MTM-lo": [],
         }
residuals = {"DFB": [],
             "MTM-lo": [],
             }

ndofs =  {"DFB": [],
             "MTM-lo": [],
             }

ndofs_inv =  {"DFB": [],
             "MTM-lo": [],
             }

delta_t_mtm_full = [ ] 

if order in [1,2,3]:
    max_ref_lvl = 6-order
    for ref_lvl in range(0, max_ref_lvl):
     
        N = ref_lvl_to_N[ref_lvl]
        Nx = Nxs[ref_lvl]
        msh = create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                                  np.array([1.0, 1.0, 1.0])], [Nx, Nx, Nx],
                         CellType.hexahedron, ghost_mode=GhostMode.shared_facet)
        #msh = create_unit_interval(MPI.COMM_WORLD, Nx)


        omega_Ind = get_indicator_function(msh=msh,Nx=Nx,GCC=GCC)
        
        # Solve problem using forward backward sweep
        result,res,ndof,ndofi = SolveProblemFB(msh,N) 
        Linfty_L2_u["DFB"].append( result["L-infty-L2-error-u"])
        L2_L2_ut["DFB"].append( result["L2-L2-error-u_t"])
        residuals["DFB"].append(res)
        iters["DFB"].append(len(res))
        ndofs["DFB"].append(ndof)
        ndofs_inv["DFB"].append(ndofi)


        # Solve problem using monolithic forward sweep
        result,res,ndof,delta_t,ndofi = SolveProblem(msh,N,solver_option= "MTM-lo",plot_sol=ref_lvl==max_ref_lvl-1) 
        Linfty_L2_u["MTM-lo"].append( result["L-infty-L2-error-u"])
        L2_L2_ut["MTM-lo"].append( result["L2-L2-error-u_t"])
        residuals["MTM-lo"].append(res)
        iters["MTM-lo"].append(len(res))
        ndofs["MTM-lo"] .append(ndof) 
        ndofs_inv["MTM-lo"] .append(ndofi) 
        delta_t_mtm_full.append(delta_t)

    for data_type,data_string in zip([Linfty_L2_u, L2_L2_ut, iters ], ["LinftyL2u","L2L2ut","iters"] ):
        fname = "precond-3d-"+GCC_string+data_string+"-"+order_string+".dat"
         
        if data_string == "iters":
            results = [ np.array(ref_lvl_to_N[:max_ref_lvl],dtype=int), np.array( ndofs["DFB"],dtype=int), np.array( ndofs_inv["DFB"],dtype=int), np.array(ndofs["MTM-lo"],dtype=int), np.array(ndofs_inv["MTM-lo"],dtype=int)  ] 
            header_str = "N ndof-tot-DFB ndof-inv-DFB ndof-tot-MTM-lo ndof-inv-MTM-lo DFB-iter MTM-lo-iter"    
        else:
            results = [ np.array(delta_t_mtm_full[:max_ref_lvl],dtype=float) ] 
            header_str = "deltat DFB MTM-lo"   
        
        for solver_type in [ "DFB", "MTM-lo"]: 
            #print( solver_type)
            #print(data_type[ solver_type])
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

