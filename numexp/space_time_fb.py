import numpy as np
from math import pi,sqrt 
import ufl
from ufl import ds, dx, grad, inner, dS,jump,div
from dolfinx import fem, io, mesh, plot,geometry
from mpi4py import MPI
from petsc4py import PETSc
from precomp_time_int import get_elmat_time, quad_rule, basis_in_time,theta_ref,d_theta_ref 
import scipy.sparse as sp
from dolfinx.fem.petsc import assemble_matrix
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator, LinearLocator, FormatStrFormatter, NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#import sparse_dot_mkl
from decimal import Decimal

def GetSpMat(mat):
    ai, aj, av = mat.getValuesCSR()
    Asp = sp.csr_matrix((av, aj, ai))
    return Asp

# copied this here from https://github.com/UCL/dxh for experimental purposes 
def evaluate_function_at_points(
    function: fem.Function,
    points: NDArray[np.float64],) -> ArrayLike:
    """
    Evaluate a finite element function at one or more points.

    Args:
        function: Finite element function to evaluate.
        points: One or more points in domain of function to evaluate at. Should be
            either a one-dimensional array corresponding to a single point (with size
            equal to the geometric dimension or 3) or a two-dimensional array
            corresponding to one point per row (with size of last axis equal to the
            geometric dimension or 3).

    Returns:
        Value(s) of function evaluated at point(s).
    """
    mesh = function.function_space.mesh
    if points.ndim not in (1, 2):
        msg = "points argument should be one or two-dimensional array"
        raise ValueError(msg)
    if points.shape[-1] not in (3, mesh.geometry.dim):
        msg = "Last axis of points argument should be of size 3 or spatial dimension"
        raise ValueError(msg)
    if points.ndim == 1:
        points = points[None]
    if points.shape[-1] != 3:
        padded_points = np.zeros(points.shape[:-1] + (3,))
        padded_points[..., : points.shape[-1]] = points
        points = padded_points
    tree = geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    if not np.all(cell_candidates.offsets[1:] > 0):
        msg = "One or more points not within domain"
        raise ValueError(msg)
    cell_adjacency_list = geometry.compute_colliding_cells(
        mesh,
        cell_candidates,
        points,
    )
    first_cell_indices = cell_adjacency_list.array[cell_adjacency_list.offsets[:-1]]
    return np.squeeze(function.eval(points, first_cell_indices))


class space_time:
    
    # constructor
    def __init__(self,q,qstar,k,kstar,N,T,t,msh,Omega_Ind,stabs,sol,dt_sol,jumps_in_fw_problem=True,well_posed=False,data_dom_fitted=True, csq_Ind = None  ):
        self.name = "space-time-wave"
        self.q = q 
        self.qstar = qstar
        self.k = k
        self.kstar = kstar
        self.csq_Ind = csq_Ind 

        self.jumps_in_fw_problem = jumps_in_fw_problem
        self.N = N
        self.T = T
        self.t = t 
        self.delta_t = self.T/self.N
        self.msh = msh
        self.Omega_Ind = Omega_Ind 
        self.stabs = stabs

        
        self.stabs["primal-jump-vel"] = self.stabs["primal-jump"]
        if not self.stabs["primal-jump-displ-grad"]:
            self.stabs["primal-jump-displ-grad"] = self.stabs["primal-jump"]
        
        self.x = ufl.SpatialCoordinate(msh)
        self.sol = sol
        self.dt_sol = dt_sol
        self.lam_Nitsche = 200*self.k**2  
        self.jumps_in_fw_problem = jumps_in_fw_problem
        self.well_posed = well_posed
        self.data_dom_fitted = data_dom_fitted
        #self.mkl_matrix_mult = mkl_matrix_mult

        # sound speed 
        if csq_Ind == None:
            self.csq = 1.0 
        else:
           Q_ind = fem.FunctionSpace(self.msh, ("DG", 0))  
           self.csq = fem.Function(Q_ind)
           self.csq.interpolate(self.csq_Ind)

        # mesh-related 
        self.metadata = {"quadrature_degree": 2*self.k+1}
        #self.metadata = {"quadrature_degree": 2*self.k+10}
        self.dx = ufl.Measure("dx", domain=self.msh, metadata=self.metadata)
        self.n_facet = ufl.FacetNormal(self.msh)
        self.h = ufl.CellDiameter(self.msh)

        # derived quantities for time discretization
        self.elmat_time = get_elmat_time(self.q,self.qstar)
        self.qr = quad_rule("Gauss-Lobatto",5)
        self.qr_ho = quad_rule("Gauss-Lobatto",6)
        self.phi_trial, self.dt_phi_trial = basis_in_time[self.q] 
        self.phi_test, self.dt_phi_test = basis_in_time[self.q] 
        self.subspace_time_order = [self.q,self.q,self.qstar,self.qstar]

        # DG0 indicator function
        if data_dom_fitted:
            Q_ind = fem.FunctionSpace(self.msh, ("DG", 0)) 
            self.omega_ind = fem.Function(Q_ind)
            self.omega_ind.interpolate(self.Omega_Ind)
        else:
            self.omega_ind = self.Omega_Ind 

        # FESpaces  
        self.fes = None  
        self.fes_u1_0 = None 
        self.fes_u2_0 = None 

        self.fes_z1_0 = None 
        self.fes_z2_0 = None

        self.dofmaps_full = None 
        self.fes_coupling = None

        self.fes_coupling_uz = None

        self.fes_p = None 
        self.fes_u1_0_pre = None 
        self.fes_u2_0_pre = None 
        self.fes_slab_bnd = None 
        # for fb-precond
        self.fesu_slab = None 
        self.fesz_slab = None 
        self.fesu_slab_bnd = None 
        self.fesz_slab_bnd = None 


        # Dofmaps 
        self.dofmaps_full = None
        self.dofmaps_coupling = None  
        self.dofmaps_coupling_uz = None 
        self.dofmap_fes_u1_0_pre = None
        self.dofmap_fes_u2_0_pre = None
        self.dofmaps_slab_bnd = None
        # for fb-precond
        self.dofmaps_uslab = None 
        self.dofmaps_zslab = None


        # test functions for pre
        self.w1_p = None
        self.w2_p = None
        self.y1_p = None
        self.y2_p = None

        # Bilinear forms / matrices 
        # SpaceTime 
        self.SpaceTimeBfi = None 
        self.SpaceTimeMat = None
        self.ApplySpaceTimeMat = None
        # Slab matrix with DG-jumps
        self.SlabBfi = None 
        self.SlabMat = None 
        self.SolverSlab = None

        # for fb-precond 
        self.SolverSlab_u = None
        self.SolverSlab_z = None

        # for fb-precond
        self.SlabMat_u = None
        self.SlabMat_u_FirstSlab = None 
        self.SlabMat_z = None
        self.SlabMat_z_FirstSlab = None 
        self.RhsCorrectionMatZ = None 

        # Slab matrix without DG-jumps
        self.SlabBfiNoDGJumps = None 
        self.SlabMatNoDGJumps = None 
        #self.SlabMatNoDGJumps_sp = None 
        self.SolverSlabNoDGJumps = None

        # Coupling between slice
        self.CouplingMatBetweenSlices = None
        self.CouplingMatBetweenSlices_uz =  None
        

        # Scaled mass matrix between slice 
        self.ScaledMassMatrixBetweenSlices = None 
        # for fb-precond
        self.MassMatrixBetweenSlices_u =  None
        self.MassMatrixBetweenSlices_z =  None

        # Special treatment of first time-slab
        self.SlabBfiFirstSlab = None 
        self.SlabMatFirstSlab = None 
        self.SolverFirstSlab = None
        # for fb-precond 
        self.SolverFirstSlab_u = None
        self.SolverFirstSlab_z = None
        # Special treatment of first time-slab
        self.SlabMatLastSlab = None 

        
        # LinearForm 
        self.SpaceTimeLfi = None  
        self.SpaceTimeRhs = None  

        # auxiliary vectors 
        self.vec_slab1 = None 
        self.vec_slab2 = None 
        self.vec_coupling1 = None 
        self.vec_coupling2 = None 
        self.u1_0_pre = None
        self.u2_0_pre = None
        self.vec_0_bnd_in = None 
        self.vec_0_bnd_out = None 
        self.vec_u_bnd_in = None
        self.vec_u_bnd_out = None 
        self.x_fw_sweep = None
        self.b_z_corrected = None
        self.vec_only_uh_in = None 
        self.vec_only_uh_out = None
        self.vec_coupling1_uz = None 
        self.vec_coupling2_uz = None


        # auxiliary FEM functions
        self.u1_0  = None
        self.u2_0  = None
        self.z1_0 = None
        self.z2_0 = None
        self.uh_pre = None
        self.uh = None  
        self.u1_minus = None
        self.u1_slab = None 
        self.uh_best = None  
        # for fb-precond
        self.only_uh_slab = None
        self.only_uh_slab_out = None 
        self.only_zh_slab = None


        # Miscalleneous
        #self.sample_pts_error = np.linspace(0,self.T,105).tolist()
        self.sample_pts_error = np.linspace(0,self.T,60).tolist() #reduced to speed up the calculation of the errors
        self.eps = 0
        if self.qstar == 0:
            self.eps = 1e-15 # hack to trick form compiler

    def SetupSpaceTimeFEs(self):

        # Full Space-time system
        vel_q  = ufl.VectorElement('CG', self.msh.ufl_cell(), self.k, int((self.q+1)*self.N))
        vel_qstar  = ufl.VectorElement('CG', self.msh.ufl_cell(), self.kstar, int((self.qstar+1)*self.N))
        mel = ufl.MixedElement([vel_q,vel_q,vel_qstar,vel_qstar])
        self.fes = fem.FunctionSpace(self.msh,mel)
        dofmaps_full = [] 
        for j in range(4):
            dofmaps_j = [ ]
            int_idx_q = 0
            int_idx_qstar = 0
            for n in range(self.N):
                for l in range(self.subspace_time_order[j]+1):
                    if j in [0,1]: 
                        shift_idx = int_idx_q 
                    else:
                        shift_idx = int_idx_qstar 
                    dofmaps_j.append( self.fes.sub(j).sub(shift_idx+l).collapse()[1] )
                int_idx_q += (self.q+1)
                int_idx_qstar += (self.qstar+1)
            dofmaps_full.append(dofmaps_j)  
        self.dofmaps_full = dofmaps_full

        self.uh  = fem.Function(self.fes)
        self.uh_best = fem.Function(self.fes)

        self.fes_u1_0 = self.fes.sub(0).sub(0).collapse()[0] 
        self.fes_u2_0 = self.fes.sub(1).sub(0).collapse()[0] 
        self.u1_0 = fem.Function(self.fes_u1_0)
        self.u2_0 = fem.Function(self.fes_u2_0)

        self.fes_z1_0 = self.fes.sub(2).sub(0).collapse()[0] 
        self.fes_z2_0 = self.fes.sub(3).sub(0).collapse()[0] 
        self.z1_0 = fem.Function(self.fes_z1_0)
        self.z2_0 = fem.Function(self.fes_z2_0)


        self.u1_minus = fem.Function(self.fes_u1_0)
        self.u1_slab = [fem.Function(self.fes_u1_0) for j in range(self.q+1)]

        # For strong coupling between slabs 
        vel_coupling = ufl.VectorElement('CG', self.msh.ufl_cell(), self.k, 2)
        mel_coupling = ufl.MixedElement([vel_coupling, vel_coupling ])
        self.fes_coupling = fem.FunctionSpace(self.msh,mel_coupling)
        self.dofmaps_coupling = [ [self.fes_coupling.sub(j).sub(i).collapse()[1] for i in range(2)] for j in range(2) ] 
       
        vel_coupling_u = ufl.VectorElement('CG', self.msh.ufl_cell(), self.k, 2)
        vel_coupling_z = ufl.VectorElement('CG', self.msh.ufl_cell(), self.kstar, 2)
        mel_coupling_uz = ufl.MixedElement([vel_coupling_u, vel_coupling_u, vel_coupling_z,vel_coupling_z])
        self.fes_coupling_uz = fem.FunctionSpace(self.msh,mel_coupling_uz)
        self.dofmaps_coupling_uz = [ [self.fes_coupling_uz.sub(j).sub(i).collapse()[1] for i in range(2)] for j in range(4) ] 
        

        # For mass matrix between slabs 
        vel_slab_bnd = ufl.VectorElement('CG', self.msh.ufl_cell(), self.k, 1)
        mel_slab_bnd = ufl.MixedElement([ vel_slab_bnd, vel_slab_bnd  ]) 
        self.fes_slab_bnd = fem.FunctionSpace( self.msh, mel_slab_bnd )
        self.dofmaps_slab_bnd =  [ self.fes_slab_bnd.sub(i).collapse()[1] for i in range(2) ] 
        self.u_0_slab_bnd = fem.Function( self.fes_slab_bnd )

        
        

        # For slab 
        vel_pre_q = ufl.VectorElement('CG', self.msh.ufl_cell(), self.k, self.q+1)
        vel_pre_qstar = ufl.VectorElement('CG', self.msh.ufl_cell(), self.kstar, self.qstar+1)
        mel_pre = ufl.MixedElement([vel_pre_q,vel_pre_q,vel_pre_qstar,vel_pre_qstar])
        self.fes_p = fem.FunctionSpace(self.msh,mel_pre)
        self.dofmaps_pre = [ [self.fes_p.sub(j).sub(i).collapse()[1] for i in range(self.subspace_time_order[j]+1) ]  for j in range(4) ] 
         
        # for fb-precond
        mel_u = ufl.MixedElement([vel_pre_q,vel_pre_q])
        mel_z = ufl.MixedElement([vel_pre_qstar,vel_pre_qstar])
        self.fesu_slab = fem.FunctionSpace(self.msh,mel_u)
        self.fesz_slab =  fem.FunctionSpace(self.msh,mel_z)
        self.dofmaps_uslab = [ [self.fesu_slab.sub(j).sub(i).collapse()[1] for i in range(self.subspace_time_order[j]+1) ]  for j in range(2) ] 
        #self.dofmaps_zslab = [ [self.fesz_slab.sub(j).sub(i).collapse()[1] for i in range(self.subspace_time_order[j]+1) ]  for j in range(2,4) ] 
        self.dofmaps_zslab = [ [self.fesz_slab.sub(j).sub(i).collapse()[1] for i in range(self.subspace_time_order[j]+1) ]  for j in range(2) ] 
 
        self.w1_p,self.w2_p,self.y1_p,self.y2_p = ufl.TestFunctions(self.fes_p)
        self.fes_u1_0_pre, self.dofmap_fes_u1_0_pre = self.fes_p.sub(0).sub(self.q).collapse() # for top of slice   
        self.fes_u2_0_pre, self.dofmap_fes_u2_0_pre = self.fes_p.sub(1).sub(self.q).collapse() # for top of slice
        self.uh_pre = fem.Function(self.fes_p)
        self.u1_0_pre = fem.Function(self.fes_u1_0_pre)
        self.u2_0_pre = fem.Function(self.fes_u2_0_pre)

        self.only_uh_slab = fem.Function(self.fesu_slab )
        self.only_uh_slab_out = fem.Function(self.fesu_slab )
        self.only_zh_slab = fem.Function(self.fesz_slab )
        
        # for grid transfer 
        fes_node, dofmap_node = self.fes.sub(0).sub(0).collapse() 
        self.u_node_tmp = fem.Function(fes_node)
        self.dofmap_node_tmp = dofmap_node 
        self.tmp_transfer = [ fem.Function(fes_node) for i in range(2*(self.q+1)) ]

    def SetupSpaceTimeMatrix(self,afes,precond=False, coarse_mesh_fits_data_dom = True):
        print("Hello from SetupSpaceTimeMatrix") 
        u1,u2,z1,z2 = ufl.TrialFunctions(afes)
        w1,w2,y1,y2 = ufl.TestFunctions(afes)

        dx = ufl.Measure("dx", domain=afes.mesh, metadata=self.metadata)
        n_facet = ufl.FacetNormal(afes.mesh)
        h = ufl.CellDiameter(afes.mesh)
        
        delta_t = self.delta_t
        elmat_time = self.elmat_time
       
        if afes == self.fes:
            omega_ind = self.omega_ind
        else: # coarse-grid correction
            if self.coarse_mesh_fits_data_dom:
                Q_ind = fem.FunctionSpace(afes.mesh, ("DG", 0)) 
                omega_ind = fem.Function(Q_ind)
                omega_ind.interpolate(self.omega_Ind_coarse)
            else:
                omega_ind = self.omega_Ind_coarse 

        # retrieve stabilization parameter 
        gamma_data = self.stabs["data"]
        gamma_dual = self.stabs["dual"]
        gamma_primal = self.stabs["primal"]
        gamma_primal_jump = self.stabs["primal-jump"]
       
        gamma_primal_jump_vel = self.stabs["primal-jump-vel"]
        gamma_primal_jump_displ_grad = self.stabs["primal-jump-displ-grad"] 


        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1


        a = 1e-20*u1[0]*y1[0]*dx 

        for n in range(self.N):

            t_n = n*delta_t 

            # A[U_h,Y_h] 
            for j in range(self.q+1):
                for k in range(self.qstar+1):
                    a += elmat_time["DM_time_q_qstar"][k,j] * inner(u2[int_idx_q+j] , y1[int_idx_qstar+k]) * dx
                    a += delta_t * self.elmat_time["M_time_q_qstar"][k,j] * inner(self.csq * grad(u1[int_idx_q+j] ), grad(y1[int_idx_qstar+k])) * dx  
                    a += elmat_time["DM_time_q_qstar"][k,j] * inner(u1[int_idx_q+j] , y2[int_idx_qstar+k]) * dx
                    a -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner( u2[int_idx_q+j], y2[int_idx_qstar+k] ) * dx 
                    if abs(elmat_time["M_time_q_qstar"][k,j]) > 1e-10:
                        a -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner(n_facet,self.csq * grad(u1[int_idx_q+j])) * y1[int_idx_qstar+k] * ds  # Nitsche term
                    a += (self.lam_Nitsche / h ) *  delta_t * elmat_time["M_time_q_qstar"][k,j]  * u1[int_idx_q+j] * y1[int_idx_qstar+k] * ds  # Nitsche term
                    # data constraint
                    a +=  delta_t * self.elmat_time["M_time_q_qstar"][k,j] * omega_ind * inner( u1[int_idx_q+j] , y1[int_idx_qstar+k]) * dx  


            # (u_1,w_1) 
            for j in range(self.q+1):
                for k in range(self.q+1):
                    a += gamma_data * delta_t * elmat_time["M_time_q_q"][k,j] * omega_ind * inner( u1[int_idx_q+j], w1[int_idx_q+k] ) * dx 

            # A[W_h,Z_h]
            for j in range(self.q+1):
                for k in range(self.qstar+1):
                    a += (-1)*elmat_time["DMT_time_q_qstar"][k,j] * inner(w2[int_idx_q+j] , z1[int_idx_qstar+k]) * dx
                    a += delta_t * elmat_time["M_time_q_qstar"][k,j] * inner (self.csq *  grad(  w1[int_idx_q+j] ), grad(z1[int_idx_qstar+k])) * dx 
                    a += (-1)*elmat_time["DMT_time_q_qstar"][k,j] * inner(w1[int_idx_q+j] , z2[int_idx_qstar+k]) * dx
                    a -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner( w2[int_idx_q+j], z2[int_idx_qstar+k] ) * dx 
                    if abs(elmat_time["M_time_q_qstar"][k,j]) > 1e-10:
                        a -= delta_t * elmat_time["M_time_q_qstar"][k,j]  * inner(n_facet, self.csq *  grad(w1[int_idx_q+j])) * z1[int_idx_qstar+k] * ds  # Nitsche term
                    a += (self.lam_Nitsche / h ) *  delta_t * elmat_time["M_time_q_qstar"][k,j]  * w1[int_idx_q+j] * z1[int_idx_qstar+k] * ds  # Nitsche term
                    # data constraint
                    a +=  delta_t * self.elmat_time["M_time_q_qstar"][k,j] * omega_ind * inner( w1[int_idx_q+j] , z1[int_idx_qstar+k]) * dx  

            # S*(Y_h,Z_h)
            for j in range(self.qstar+1):
                for k in range(self.qstar+1):
                    a -= gamma_dual * delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(y1[int_idx_qstar+j] , z1[int_idx_qstar+k]) * dx
                    a -= gamma_dual *  delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(grad(y1[int_idx_qstar+j]), grad(z1[int_idx_qstar+k])) * dx
                    a -= gamma_dual * delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(y2[int_idx_qstar+j] , z2[int_idx_qstar + k]) * dx
                    a -= gamma_dual * delta_t * (self.lam_Nitsche / h ) * elmat_time["M_time_qstar_qstar"][k,j] * inner(y1[int_idx_qstar +j] , z1[int_idx_qstar+k]) * ds
            if self.jumps_in_fw_problem and n > 0: 
                a -=  gamma_dual * delta_t * inner(y2[coupling_idx_qstar+1], z2[coupling_idx_qstar+1]  ) * dx
                a -=  gamma_dual * delta_t * inner(y1[coupling_idx_qstar+1], z1[coupling_idx_qstar+1]  ) * dx

            #if n == self.N-1: 
            #    a -= 1e3 * delta_t * inner(y2[coupling_idx_qstar+1], z2[coupling_idx_qstar+1]  ) * dx
            #    a -= 1e3 * delta_t * inner(y1[coupling_idx_qstar+1], z1[coupling_idx_qstar+1]  ) * dx


            # S(U_h,W_h) 
            
            for j in range(self.q+1):
                for k in range(self.q+1):
                    # boundary term
                    a += gamma_primal * delta_t * (self.lam_Nitsche / h ) * elmat_time["M_time_q_q"][k,j] * inner(u1[int_idx_q+j] , w1[int_idx_q+k]) * ds 
                    # J(U_h,W_h)
                    a += gamma_primal * delta_t * elmat_time["M_time_q_q"][k,j] * 0.5*(h('+')+h('-')) * inner( jump(self.csq * grad(u1[int_idx_q+j])) , jump(self.csq * grad(w1[int_idx_q+k])) )  * dS
                    # I_0(U_h,W_h)
                    a += gamma_primal * delta_t * elmat_time["M_time_q_q"][k,j] * inner(u2[int_idx_q+j] , w2[int_idx_q+k]) * dx
                    a -= gamma_primal *  (elmat_time["DM_time_q_q"].T)[k,j] * inner(u2[int_idx_q+j] , w1[int_idx_q+k]) * dx
                    a -= gamma_primal * elmat_time["DM_time_q_q"][k,j] * inner(u1[int_idx_q+j] , w2[int_idx_q+k]) * dx
                    a += gamma_primal * (1/delta_t) * elmat_time["DDM_time_q_q"][k,j] * inner(u1[int_idx_q+j] , w1[int_idx_q+k]) * dx
                    # G(U_h,W_h)
                    a += gamma_primal * h**2 * (1/delta_t) * elmat_time["DDM_time_q_q"][k,j] * inner(u2[int_idx_q+j] , w2[int_idx_q+k]) * dx
                    a -= gamma_primal * h**2 * elmat_time["DM_time_q_q"][k,j] * inner( u2[int_idx_q+j] , div(self.csq * grad(w1[int_idx_q+k]))  ) * dx
                    a -= gamma_primal * h**2 * (elmat_time["DM_time_q_q"].T)[k,j] * inner( div(self.csq * grad(u1[int_idx_q+j])) , w2[int_idx_q+k] ) * dx
                    a += gamma_primal * h**2 * delta_t * elmat_time["M_time_q_q"][k,j] * inner( div(self.csq * grad(u1[int_idx_q+j])) , div( self.csq * grad(w1[int_idx_q+k])) ) * dx

            if n == 0 and self.well_posed:
                print("Adding initial data.")
                # I_1(U_h,W_h)
                a += gamma_primal_jump  * (1/delta_t) * inner(u1[0],w1[0]) * dx + gamma_primal_jump  * delta_t * inner(grad(u1[0]),grad(w1[0]))  * dx
                # I_2(U_h,W_h)
                a += gamma_primal_jump  * (1/delta_t) * inner(u2[0],w2[0]) * dx

            # coupling terms in forward problem 
            if self.jumps_in_fw_problem:  
                coupling_idx_q_top = coupling_idx_q + (self.q+1)
                coupling_idx_qstar_top = coupling_idx_qstar + (self.qstar+1)
                
                if n > 0: 
                    a +=  (1.0+self.eps) * inner(u1[coupling_idx_q+1] - u1[coupling_idx_q], y2[coupling_idx_qstar+1]  ) * dx
                    a +=  (1.0+self.eps) * inner(u2[coupling_idx_q+1] - u2[coupling_idx_q], y1[coupling_idx_qstar+1]  ) * dx

                if n == 0:
                    if not self.well_posed: 
                        a += (-1) * (1.0+self.eps) * inner(w1[coupling_idx_q+1] , z2[coupling_idx_qstar+1]  ) * dx
                        a += (-1) * (1.0+self.eps) * inner(w2[coupling_idx_q+1] , z1[coupling_idx_qstar+1]  ) * dx
                    if self.well_posed: 
                        a +=   (1.0+self.eps) * inner(u1[coupling_idx_q+1] , y2[coupling_idx_qstar+1]  ) * dx 
                        a +=   (1.0+self.eps) * inner(u2[coupling_idx_q+1] , y1[coupling_idx_qstar+1]  ) * dx

                    if self.N-1 != 0:
                        a += (1.0+self.eps) * (-1) * inner( w1[coupling_idx_q_top], z2[coupling_idx_qstar_top+1] - z2[coupling_idx_qstar_top]  ) * dx
                        a += (1.0+self.eps) * (-1) * inner( w2[coupling_idx_q_top], z1[coupling_idx_qstar_top+1] - z1[coupling_idx_qstar_top]  ) * dx
                    else: 
                        a +=   inner( w1[coupling_idx_q_top], z2[coupling_idx_qstar_top]  ) * dx
                        a +=   inner( w2[coupling_idx_q_top], z1[coupling_idx_qstar_top]  ) * dx
                
                elif n == self.N-1: 
                    a +=  (1.0+self.eps) * inner(w1[coupling_idx_q_top], z2[coupling_idx_qstar_top]) * dx
                    a +=  (1.0+self.eps) * inner(w2[coupling_idx_q_top], z1[coupling_idx_qstar_top]) * dx
                    #a +=  (1.0+self.eps) * inner(u1[coupling_idx_q+1] - u1[coupling_idx_q], y2[coupling_idx_qstar+1]  ) * dx
                    #a +=  (1.0+self.eps) * inner(u2[coupling_idx_q+1] - u2[coupling_idx_q], y1[coupling_idx_qstar+1]  ) * dx
                else:
                    a += (1.0+self.eps) * (-1) * inner( w1[coupling_idx_q_top], z2[coupling_idx_qstar_top+1] - z2[coupling_idx_qstar_top]  ) * dx
                    a += (1.0+self.eps) * (-1) * inner( w2[coupling_idx_q_top], z1[coupling_idx_qstar_top+1] - z1[coupling_idx_qstar_top]  ) * dx
                    

            if n > 0:
                if precond:
                    # I_1(U_h,W_h)
                    a += gamma_primal_jump  * (1/delta_t) * inner(u1[coupling_idx_q+1] - u1[coupling_idx_q], w1[coupling_idx_q+1]  ) * dx 
                    a +=  gamma_primal_jump_displ_grad  * delta_t * inner( self.csq * ( grad(u1[coupling_idx_q+1]) - grad(u1[coupling_idx_q]) ), grad(w1[coupling_idx_q+1]))  * dx
                   # I_2(U_h,W_h)
                    a += gamma_primal_jump_vel * (1/delta_t) * inner(  u2[coupling_idx_q+1] - u2[coupling_idx_q] , w2[coupling_idx_q+1] )  * dx
                else: 
                    # I_1(U_h,W_h)
                    a += gamma_primal_jump  * (1/delta_t) * inner(u1[coupling_idx_q+1] - u1[coupling_idx_q], w1[coupling_idx_q+1] - w1[coupling_idx_q] ) * dx 
                    a += gamma_primal_jump_displ_grad  * delta_t * inner( self.csq * ( grad(u1[coupling_idx_q+1]) - grad(u1[coupling_idx_q])), self.csq * ( grad(w1[coupling_idx_q+1]) - grad(w1[coupling_idx_q] ) ))  * dx
                    # I_2(U_h,W_h)
                    a += gamma_primal_jump_vel * (1/delta_t) * inner( u2[coupling_idx_q+1] - u2[coupling_idx_q] , w2[coupling_idx_q+1] - w2[coupling_idx_q] )  * dx

            int_idx_q += (self.q+1)
            coupling_idx_q += (self.q+1)
            int_idx_qstar += (self.qstar+1)
            coupling_idx_qstar += (self.qstar+1)

        print("Creating bfi")
        bilinear_form = fem.form(a)
        print("assemble matrix")
        A = assemble_matrix(bilinear_form, bcs=[])
        A.assemble() 
        
        return A, bilinear_form 


    def SetupSpaceTimeMatrixNoDualStab(self,afes,precond=False, coarse_mesh_fits_data_dom = True):
        print("Hello from SetupSpaceTimeMatrix") 
        u1,u2,z1,z2 = ufl.TrialFunctions(afes)
        w1,w2,y1,y2 = ufl.TestFunctions(afes)

        dx = ufl.Measure("dx", domain=afes.mesh, metadata=self.metadata)
        n_facet = ufl.FacetNormal(afes.mesh)
        h = ufl.CellDiameter(afes.mesh)
        
        delta_t = self.delta_t
        elmat_time = self.elmat_time
       
        if afes == self.fes:
            omega_ind = self.omega_ind
        else: # coarse-grid correction
            if self.coarse_mesh_fits_data_dom:
                Q_ind = fem.FunctionSpace(afes.mesh, ("DG", 0)) 
                omega_ind = fem.Function(Q_ind)
                omega_ind.interpolate(self.omega_Ind_coarse)
            else:
                omega_ind = self.omega_Ind_coarse 

        # retrieve stabilization parameter 
        gamma_data = self.stabs["data"]
        gamma_dual = self.stabs["dual"]
        gamma_primal = self.stabs["primal"]
        gamma_primal_jump = self.stabs["primal-jump"]
       
        gamma_primal_jump_vel = self.stabs["primal-jump-vel"]
        gamma_primal_jump_displ_grad = self.stabs["primal-jump-displ-grad"] 


        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1


        a = 1e-20*u1[0]*y1[0]*dx

        for n in range(self.N):

            t_n = n*delta_t 

            # A[U_h,Y_h] 
            for j in range(self.q+1):
                for k in range(self.qstar+1):
                    a += elmat_time["DM_time_q_qstar"][k,j] * inner(u2[int_idx_q+j] , y1[int_idx_qstar+k]) * dx
                    a += delta_t * self.elmat_time["M_time_q_qstar"][k,j] * inner(self.csq * grad(u1[int_idx_q+j] ), grad(y1[int_idx_qstar+k])) * dx  
                    a += elmat_time["DM_time_q_qstar"][k,j] * inner(u1[int_idx_q+j] , y2[int_idx_qstar+k]) * dx
                    a -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner( u2[int_idx_q+j], y2[int_idx_qstar+k] ) * dx 
                    if abs(elmat_time["M_time_q_qstar"][k,j]) > 1e-10:
                        a -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner(n_facet,self.csq * grad(u1[int_idx_q+j])) * y1[int_idx_qstar+k] * ds  # Nitsche term 
                    a += (self.lam_Nitsche / h ) *  delta_t * elmat_time["M_time_q_qstar"][k,j]  * u1[int_idx_q+j] * y1[int_idx_qstar+k] * ds  # Nitsche term
                    # data constraint
                    a +=  delta_t * self.elmat_time["M_time_q_qstar"][k,j] * omega_ind * inner( u1[int_idx_q+j] , y1[int_idx_qstar+k]) * dx  


            # (u_1,w_1) 
            for j in range(self.q+1):
                for k in range(self.q+1):
                    a += gamma_data * delta_t * elmat_time["M_time_q_q"][k,j] * omega_ind * inner( u1[int_idx_q+j], w1[int_idx_q+k] ) * dx 

            # A[W_h,Z_h]
            for j in range(self.q+1):
                for k in range(self.qstar+1):
                    a += (-1)*elmat_time["DMT_time_q_qstar"][k,j] * inner(w2[int_idx_q+j] , z1[int_idx_qstar+k]) * dx
                    a += delta_t * elmat_time["M_time_q_qstar"][k,j] * inner (self.csq *  grad(  w1[int_idx_q+j] ), grad(z1[int_idx_qstar+k])) * dx 
                    a += (-1)*elmat_time["DMT_time_q_qstar"][k,j] * inner(w1[int_idx_q+j] , z2[int_idx_qstar+k]) * dx
                    a -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner( w2[int_idx_q+j], z2[int_idx_qstar+k] ) * dx 
                    if abs(elmat_time["M_time_q_qstar"][k,j]) > 1e-10:
                        a -= delta_t * elmat_time["M_time_q_qstar"][k,j]  * inner(n_facet, self.csq *  grad(w1[int_idx_q+j])) * z1[int_idx_qstar+k] * ds  # Nitsche term
                    a += (self.lam_Nitsche / h ) *  delta_t * elmat_time["M_time_q_qstar"][k,j] * w1[int_idx_q+j] * z1[int_idx_qstar+k] * ds  # Nitsche term
                    # data constraint
                    a +=  delta_t * self.elmat_time["M_time_q_qstar"][k,j] * omega_ind * inner( w1[int_idx_q+j] , z1[int_idx_qstar+k]) * dx  

            # S*(Y_h,Z_h)
            #if n == 0: 
            #    a += (-1) * 1e-3 * inner(y1[coupling_idx_q+1] , z1[coupling_idx_qstar+1]  ) * dx
            #    a += (-1) * 1e-3 * inner(y2[coupling_idx_q+1] , z2[coupling_idx_qstar+1]  ) * dx
            #if n == 0: 
            #     for j in range(self.qstar+1):
            #         for k in range(self.qstar+1):
            #            a -=   delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(grad(y1[int_idx_qstar+j]), grad(z1[int_idx_qstar+k])) * dx
            #            a -=   delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(y2[int_idx_qstar+j] , z2[int_idx_qstar + k]) * dx

            if True:
            # S*(Y_h,Z_h)
                gamma_dual_red = 1e-15*gamma_dual 
                for j in range(self.qstar+1):
                    for k in range(self.qstar+1):
                        a -= gamma_dual_red * delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(y1[int_idx_qstar+j] , z1[int_idx_qstar+k]) * dx
                        a -= gamma_dual_red *  delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(grad(y1[int_idx_qstar+j]), grad(z1[int_idx_qstar+k])) * dx
                        a -= gamma_dual_red * delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(y2[int_idx_qstar+j] , z2[int_idx_qstar + k]) * dx
                        a -= gamma_dual_red * delta_t * (self.lam_Nitsche / h ) * elmat_time["M_time_qstar_qstar"][k,j] * inner(y1[int_idx_qstar +j] , z1[int_idx_qstar+k]) * ds
                if self.jumps_in_fw_problem and n > 0: 
                    a -=  gamma_dual_red * delta_t * inner(y2[coupling_idx_qstar+1], z2[coupling_idx_qstar+1]  ) * dx
                    a -=  gamma_dual_red * delta_t * inner(y1[coupling_idx_qstar+1], z1[coupling_idx_qstar+1]  ) * dx
            
                
                #if self.jumps_in_fw_problem and n > 0: 
                #    a -=   delta_t * inner(y2[coupling_idx_qstar+1], z2[coupling_idx_qstar+1]  ) * dx
                #    a -=   delta_t * inner(y1[coupling_idx_qstar+1], z1[coupling_idx_qstar+1]  ) * dx
            
                #for j in range(self.qstar+1):
                #    for k in range(self.qstar+1):
                #        a -=  delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(y1[int_idx_qstar+j] , z1[int_idx_qstar+k]) * dx
                #        a -=  delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(y2[int_idx_qstar+j] , z2[int_idx_qstar + k]) * dx
                
            #if n == self.N-1:
                
                #a -= 1e3 * delta_t * inner(y2[coupling_idx_qstar+1], z2[coupling_idx_qstar+1]  ) * dx
                #a -= 1e3 * delta_t * inner(y1[coupling_idx_qstar+1], z1[coupling_idx_qstar+1]  ) * dx
                 

            # S(U_h,W_h) 
            gamma_primal_red = 1
            for j in range(self.q+1):
                for k in range(self.q+1):
                    # boundary term
                    a += gamma_primal_red * gamma_primal * delta_t * (self.lam_Nitsche / h ) * elmat_time["M_time_q_q"][k,j] * inner(u1[int_idx_q+j] , w1[int_idx_q+k]) * ds 
                    # J(U_h,W_h)
                    a += gamma_primal_red * gamma_primal * delta_t * elmat_time["M_time_q_q"][k,j] * 0.5*(h('+')+h('-')) * inner( jump(self.csq * grad(u1[int_idx_q+j])) , jump(self.csq * grad(w1[int_idx_q+k])) )  * dS
                    # I_0(U_h,W_h)
                    a += gamma_primal_red * gamma_primal * delta_t * elmat_time["M_time_q_q"][k,j] * inner(u2[int_idx_q+j] , w2[int_idx_q+k]) * dx
                    a -= gamma_primal_red * gamma_primal *  (elmat_time["DM_time_q_q"].T)[k,j] * inner(u2[int_idx_q+j] , w1[int_idx_q+k]) * dx
                    a -= gamma_primal_red * gamma_primal * elmat_time["DM_time_q_q"][k,j] * inner(u1[int_idx_q+j] , w2[int_idx_q+k]) * dx
                    a += gamma_primal_red * gamma_primal * (1/delta_t) * elmat_time["DDM_time_q_q"][k,j] * inner(u1[int_idx_q+j] , w1[int_idx_q+k]) * dx
                    # G(U_h,W_h)
                    a += gamma_primal_red * gamma_primal * h**2 * (1/delta_t) * elmat_time["DDM_time_q_q"][k,j] * inner(u2[int_idx_q+j] , w2[int_idx_q+k]) * dx
                    a -= gamma_primal_red * gamma_primal * h**2 * elmat_time["DM_time_q_q"][k,j] * inner( u2[int_idx_q+j] , div(self.csq * grad(w1[int_idx_q+k]))  ) * dx
                    a -= gamma_primal_red * gamma_primal * h**2 * (elmat_time["DM_time_q_q"].T)[k,j] * inner( div(self.csq * grad(u1[int_idx_q+j])) , w2[int_idx_q+k] ) * dx
                    a += gamma_primal_red * gamma_primal * h**2 * delta_t * elmat_time["M_time_q_q"][k,j] * inner( div(self.csq * grad(u1[int_idx_q+j])) , div( self.csq * grad(w1[int_idx_q+k])) ) * dx
            

            if n == 0 and self.well_posed:
                print("Adding initial data.")
                # I_1(U_h,W_h)
                a += gamma_primal_jump  * (1/delta_t) * inner(u1[0],w1[0]) * dx + gamma_primal_jump  * delta_t * inner(grad(u1[0]),grad(w1[0]))  * dx
                # I_2(U_h,W_h)
                a += gamma_primal_jump  * (1/delta_t) * inner(u2[0],w2[0]) * dx

            # coupling terms in forward problem 
            if self.jumps_in_fw_problem:  
                coupling_idx_q_top = coupling_idx_q + (self.q+1)
                coupling_idx_qstar_top = coupling_idx_qstar + (self.qstar+1)
                
                if n > 0: 
                    a +=  (1.0+self.eps) * inner(u1[coupling_idx_q+1] - u1[coupling_idx_q], y2[coupling_idx_qstar+1]  ) * dx
                    a +=  (1.0+self.eps) * inner(u2[coupling_idx_q+1] - u2[coupling_idx_q], y1[coupling_idx_qstar+1]  ) * dx

                if n == 0:
                    if not self.well_posed: 
                        a += (-1) * (1.0+self.eps) * inner(w1[coupling_idx_q+1] , z2[coupling_idx_qstar+1]  ) * dx
                        a += (-1) * (1.0+self.eps) * inner(w2[coupling_idx_q+1] , z1[coupling_idx_qstar+1]  ) * dx
                    if self.well_posed: 
                        a +=   (1.0+self.eps) * inner(u1[coupling_idx_q+1] , y2[coupling_idx_qstar+1]  ) * dx
                        a +=   (1.0+self.eps) * inner(u2[coupling_idx_q+1] , y1[coupling_idx_qstar+1]  ) * dx
                    if self.N-1 != 0:
                        a += (1.0+self.eps) * (-1) * inner( w1[coupling_idx_q_top], z2[coupling_idx_qstar_top+1] - z2[coupling_idx_qstar_top]  ) * dx
                        a += (1.0+self.eps) * (-1) * inner( w2[coupling_idx_q_top], z1[coupling_idx_qstar_top+1] - z1[coupling_idx_qstar_top]  ) * dx
                    else: 
                        a +=   inner( w1[coupling_idx_q_top], z2[coupling_idx_qstar_top]  ) * dx
                        a +=   inner( w2[coupling_idx_q_top], z1[coupling_idx_qstar_top]  ) * dx
                
                elif n == self.N-1: 
                    a +=  (1.0+self.eps) * inner(w1[coupling_idx_q_top], z2[coupling_idx_qstar_top]) * dx
                    a +=  (1.0+self.eps) * inner(w2[coupling_idx_q_top], z1[coupling_idx_qstar_top]) * dx
                    #a +=  (1.0+self.eps) * inner(u1[coupling_idx_q+1] - u1[coupling_idx_q], y2[coupling_idx_qstar+1]  ) * dx
                    #a +=  (1.0+self.eps) * inner(u2[coupling_idx_q+1] - u2[coupling_idx_q], y1[coupling_idx_qstar+1]  ) * dx
                else:
                    a += (1.0+self.eps) * (-1) * inner( w1[coupling_idx_q_top], z2[coupling_idx_qstar_top+1] - z2[coupling_idx_qstar_top]  ) * dx
                    a += (1.0+self.eps) * (-1) * inner( w2[coupling_idx_q_top], z1[coupling_idx_qstar_top+1] - z1[coupling_idx_qstar_top]  ) * dx
                    

            if n > 0:
                if precond:
                    # I_1(U_h,W_h)
                    a += gamma_primal_red * gamma_primal_jump  * (1/delta_t) * inner(u1[coupling_idx_q+1] - u1[coupling_idx_q], w1[coupling_idx_q+1]  ) * dx 
                    a +=  gamma_primal_red * gamma_primal_jump_displ_grad  * delta_t * inner( self.csq * ( grad(u1[coupling_idx_q+1]) - grad(u1[coupling_idx_q]) ), grad(w1[coupling_idx_q+1]))  * dx
                   # I_2(U_h,W_h)
                    a += gamma_primal_red * gamma_primal_jump_vel * (1/delta_t) * inner(  u2[coupling_idx_q+1] - u2[coupling_idx_q] , w2[coupling_idx_q+1] )  * dx
                else: 
                    # I_1(U_h,W_h)
                    a += gamma_primal_red * gamma_primal_jump  * (1/delta_t) * inner(u1[coupling_idx_q+1] - u1[coupling_idx_q], w1[coupling_idx_q+1] - w1[coupling_idx_q] ) * dx 
                    a += gamma_primal_red * gamma_primal_jump_displ_grad  * delta_t * inner( self.csq * ( grad(u1[coupling_idx_q+1]) - grad(u1[coupling_idx_q])), self.csq * ( grad(w1[coupling_idx_q+1]) - grad(w1[coupling_idx_q] ) ))  * dx
                    # I_2(U_h,W_h)
                    a += gamma_primal_red * gamma_primal_jump_vel * (1/delta_t) * inner( u2[coupling_idx_q+1] - u2[coupling_idx_q] , w2[coupling_idx_q+1] - w2[coupling_idx_q] )  * dx

            int_idx_q += (self.q+1)
            coupling_idx_q += (self.q+1)
            int_idx_qstar += (self.qstar+1)
            coupling_idx_qstar += (self.qstar+1)

        print("Creating bfi")
        bilinear_form = fem.form(a)
        print("assemble matrix")
        A = assemble_matrix(bilinear_form, bcs=[])
        A.assemble() 
        
        return A, bilinear_form 


    def GetSpaceTimeMatrix(self,precond=False): 
        if not self.SpaceTimeMat:
            A, bilinear_form = self.SetupSpaceTimeMatrix(self.fes,precond=False)
            self.SpaceTimeBfi = bilinear_form
            self.SpaceTimeMat = A   
        return self.SpaceTimeMat   

    def GetSpaceTimeMatrixNoDualStab(self,precond=False): 
        A, bilinear_form = self.SetupSpaceTimeMatrixNoDualStab(self.fes,precond=False)
        return A  

    def GetSpaceTimeBfi(self,precond=False): 
        if not self.SpaceTimeBfi: 
            A, bilinear_form = self.SetupSpaceTimeMatrix(self.fes,precond=False)
            self.SpaceTimeBfi = bilinear_form
            self.SpaceTimeMat = A   
        return self.SpaceTimeBfi 


    def SetupDummyRhs(self):
 
        w1,w2,y1,y2 = ufl.TestFunctions(self.fes)
        dx = self.dx
        L = 1e-20*inner(1.0, y1[0] ) * dx
        linear_form = fem.form(L)
        self.SpaceTimeLfi = linear_form 
        b_rhs = fem.petsc.create_vector(linear_form)
        fem.petsc.assemble_vector(b_rhs, linear_form)
        self.residual = fem.petsc.create_vector(linear_form)
        self.rhs = fem.petsc.create_vector(linear_form)
        self.eps = fem.petsc.create_vector(linear_form)
        self.prolong = fem.petsc.create_vector(linear_form)
        self.x_sol = fem.petsc.create_vector(linear_form)
        self.x_tmp = fem.petsc.create_vector(linear_form)
        self.x_tmp1 = fem.petsc.create_vector(linear_form)
        self.x_tmp2 = fem.petsc.create_vector(linear_form)


    def SetupRhs(self):
        
        w1,w2,y1,y2 = ufl.TestFunctions(self.fes)

        dx = self.dx
        n_facet = self.n_facet
        h = self.h
        delta_t = self.delta_t
        elmat_time = self.elmat_time
        gamma_primal_jump = self.stabs["primal-jump"]
        gamma_data = self.stabs["data"]
        gamma_primal_jump_vel = self.stabs["primal-jump-vel"]
        gamma_primal_jump_displ_grad = self.stabs["primal-jump-displ-grad"] 
        gamma_primal = self.stabs["primal"]



        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1
        
        L = 1e-20*inner(1.0, y1[0] ) * dx

        for n in range(self.N):

            t_n = n*delta_t 

            # right hand side 
            for tau_i,omega_i in zip(self.qr.current_pts(0,1),self.qr.t_weights(1)):
                time_ti = t_n + delta_t*tau_i                 
                sol_ti = self.sol(time_ti,self.x)   
                L += gamma_data * delta_t * omega_i * sol_ti * self.omega_ind * sum([  w1[int_idx_q+k]*self.phi_test[k](tau_i) for k in range(self.q+1) ]) *dx
                L += gamma_primal * delta_t * (self.lam_Nitsche / h ) * omega_i * sol_ti * self.omega_ind * sum([ w1[int_idx_q+k]*self.phi_test[k](tau_i) for k in range(self.q+1) ]) *ds
                # data term in PDE constraint
                L +=  delta_t * omega_i * sol_ti * self.omega_ind * sum([  y1[int_idx_qstar+k]*self.phi_test[k](tau_i) for k in range(self.qstar+1) ]) *dx
                L +=  (self.lam_Nitsche / h ) * delta_t * omega_i * sol_ti * sum([  y1[int_idx_qstar+k]*self.phi_test[k](tau_i) for k in range(self.qstar+1) ]) * ds  # Nitsche term

            if n == 0 and self.well_posed:
                print("Adding initial data on right hand side")
                #L += gamma_primal_jump  * (1/delta_t) * inner(u1_0,w1[0]) * dx + gamma_primal_jump_displ_grad * delta_t * inner(grad(u1_0),grad(w1[0]))  * dx
                #L += gamma_primal_jump_vel * (1/delta_t) * inner(u2_0,w2[0]) * dx
                L += gamma_primal_jump  * (1/delta_t) * inner( self.sol(0,self.x) ,w1[0]) * dx + gamma_primal_jump_displ_grad * delta_t * inner(  grad(self.sol(0,self.x)) ,grad(w1[0]))  * dx
                L += gamma_primal_jump_vel * (1/delta_t) * inner( self.dt_sol(0 ,self.x ),w2[0]) * dx
                L += inner( self.sol(0 ,self.x ),y2[0]) * dx
                L += inner( self.dt_sol(0 ,self.x ),y1[0]) * dx

            int_idx_q += (self.q+1)
            coupling_idx_q += (self.q+1)
            int_idx_qstar += (self.qstar+1)
            coupling_idx_qstar += (self.qstar+1)

        linear_form = fem.form(L)
        b_rhs = fem.petsc.create_vector(linear_form)
        b_tmp  = fem.petsc.create_vector(linear_form)
        fem.petsc.assemble_vector(b_rhs, linear_form)
        
        self.SpaceTimeLfi = linear_form 
        self.SpaceTimeRhs = b_rhs  
   
    def SetupRhsSlabwise(self):
       
        # Create Rhs for space-time system
        w1,w2,y1,y2 = ufl.TestFunctions(self.fes)
        dx = self.dx
        L = 1e-20*inner(1.0, y1[0] ) * dx
        linear_form = fem.form(L)
        b_rhs = fem.petsc.create_vector(linear_form)
        self.SpaceTimeLfi = linear_form 
        # Strategy: Assembly rhs slabwise and then write and big space-time vector

        u1_p,u2_p,z1_p,z2_p = ufl.TrialFunctions(self.fes_p)
        w1_p,w2_p,y1_p,y2_p = ufl.TestFunctions(self.fes_p)

        #dx = self.dx
        n_facet = self.n_facet
        h = self.h
        delta_t = self.delta_t
        elmat_time = self.elmat_time
        gamma_primal_jump = self.stabs["primal-jump"]
        gamma_data = self.stabs["data"]
        gamma_primal_jump_vel = self.stabs["primal-jump-vel"]
        gamma_primal_jump_displ_grad = self.stabs["primal-jump-displ-grad"] 
        gamma_primal = self.stabs["primal"]


        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1
        
        for n in range(self.N):

            L_n = 1e-20*inner(1.0, y1_p[0] ) * dx
            t_n = n*delta_t 

            # right hand side 
            for tau_i,omega_i in zip(self.qr.current_pts(0,1),self.qr.t_weights(1)):
                time_ti = t_n + delta_t*tau_i                 
                sol_ti = self.sol(time_ti,self.x)   
                L_n += gamma_data * delta_t * omega_i * sol_ti * self.omega_ind * sum([  w1_p[k]*self.phi_test[k](tau_i) for k in range(self.q+1) ]) *dx
                L_n += gamma_primal * delta_t * (self.lam_Nitsche / h ) * omega_i * sol_ti * sum([ w1_p[k]*self.phi_test[k](tau_i) for k in range(self.q+1) ]) *ds
                # data term in PDE constraint
                L_n +=  delta_t * omega_i * sol_ti * self.omega_ind * sum([  y1_p[k]*self.phi_test[k](tau_i) for k in range(self.qstar+1) ]) *dx
                L_n +=  (self.lam_Nitsche / h ) * delta_t * omega_i * sol_ti * sum([  y1_p[k]*self.phi_test[k](tau_i) for k in range(self.qstar+1) ]) * ds  # Nitsche term

            if n == 0 and self.well_posed:
                print("Adding initial data on right hand side")
                L_n += gamma_primal_jump  * (1/delta_t) * inner( self.sol(0,self.x) ,w1_p[0]) * dx + gamma_primal_jump_displ_grad * delta_t * inner(  grad(self.sol(0,self.x)) ,grad(w1_p[0]))  * dx
                L_n += gamma_primal_jump_vel * (1/delta_t) * inner( self.dt_sol(0 ,self.x ),w2_p[0]) * dx
                L_n += inner( self.sol(0 ,self.x ),y2_p[0]) * dx
                L_n += inner( self.dt_sol(0 ,self.x ),y1_p[0]) * dx

            linear_form_n = fem.form(L_n)
            b_n = fem.petsc.create_vector(linear_form_n)
            fem.petsc.assemble_vector(b_n, linear_form_n)

            for j in range(4):  
                if j in [0,1]: 
                    shift_idx = int_idx_q 
                else:
                    shift_idx = int_idx_qstar 
                for l in range(self.subspace_time_order[j]+1):
                    b_rhs.array[ self.dofmaps_full[j][shift_idx+l] ] =  b_n.array[self.dofmaps_pre[j][l]] 

            int_idx_q += (self.q+1)
            coupling_idx_q += (self.q+1)
            int_idx_qstar += (self.qstar+1)
            coupling_idx_qstar += (self.qstar+1)

        #linear_form = fem.form(L)
        #b_rhs = fem.petsc.create_vector(linear_form)
        #b_tmp  = fem.petsc.create_vector(linear_form)
        #fem.petsc.assemble_vector(b_rhs, linear_form)
        
        #self.SpaceTimeLfi = linear_form 
        self.SpaceTimeRhs = b_rhs  


    def GetSpaceTimeRhs(self):
        if not self.SpaceTimeRhs:
            self.SetupRhsSlabwise()
            #self.SetupRhs()
        return self.SpaceTimeRhs

    #def GetSpaceTimeLfi(self):
    #    if not self.SpaceTimeLfi: 
    #        self.SetupRhsSlabwise()
    #        #self.SetupRhs()
    #    return self.SpaceTimeLfi


    def SetupCouplingMatrixBetweenSlices(self):
        
        u1_c,u2_c  = ufl.TrialFunctions(self.fes_coupling)
        w1_c,w2_c  = ufl.TestFunctions(self.fes_coupling)
        
        dx = self.dx
        n_facet = self.n_facet
        delta_t = self.delta_t
        gamma_primal_jump = self.stabs["primal-jump"]
        gamma_primal_jump_vel = self.stabs["primal-jump-vel"]
        gamma_primal_jump_displ_grad = self.stabs["primal-jump-displ-grad"] 

        a_coupling = 1e-20 * u1_c[0]* w1_c[0]* dx
        a_coupling += gamma_primal_jump  * (1/delta_t) * inner(u1_c[1] - u1_c[0], w1_c[1] - w1_c[0] ) * dx 
        a_coupling += gamma_primal_jump_displ_grad * delta_t * inner( self.csq *  (grad(u1_c[1]) - grad(u1_c[0])) ,self.csq * (  grad(w1_c[1]) - grad(w1_c[0] )) )  * dx
        a_coupling += gamma_primal_jump_vel * (1/delta_t) * inner( u2_c[1] - u2_c[0] , w2_c[1] - w2_c[0] )  * dx

        bfi_coupling = fem.form(a_coupling)
        A_coupling = assemble_matrix(bfi_coupling, bcs=[])
        A_coupling.assemble()
        
        self.CouplingMatBetweenSlices =  A_coupling
        self.vec_coupling1, self.vec_coupling2 = self.CouplingMatBetweenSlices.createVecs()

    def SetupCouplingMatrixBetweenSlices_uz(self):
        
        u1_c,u2_c,z1_c,z2_c  = ufl.TrialFunctions(self.fes_coupling_uz )
        w1_c,w2_c,y1_c,y2_c  = ufl.TestFunctions(self.fes_coupling_uz  )
        
        dx = self.dx
        n_facet = self.n_facet
        delta_t = self.delta_t

        a_coupling = 1e-20 * u1_c[0] * y1_c[0] * dx        
        a_coupling +=  inner(u1_c[1] - u1_c[0], y2_c[1]  ) * dx 
        a_coupling +=  inner(u2_c[1] - u2_c[0], y1_c[1]  ) * dx 
        a_coupling +=  inner(w2_c[0]  ,  z1_c[0] - z1_c[1] ) * dx 
        a_coupling +=  inner(w1_c[0]  ,  z2_c[0] - z2_c[1] ) * dx 

        bfi_coupling = fem.form(a_coupling)
        A_coupling = assemble_matrix(bfi_coupling, bcs=[])
        A_coupling.assemble()
        
        self.CouplingMatBetweenSlices_uz =  A_coupling
        self.vec_coupling1_uz, self.vec_coupling2_uz = self.CouplingMatBetweenSlices_uz.createVecs()

    def SetupMassMatrixBetweenSlices_u(self):
        
        u0_bnd = ufl.TrialFunctions(self.fes_slab_bnd)
        v0_bnd = ufl.TestFunctions(self.fes_slab_bnd)
        
        dx = self.dx
        n_facet = self.n_facet
        delta_t = self.delta_t

        m_bnd =  inner(u0_bnd[0]  , v0_bnd[1]) * dx 
        m_bnd += inner(u0_bnd[1]  , v0_bnd[0]) * dx

        bfi_m = fem.form(m_bnd)
        M_bnd = assemble_matrix( bfi_m, bcs=[])
        M_bnd.assemble()

        self.MassMatrixBetweenSlices_u =  M_bnd
        self.vec_u_bnd_in, self.vec_u_bnd_out  = self.MassMatrixBetweenSlices_u.createVecs()


    def SetupScaledMassMatrixBetweenSlices(self):
        
        u0_bnd = ufl.TrialFunctions(self.fes_slab_bnd)
        v0_bnd = ufl.TestFunctions(self.fes_slab_bnd)
        
        dx = self.dx
        n_facet = self.n_facet
        delta_t = self.delta_t
        gamma_primal_jump = self.stabs["primal-jump"]
        gamma_primal_jump_vel = self.stabs["primal-jump-vel"]
        gamma_primal_jump_displ_grad = self.stabs["primal-jump-displ-grad"] 

        m_bnd = gamma_primal_jump  * (1/delta_t) * inner(u0_bnd[0] , v0_bnd[0]) * dx 
        m_bnd += gamma_primal_jump_displ_grad * delta_t * inner(self.csq * grad(u0_bnd[0]) ,self.csq *  grad( v0_bnd[0] )  )  * dx
        m_bnd += gamma_primal_jump_vel * (1/delta_t) * inner( u0_bnd[1]  , v0_bnd[1] )  * dx

        bfi_m = fem.form(m_bnd)
        M_bnd = assemble_matrix( bfi_m, bcs=[])
        M_bnd.assemble()

        self.ScaledMassMatrixBetweenSlices =  M_bnd
        self.vec_0_bnd_in, self.vec_0_bnd_out  = self.ScaledMassMatrixBetweenSlices.createVecs()


    def SetupSlabMatrix(self,which_slab="general"): 
    
        u1_p,u2_p,z1_p,z2_p = ufl.TrialFunctions(self.fes_p)
        w1_p,w2_p,y1_p,y2_p = ufl.TestFunctions(self.fes_p)
        
        dx = self.dx
        n_facet = self.n_facet
        h = self.h
        delta_t = self.delta_t
        elmat_time = self.elmat_time
        
        omega_ind = self.omega_ind
        
        # retrieve stabilization parameter 
        gamma_data = self.stabs["data"]
        gamma_dual = self.stabs["dual"]
        gamma_primal = self.stabs["primal"]
        gamma_primal_jump = self.stabs["primal-jump"]
        gamma_primal_jump_vel = self.stabs["primal-jump-vel"]
        gamma_primal_jump_displ_grad = self.stabs["primal-jump-displ-grad"] 
        
        a_pre = 1e-20*u1_p[0]*y1_p[0]*dx 

        # A[U_h,Y_h] 
        for j in range(self.q+1):
            for k in range(self.qstar+1):
                a_pre += elmat_time["DM_time_q_qstar"][k,j] * inner(u2_p[j] , y1_p[k]) * dx
                a_pre += delta_t * self.elmat_time["M_time_q_qstar"][k,j] * inner(self.csq * grad(u1_p[j] ), grad(y1_p[k])) * dx  
                a_pre += elmat_time["DM_time_q_qstar"][k,j] * inner(u1_p[j] , y2_p[k]) * dx
                a_pre -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner( u2_p[j], y2_p[k] ) * dx 
                if abs(elmat_time["M_time_q_qstar"][k,j]) > 1e-10:
                    a_pre -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner(n_facet,self.csq * grad(u1_p[j])) * y1_p[k] * ds  # Nitsche term
                a_pre += (self.lam_Nitsche / h ) *  delta_t * elmat_time["M_time_q_qstar"][k,j]  * u1_p[j] * y1_p[k] * ds  # Nitsche term
                # data constraint
                a_pre +=  delta_t * self.elmat_time["M_time_q_qstar"][k,j] * omega_ind * inner( u1_p[j] , y1_p[k]) * dx  

        # (u_1,w_1) 
        for j in range(self.q+1):
            for k in range(self.q+1):
                a_pre += gamma_data * delta_t *  elmat_time["M_time_q_q"][k,j] * self.omega_ind * inner( u1_p[j], w1_p[k] ) * dx 

        # A[W_h,Z_h]
        for j in range(self.q+1):
            for k in range(self.qstar+1):

                a_pre += (-1)*elmat_time["DMT_time_q_qstar"][k,j] * inner(w2_p[j] , z1_p[k]) * dx
                a_pre += delta_t * elmat_time["M_time_q_qstar"][k,j] * inner (self.csq *  grad(  w1_p[j] ), grad(z1_p[k])) * dx 
                a_pre += (-1)*elmat_time["DMT_time_q_qstar"][k,j] * inner(w1_p[j] , z2_p[k]) * dx
                a_pre -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner( w2_p[j], z2_p[k] ) * dx 
                if abs(elmat_time["M_time_q_qstar"][k,j]) > 1e-10:
                    a_pre -= delta_t * elmat_time["M_time_q_qstar"][k,j]  * inner(n_facet, self.csq *  grad(w1_p[j])) * z1_p[k] * ds  # Nitsche term
                a_pre += (self.lam_Nitsche / h ) *  delta_t * elmat_time["M_time_q_qstar"][k,j]  * w1_p[j] * z1_p[k] * ds  # Nitsche term
                # data constraint
                a_pre +=  delta_t * self.elmat_time["M_time_q_qstar"][k,j] * omega_ind * inner( w1_p[j] , z1_p[k]) * dx  

        # S*(Y_h,Z_h)
        for j in range(self.qstar+1):
            for k in range(self.qstar+1):
                a_pre -= gamma_dual * delta_t * elmat_time["M_time_qstar_qstar"][k,j]  * inner(y1_p[j] , z1_p[k]) * dx
                a_pre -= gamma_dual *  delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(grad(y1_p[j]), grad(z1_p[k])) * dx
                a_pre -= gamma_dual * delta_t * elmat_time["M_time_qstar_qstar"][k,j] * inner(y2_p[j] , z2_p[k]) * dx
                a_pre -= gamma_dual * delta_t * (self.lam_Nitsche / h ) * elmat_time["M_time_qstar_qstar"][k,j]  * inner(y1_p[j] , z1_p[k]) * ds
        
        # S(U_h,W_h) 
        for j in range(self.q+1):
            for k in range(self.q+1):
                # boundary term
                a_pre += gamma_primal * delta_t * (self.lam_Nitsche / h ) * elmat_time["M_time_q_q"][k,j] * inner(u1_p[j] , w1_p[k]) * ds 
                # J(U_h,W_h)
                a_pre += gamma_primal * delta_t * elmat_time["M_time_q_q"][k,j]  * 0.5*(h('+')+h('-'))*inner( jump(self.csq * grad(u1_p[j])) , jump(self.csq * grad(w1_p[k])) )  * dS
                # I_0(U_h,W_h)
                a_pre += gamma_primal * delta_t * elmat_time["M_time_q_q"][k,j] * inner(u2_p[j] , w2_p[k]) * dx
                a_pre -= gamma_primal * (elmat_time["DM_time_q_q"].T)[k,j] * inner(u2_p[j] , w1_p[k]) * dx
                a_pre -= gamma_primal * elmat_time["DM_time_q_q"][k,j] * inner(u1_p[j] , w2_p[k]) * dx
                a_pre += gamma_primal * (1/delta_t) * elmat_time["DDM_time_q_q"][k,j] * inner(u1_p[j] , w1_p[k]) * dx
                # G(U_h,W_h)
                a_pre += gamma_primal * h**2 * (1/delta_t) * elmat_time["DDM_time_q_q"][k,j] * inner(u2_p[j] , w2_p[k]) * dx
                a_pre -= gamma_primal * h**2 * elmat_time["DM_time_q_q"][k,j] * inner( u2_p[j] , div(self.csq *  grad(w1_p[k]))  ) * dx
                a_pre -= gamma_primal * h**2 * (elmat_time["DM_time_q_q"].T)[k,j]  * inner( div( self.csq * grad(u1_p[j])) , w2_p[k] ) * dx
                a_pre += gamma_primal * h**2 * delta_t * elmat_time["M_time_q_q"][k,j] * inner( div(self.csq *  grad(u1_p[j])) , div(self.csq * grad(w1_p[k])) ) * dx


        

        if which_slab == "first":
            if self.well_posed: 
                # I_1(U_h,W_h)
                a_pre += gamma_primal_jump  * (1/delta_t) * inner(u1_p[0],w1_p[0]) * dx + gamma_primal_jump_displ_grad * delta_t * inner(self.csq *  grad(u1_p[0]), self.csq * grad(w1_p[0]))  * dx
                # I_2(U_h,W_h)
                a_pre += gamma_primal_jump_vel * (1/delta_t) * inner(u2_p[0],w2_p[0]) * dx

                if self.jumps_in_fw_problem: 
                    a_pre +=  (1.0+self.eps) * inner(y1_p[0] , u2_p[0]) * dx
                    a_pre +=  (1.0+self.eps) * inner(y2_p[0] , u1_p[0]) * dx
            else:
                a_pre +=  (-1) * inner(w1_p[0], z2_p[0]) * dx
                a_pre +=  (-1) * inner(w2_p[0], z1_p[0]) * dx

            if self.N == 1: 
                a_pre +=  inner(w1_p[self.q], z2_p[self.qstar]) * dx
                a_pre +=  inner(w2_p[self.q], z1_p[self.qstar]) * dx
            
            #a_pre +=  (1.0+self.eps) * inner(w1_p[self.q], z2_p[self.qstar]) * dx
            #a_pre +=  (1.0+self.eps) * inner(w2_p[self.q], z1_p[self.qstar]) * dx

        elif which_slab == "last":
            
            a_pre +=    inner(w1_p[self.q], z2_p[self.qstar]) * dx
            a_pre +=    inner(w2_p[self.q], z1_p[self.qstar]) * dx
            if self.jumps_in_fw_problem: 
                a_pre -= gamma_dual * delta_t * inner(y2_p[0],z2_p[0]) * dx
                a_pre -= gamma_dual * delta_t * inner(y1_p[0],z1_p[0]) * dx
            
        else:
            if self.jumps_in_fw_problem: 
                a_pre -= gamma_dual * delta_t * inner(y2_p[0],z2_p[0]) * dx
                a_pre -= gamma_dual * delta_t * inner(y1_p[0],z1_p[0]) * dx

            # I_1(U_h,W_h)
            #a_pre += gamma_primal_jump  * (1/delta_t) * inner(u1_p[0],w1_p[0]) * dx + gamma_primal_jump_displ_grad * delta_t * inner(self.csq *  grad(u1_p[0]), self.csq * grad(w1_p[0]))  * dx
            # I_2(U_h,W_h)
            #a_pre += gamma_primal_jump_vel * (1/delta_t) * inner(u2_p[0],w2_p[0]) * dx

            #if self.jumps_in_fw_problem: 
            #    a_pre +=  (1.0+self.eps) * inner(y1_p[0] , u2_p[0]) * dx
            #    a_pre +=  (1.0+self.eps) * inner(y2_p[0] , u1_p[0]) * dx
            #
            #    a_pre +=  (1.0+self.eps) * inner(w1_p[self.q], z2_p[self.qstar]) * dx
            #    a_pre +=  (1.0+self.eps) * inner(w2_p[self.q], z1_p[self.qstar]) * dx


        bfi_pre = fem.form(a_pre)
        A_pre = assemble_matrix(bfi_pre, bcs=[])
        A_pre.assemble()
    
        
        if which_slab == "first":
            self.SlabMatFirstSlab = A_pre
        elif which_slab == "last":
            self.SlabMatLastSlab = A_pre 
        else:
            self.SlabBfi = bfi_pre 
            self.SlabMat =  A_pre
            self.vec_slab1, self.vec_slab2 = A_pre.createVecs()


    def SetupSlabMatrix_u(self,with_jumps=True): 
    
        u1_p,u2_p = ufl.TrialFunctions(self.fesu_slab)
        y1_p,y2_p = ufl.TestFunctions(self.fesu_slab)
        
        dx = self.dx
        n_facet = self.n_facet
        h = self.h
        delta_t = self.delta_t
        elmat_time = self.elmat_time
         
        a_pre = 1e-20*u1_p[0]*y1_p[0]*dx 

        # A[U_h,Y_h] 
        for j in range(self.q+1):
            for k in range(self.qstar+1):
                a_pre += elmat_time["DM_time_q_qstar"][k,j] * inner(u2_p[j] , y1_p[k]) * dx
                a_pre += delta_t * elmat_time["M_time_q_qstar"][k,j] * inner(self.csq * grad(u1_p[j] ), grad(y1_p[k])) * dx 
                a_pre += elmat_time["DM_time_q_qstar"][k,j] * inner(u1_p[j] , y2_p[k]) * dx
                a_pre -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner( u2_p[j], y2_p[k] ) * dx 
                #print("M_time_q_qstar{0},{1} = {2}".format(k,j, elmat_time["M_time_q_qstar"][k,j] ))  
                if abs(elmat_time["M_time_q_qstar"][k,j]) > 1e-10:
                    a_pre -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner(n_facet,self.csq * grad(u1_p[j])) * y1_p[k] * ds  # Nitsche term
                a_pre += (self.lam_Nitsche / h ) *  delta_t * elmat_time["M_time_q_qstar"][k,j]  * u1_p[j] * y1_p[k] * ds  # Nitsche term
                a_pre +=  delta_t * self.elmat_time["M_time_q_qstar"][k,j] * self.omega_ind * inner( u1_p[j] , y1_p[k]) * dx          


        if with_jumps == True:
            a_pre +=  inner(u2_p[0] , y1_p[0]) * dx
            a_pre +=  inner(u1_p[0] , y2_p[0]) * dx

        bfi_pre = fem.form(a_pre)
        A_pre = assemble_matrix(bfi_pre, bcs=[])
        A_pre.assemble()

        if with_jumps:
            self.SlabMat_u = A_pre
        else:
            self.SlabMat_u_FirstSlab = A_pre

    def SetupSlabMatrix_z(self,jump_flag="plus"):

        z1_p,z2_p = ufl.TrialFunctions(self.fesz_slab)
        w1_p,w2_p = ufl.TestFunctions(self.fesz_slab)
        
        dx = self.dx
        n_facet = self.n_facet
        h = self.h
        delta_t = self.delta_t
        elmat_time = self.elmat_time

        a_pre = 1e-20*z1_p[0]*w1_p[0]*dx 
        
        for j in range(self.q+1):
            for k in range(self.qstar+1):
                a_pre += (-1)*elmat_time["DMT_time_q_qstar"][k,j] * inner( w2_p[j] , z1_p[k] ) * dx
                a_pre += delta_t * elmat_time["M_time_q_qstar"][k,j] * inner (self.csq *  grad(  w1_p[j] ), grad(z1_p[k])) * dx 
                a_pre += (-1)*elmat_time["DMT_time_q_qstar"][k,j] * inner(w1_p[j] , z2_p[k]) * dx
                a_pre -= delta_t * elmat_time["M_time_q_qstar"][k,j] * inner( w2_p[j], z2_p[k] ) * dx 
                if abs(elmat_time["M_time_q_qstar"][k,j]) > 1e-10:
                    a_pre -= delta_t * elmat_time["M_time_q_qstar"][k,j]  * inner(n_facet, self.csq *  grad(w1_p[j])) * z1_p[k] * ds  # Nitsche term
                a_pre += (self.lam_Nitsche / h ) *  delta_t * elmat_time["M_time_q_qstar"][k,j]  * w1_p[j] * z1_p[k] * ds  # Nitsche term
                # data constraint
                a_pre +=  delta_t * self.elmat_time["M_time_q_qstar"][k,j] * self.omega_ind * inner( w1_p[j] , z1_p[k]) * dx 


        #if jump_flag=="plus":
        a_pre +=  inner(w2_p[self.q] , z1_p[self.qstar]) * dx
        a_pre +=  inner(w1_p[self.q] , z2_p[self.qstar]) * dx
        if jump_flag=="minus":
            a_pre += (-1) * inner(w2_p[0] , z1_p[0]) * dx
            a_pre += (-1) * inner(w1_p[0] , z2_p[0]) * dx
        else: 
            pass

        bfi_pre = fem.form(a_pre)
        A_pre = assemble_matrix(bfi_pre, bcs=[])
        A_pre.assemble()

        if jump_flag=="plus":
            self.SlabMat_z = A_pre
        else: 
            self.SlabMat_z_FirstSlab = A_pre 


    def SetupRhsCorrectionMatZ(self):
       
        dx = self.dx
        n_facet = self.n_facet
        h = self.h
        delta_t = self.delta_t
        elmat_time = self.elmat_time
        
        # retrieve stabilization parameter 
        gamma_data = self.stabs["data"]
        gamma_dual = self.stabs["dual"]
        gamma_primal = self.stabs["primal"]
        gamma_primal_jump = self.stabs["primal-jump"]
        gamma_primal_jump_vel = self.stabs["primal-jump-vel"]
        gamma_primal_jump_displ_grad = self.stabs["primal-jump-displ-grad"] 

        u1_p,u2_p = ufl.TrialFunctions(self.fesu_slab)
        w1_p,w2_p = ufl.TestFunctions(self.fesu_slab)
        
        a_pre = 1e-20*u1_p[0]*w1_p[0]*dx 
        
        for j in range(self.q+1):
            for k in range(self.q+1):
                # boundary term
                a_pre += (-1) * gamma_primal * delta_t * (self.lam_Nitsche / h ) * elmat_time["M_time_q_q"][k,j] * inner(u1_p[j] , w1_p[k]) * ds 
                # J(U_h,W_h)
                a_pre += (-1) * gamma_primal * delta_t * elmat_time["M_time_q_q"][k,j] * 0.5*(h('+')+h('-')) * inner( jump(self.csq * grad(u1_p[j])) , jump(self.csq * grad(w1_p[k])) )  * dS
                # I_0(U_h,W_h)
                a_pre += (-1) * gamma_primal * delta_t * elmat_time["M_time_q_q"][k,j] * inner(u2_p[j] , w2_p[k]) * dx
                a_pre -= (-1) * gamma_primal *  (elmat_time["DM_time_q_q"].T)[k,j] * inner(u2_p[j] , w1_p[k]) * dx
                a_pre -= (-1) *  gamma_primal * elmat_time["DM_time_q_q"][k,j] * inner(u1_p[j] , w2_p[k]) * dx
                a_pre += (-1) * gamma_primal * (1/delta_t) * elmat_time["DDM_time_q_q"][k,j] * inner(u1_p[j] , w1_p[k]) * dx
                # G(U_h,W_h)
                a_pre += (-1) * gamma_primal * h**2 * (1/delta_t) * elmat_time["DDM_time_q_q"][k,j] * inner(u2_p[j] , w2_p[k]) * dx
                a_pre -= (-1) * gamma_primal * h**2 * elmat_time["DM_time_q_q"][k,j] * inner( u2_p[j] , div(self.csq * grad(w1_p[k]))  ) * dx
                a_pre -= (-1) * gamma_primal * h**2 * (elmat_time["DM_time_q_q"].T)[k,j] * inner( div(self.csq * grad(u1_p[j])) , w2_p[k] ) * dx
                a_pre += (-1) * gamma_primal * h**2 * delta_t * elmat_time["M_time_q_q"][k,j] * inner( div(self.csq * grad(u1_p[j])) , div( self.csq * grad(w1_p[k])) ) * dx

        # (u_1,w_1) 
        for j in range(self.q+1):
            for k in range(self.q+1):
                a_pre += (-1) * gamma_data * delta_t *  elmat_time["M_time_q_q"][k,j] * self.omega_ind * inner( u1_p[j], w1_p[k] ) * dx 

        bfi_pre = fem.form(a_pre)
        A_pre = assemble_matrix(bfi_pre, bcs=[])
        A_pre.assemble()

        self.RhsCorrectionMatZ =  A_pre



    def GetSlabBfi(self):
        if not self.SlabBfi:
            self.SetupSlabMatrix(with_jumps=True)
            self.SetupSlabMatrix(with_jumps=False)
        return self.SlabBfi

    def GetSlabMat(self):
        if not self.SlabMat:
            self.SetupSlabMatrix(with_jumps=True)
            self.SetupSlabMatrix(with_jumps=False)
        return self.SlabMat

    def GetSlabBfiFirstSlab(self):
        return self.SlabBfiFirstSlab

    def GetSlabMatFirstSlab(self):
        return self.SlabMatFirstSlab 

    def SetSolverFirstSlab(self,solver):
        self.SolverFirstSlab = solver

    def SetSolverSlab(self,solver):
        self.SolverSlab = solver

    def GetSlabMat_u_FirstSlab(self):
        return self.SlabMat_u_FirstSlab 

    def SetSolverFirstSlab_u(self,solver):
        self.SolverFirstSlab_u = solver 
    
    def GetSlabMat_u(self):
        return self.SlabMat_u 

    def SetSolverSlabMat_u(self,solver):
        self.SolverSlab_u = solver 


    def GetSlabMat_z_FirstSlab(self):
        return self.SlabMat_z_FirstSlab 

    def SetSolverFirstSlab_z(self,solver):
        self.SolverFirstSlab_z = solver 
    
    def GetSlabMat_z(self):
        return self.SlabMat_z

    def SetSolverSlabMat_z(self,solver):
        self.SolverSlab_z = solver 



    def SlabMatNoDGJumps_mult(self,vec_in,vec_out): 
        #if self.mkl_matrix_mult:
        #    mkl_res = sparse_dot_mkl.dot_product_mkl( self.SlabMatNoDGJumps_sp, vec_in.array )
        #    vec_out.array[:] = mkl_res[:] 
        #else:
        self.SlabMatNoDGJumps.mult(vec_in,vec_out)
    
    def CouplingMatBetweenSlices_mult(self,vec_in,vec_out):
        self.CouplingMatBetweenSlices.mult(vec_in,vec_out)

   #def InitialDataMatrix_mult(self,vec_in,vec_out):
   #     self.InitialDataMatrix.mult(vec_in,vec_out)


    def ApplySpaceTimeMatrix(self,vec_in,vec_out):
        
        vec_out.array[:] = 0.0

        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1

        for n in range(self.N):

            for j in range(4):  
                if j in [0,1]: 
                    shift_idx = int_idx_q 
                else:
                    shift_idx = int_idx_qstar 
                for l in range(self.subspace_time_order[j]+1):
                    self.vec_slab1.array[self.dofmaps_pre[j][l]] = vec_in.array[self.dofmaps_full[j][shift_idx+l]]
            
            #self.SlabMatNoDGJumps.mult(self.vec_slab1,self.vec_slab2) # multiplication with slab matrix
            #self.SlabMatNoDGJumps_mult(self.vec_slab1,self.vec_slab2) # multiplication with slab matrix
            if n == 0:
                self.SlabMatFirstSlab.mult(self.vec_slab1,self.vec_slab2) 
                #self.SlabMatNoDGJumps.mult(self.vec_slab1,self.vec_slab2) # multiplication with slab matrix
            elif n == self.N-1:
                self.SlabMatLastSlab.mult(self.vec_slab1,self.vec_slab2) 
            else:
                self.SlabMat.mult(self.vec_slab1,self.vec_slab2) 

            for j in range(4):  
                if j in [0,1]: 
                    shift_idx = int_idx_q 
                else:
                    shift_idx = int_idx_qstar 
                for l in range(self.subspace_time_order[j]+1):
                    vec_out.array[self.dofmaps_full[j][shift_idx+l] ] += self.vec_slab2.array[self.dofmaps_pre[j][l]]                 

            if n > 0: 
                # u - coupling only
                for j in range(2):
                    for i in range(2):
                        self.vec_coupling1.array[ self.dofmaps_coupling[j][i] ] =  vec_in.array[ self.dofmaps_full[j][coupling_idx_q+i ] ]
                
                self.CouplingMatBetweenSlices_mult(self.vec_coupling1, self.vec_coupling2)
                
                for j in range(2):
                    for i in range(2):
                        vec_out.array[ self.dofmaps_full[j][coupling_idx_q+i ] ] += self.vec_coupling2.array[ self.dofmaps_coupling[j][i] ]  

                # uz-coupling
                for j in range(4):
                    for i in range(2):
                        self.vec_coupling1_uz.array[ self.dofmaps_coupling_uz[j][i] ] =  vec_in.array[ self.dofmaps_full[j][coupling_idx_q+i ] ]

                self.CouplingMatBetweenSlices_uz.mult(self.vec_coupling1_uz, self.vec_coupling2_uz)

                for j in range(4):
                    for i in range(2):
                        vec_out.array[ self.dofmaps_full[j][coupling_idx_q+i ] ] += self.vec_coupling2_uz.array[ self.dofmaps_coupling_uz[j][i] ]   

            int_idx_q += (self.q+1)
            coupling_idx_q += (self.q+1)
            int_idx_qstar += (self.qstar+1)
            coupling_idx_qstar += (self.qstar+1)
 
              
    def time_marching_u(self,b,x_sol):
        
        dx = self.dx
        delta_t = self.delta_t
        
        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1
        
        # retrieve stabilization parameter 
        gamma_primal_jump = self.stabs["primal-jump"]

        self.b_rhs_u_slab.array[:]  = 0.0
        
        for n in range(self.N):

            t_n = n*delta_t

            # copy rhs entries from global to local vector on the slice
            for j in range(2): 
                shift_idx = int_idx_qstar 
                for l in range(self.subspace_time_order[j]+1):
                    # Note that we take  self.dofmaps_full[j+2] because the data for the primal equation is with y as test function
                    self.b_rhs_u_slab.array[self.dofmaps_uslab[j][l]] += b.array[self.dofmaps_full[j+2][shift_idx+l]] 
                    #print("self.b_rhs_u_slab.array[self.dofmaps_uslab[j][l]] = ", self.b_rhs_u_slab.array[self.dofmaps_uslab[j][l]]) 

            if n == 0: 
                self.SolverFirstSlab_u.solve(self.b_rhs_u_slab, self.only_uh_slab.vector)
                self.only_uh_slab.x.scatter_forward()
            else:
                self.SolverSlab_u.solve(self.b_rhs_u_slab,self.only_uh_slab.vector)
                self.only_uh_slab.x.scatter_forward()
            
            u1_h_p,u2_h_p = self.only_uh_slab.split()

            self.vec_u_bnd_in.array[:] = 0.0
            self.vec_u_bnd_in.array[self.dofmaps_slab_bnd[0]]  += u1_h_p.x.array[ self.dofmaps_uslab[0][self.q] ]
            self.vec_u_bnd_in.array[self.dofmaps_slab_bnd[1]]  += u2_h_p.x.array[ self.dofmaps_uslab[1][self.q] ]
            #print("self.vec_u_bnd_in.array[self.dofmaps_slab_bnd[0]]   = ", self.vec_u_bnd_in.array[self.dofmaps_slab_bnd[0]] ) 
            #print("self.vec_u_bnd_in.array[self.dofmaps_slab_bnd[1]]   = ", self.vec_u_bnd_in.array[self.dofmaps_slab_bnd[1]] ) 
            self.MassMatrixBetweenSlices_u.mult(self.vec_u_bnd_in, self.vec_u_bnd_out ) 

            self.b_rhs_u_slab.array[:]  = 0.0
            self.b_rhs_u_slab.array[ self.dofmaps_uslab[0][0] ]  +=  self.vec_u_bnd_out.array[self.dofmaps_slab_bnd[0]] 
            self.b_rhs_u_slab.array[ self.dofmaps_uslab[1][0] ]  +=  self.vec_u_bnd_out.array[self.dofmaps_slab_bnd[1]]
            #print(" self.b_rhs_u_slab.array[ self.dofmaps_uslab[0][0] ]  = ", self.b_rhs_u_slab.array[ self.dofmaps_uslab[0][0] ]  )
            #print(" self.b_rhs_u_slab.array[ self.dofmaps_uslab[1][0] ]  = ", self.b_rhs_u_slab.array[ self.dofmaps_uslab[1][0] ]  )

            
            for j in range(2):  
                shift_idx = int_idx_q 
                for l in range(self.subspace_time_order[j]+1):
                    x_sol.array[ self.dofmaps_full[j][shift_idx+l] ] =  self.only_uh_slab.x.array[self.dofmaps_uslab[j][l]] 

            int_idx_q += (self.q+1)
            coupling_idx_q += (self.q+1)
            int_idx_qstar += (self.qstar+1)
            coupling_idx_qstar += (self.qstar+1)

    def backward_sweep_z(self,b,x_sol):
        
        dx = self.dx
        delta_t = self.delta_t
        
        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1
       
        for n in range(self.N-1):
            int_idx_q += (self.q+1)
            coupling_idx_q += (self.q+1)
            int_idx_qstar += (self.qstar+1)
            coupling_idx_qstar += (self.qstar+1)

        # retrieve stabilization parameter 
        gamma_primal_jump = self.stabs["primal-jump"]
        
        self.b_rhs_z_slab.array[:]  = 0.0 

        for n in range(self.N-1,-1,-1):

            t_n = n*delta_t

            # copy rhs entries from global to local vector on the slice
            for j in range(2): 
                shift_idx = int_idx_q 
                for l in range(self.subspace_time_order[j]+1):
                    self.b_rhs_z_slab.array[self.dofmaps_zslab[j][l]] += b.array[self.dofmaps_full[j][shift_idx+l]] 
                    #print("self.b_rhs_z_slab.array[self.dofmaps_zslab[j][l]] = ", self.b_rhs_z_slab.array[self.dofmaps_zslab[j][l]])

            if n == 0: 
                self.SolverFirstSlab_z.solve(self.b_rhs_z_slab, self.only_zh_slab.vector)
                self.only_zh_slab.x.scatter_forward()
            else:
                self.SolverSlab_z.solve(self.b_rhs_z_slab,self.only_zh_slab.vector)
                self.only_zh_slab.x.scatter_forward()
            
            z1_h_p,z2_h_p = self.only_zh_slab.split()

            if n != 0:
                self.vec_u_bnd_in.array[:] = 0.0
                self.vec_u_bnd_in.array[self.dofmaps_slab_bnd[0]]  += z1_h_p.x.array[ self.dofmaps_zslab[0][0] ]
                self.vec_u_bnd_in.array[self.dofmaps_slab_bnd[1]]  += z2_h_p.x.array[ self.dofmaps_zslab[1][0] ]
                self.MassMatrixBetweenSlices_u.mult(self.vec_u_bnd_in, self.vec_u_bnd_out ) 

                self.b_rhs_z_slab.array[:]  = 0.0
                self.b_rhs_z_slab.array[ self.dofmaps_zslab[0][self.qstar] ]  +=  self.vec_u_bnd_out.array[self.dofmaps_slab_bnd[0]] 
                self.b_rhs_z_slab.array[ self.dofmaps_zslab[1][self.qstar] ]  +=  self.vec_u_bnd_out.array[self.dofmaps_slab_bnd[1]]
                #print(" self.b_rhs_z_slab.array[ self.dofmaps_zslab[0][self.qstar] ] = ", self.b_rhs_z_slab.array[ self.dofmaps_zslab[0][self.qstar] ] )
 
            for j in range(2):  
                shift_idx = int_idx_qstar 
                for l in range(self.subspace_time_order[j]+1):
                    x_sol.array[ self.dofmaps_full[j+2][shift_idx+l] ] =  self.only_zh_slab.x.array[self.dofmaps_zslab[j][l]] 

            int_idx_q -= (self.q+1)
            coupling_idx_q -= (self.q+1)
            int_idx_qstar -= (self.qstar+1)
            coupling_idx_qstar -= (self.qstar+1)

    def forward_backward_sweep(self,b,x_sol):
        
        # forward sweep to compute uh
        self.x_fw_sweep.array[:] = 0.0
        self.time_marching_u(b,self.x_fw_sweep)
        # compute corrected rhs for dual equation 
        #self.b_z_corrected.array[:] = b.array[:]
        self.b_z_corrected.array[:] = 0.0
        


        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1
        
        for n in range(self.N):

            self.vec_only_uh_in.array[:] = 0.0
            self.vec_only_uh_out.array[:] = 0.0 
            
            for j in range(2):   
                shift_idx = int_idx_q 
                for l in range(self.subspace_time_order[j]+1):
                    self.vec_only_uh_in.array[self.dofmaps_uslab[j][l]] = self.x_fw_sweep[self.dofmaps_full[j][shift_idx+l]]
                    #print(" self.vec_only_uh_in.array[self.dofmaps_uslab[j][l]] = ", self.vec_only_uh_in.array[self.dofmaps_uslab[j][l]]  )

            self.RhsCorrectionMatZ.mult(self.vec_only_uh_in , self.vec_only_uh_out) 

            for j in range(2):  
                shift_idx = int_idx_q 
                for l in range(self.subspace_time_order[j]+1):
                    self.b_z_corrected.array[self.dofmaps_full[j][shift_idx+l] ] += b.array[self.dofmaps_full[j][shift_idx+l] ] 
                    #print( "b.array[self.dofmaps_full[{0}][shift_idx+{1}] ] = {2}".format(j,l, b.array[self.dofmaps_full[j][shift_idx+l] ] ))

                    self.b_z_corrected.array[self.dofmaps_full[j][shift_idx+l] ] += self.vec_only_uh_out.array[self.dofmaps_uslab[j][l]] 
                    #print("self.b_z_corrected.array[self.dofmaps_full[j][shift_idx+l] ] = ", self.b_z_corrected.array[self.dofmaps_full[j][shift_idx+l] ])  
            
            ## coupling between slices
            #if n > 0 or (n==0 and self.well_posed):
            if n == 0 and self.well_posed: 
                 
                self.vec_0_bnd_in.array[:] = 0.0
                self.vec_0_bnd_in.array[self.dofmaps_slab_bnd[0]]  += self.x_fw_sweep[ self.dofmaps_full[0][0 ] ]
                self.vec_0_bnd_in.array[self.dofmaps_slab_bnd[1]]  += self.x_fw_sweep[ self.dofmaps_full[1][0 ] ]
                self.ScaledMassMatrixBetweenSlices.mult(self.vec_0_bnd_in,self.vec_0_bnd_out)  

                self.b_rhs_pre.array[:]  = 0.0
                self.b_z_corrected.array[ self.dofmaps_full[0][0] ] -=  self.vec_0_bnd_out.array[self.dofmaps_slab_bnd[0]] 
                self.b_z_corrected.array[ self.dofmaps_full[1][0] ] -=  self.vec_0_bnd_out.array[self.dofmaps_slab_bnd[1]] 


            if n > 0:
                for j in range(2):
                    for i in range(2):
                        self.vec_coupling1.array[ self.dofmaps_coupling[j][i] ] =  self.x_fw_sweep[ self.dofmaps_full[j][coupling_idx_q+i ] ]

                self.CouplingMatBetweenSlices_mult(self.vec_coupling1, self.vec_coupling2)

                for j in range(2):
                    for i in range(2):
                        self.b_z_corrected.array[ self.dofmaps_full[j][coupling_idx_q+i ] ] -= self.vec_coupling2.array[ self.dofmaps_coupling[j][i] ]


            int_idx_q += (self.q+1)
            coupling_idx_q += (self.q+1)
            int_idx_qstar += (self.qstar+1)
            coupling_idx_qstar += (self.qstar+1)

        # backsward sweep to compute zh 
        x_sol.array[:] =  self.x_fw_sweep.array[:] 
        self.backward_sweep_z(self.b_z_corrected,x_sol)
        



    def GetSpaceTimeMatrixAsLinearOperator(self):
        return space_time.FMatrix(self)
        
    class FMatrix():
        def __init__(self,st_instance):
            self.st = st_instance
        def mult(self, vec_in , vec_out):
            self.st.ApplySpaceTimeMatrix(vec_in,vec_out)
        def createVecs(self):
            tmp1  = fem.petsc.create_vector(self.st.SpaceTimeLfi)
            tmp2  = fem.petsc.create_vector(self.st.SpaceTimeLfi)
            return tmp1,tmp2


    def PrepareFBsweep(self):
        if not self.SlabMat_u:
            self.SetupSlabMatrix_u(with_jumps=True)
            if not self.well_posed:
                self.SetupSlabMatrix_u(with_jumps=False)
            else: 
                self.SlabMat_u_FirstSlab = self.SlabMat_u 
        self.b_rhs_u_slab, self.b_rhs_z_slab = self.SlabMat_u.createVecs()
        self.SetupMassMatrixBetweenSlices_u() 
        
        if not self.SlabMat_z:
            self.SetupSlabMatrix_z(jump_flag="plus")
            if not self.well_posed:
                self.SetupSlabMatrix_z(jump_flag="minus")
            else: 
                self.SetupSlabMatrix_z(jump_flag="null")

        self.SetupRhsCorrectionMatZ()
        self.x_fw_sweep = fem.petsc.create_vector(self.SpaceTimeLfi)
        self.b_z_corrected = fem.petsc.create_vector(self.SpaceTimeLfi)

        self.vec_only_uh_in, self.vec_only_uh_out = self.RhsCorrectionMatZ.createVecs()

    def PreparePrecondGMRes(self):
        if not self.SlabMat: 
            self.SetupSlabMatrix(which_slab="first")
            self.SetupSlabMatrix(which_slab="last")
            self.SetupSlabMatrix(which_slab="any")
            self.b_rhs_pre, _ = self.SlabMat.createVecs()
        if not self.CouplingMatBetweenSlices:
            self.SetupCouplingMatrixBetweenSlices()  
            self.SetupCouplingMatrixBetweenSlices_uz()  
        if not self.SpaceTimeRhs: 
            #self.SetupRhs()
            self.SetupRhsSlabwise()
        if not self.ScaledMassMatrixBetweenSlices: 
            self.SetupScaledMassMatrixBetweenSlices()
        #if self.mkl_matrix_mult:
        #    self.SlabMatNoDGJumps_sp = GetSpMat(self.SlabMatNoDGJumps)

    def MeasureErrors(self,u_inp,verbose=False,dy_error=False):
        
        err_dict = {} 
        dx = self.dx
        delta_t = self.delta_t
        
        self.uh.x.array[:] = u_inp.array[:]
        self.uh.x.scatter_forward()
        u1_h,u2_h,z1_h,z2_h = self.uh.split()

        self.u1_0.x.array[:] = u1_h.x.array[ self.dofmaps_full[0][-1] ]
        self.u2_0.x.array[:] = u2_h.x.array[ self.dofmaps_full[1][-1] ] 
        ue =  self.sol(self.T,self.x)  
        L2_error = fem.form(  ufl.inner( self.u1_0 - ue, self.u1_0 - ue ) * dx)
        ue_L2norm = fem.form(  ufl.inner( ue, ue ) * dx)
        error_local = fem.assemble_scalar(L2_error)
        ue_L2norm_local = fem.assemble_scalar(ue_L2norm)
        error_L2_abs = np.sqrt(self.msh.comm.allreduce(error_local, op=MPI.SUM))
        ue_L2norm_abs = np.sqrt(self.msh.comm.allreduce(ue_L2norm_local, op=MPI.SUM))
        error_L2_rel = error_L2_abs / ue_L2norm_abs
        
        print("  ue_L2norm_abs =", ue_L2norm_abs ) 
        print("t = {0}, L2-error = {1} ".format(self.T, error_L2_abs  ) )
        print("L2-error = {:.2e} ".format(error_L2_abs  ) )
        print("t = {0}, L2-error rel = {1} ".format(self.T, error_L2_rel  ) )
        err_dict["L2-u1-T"] = error_L2_abs 
        err_dict["L2-u1-T-rel"] = error_L2_rel

        L2_error_at_samples = []
        L2_error_dt_at_samples = []

        int_idx = 0

        total_error_L2L2_dt = 0
        total_error_L2_dt = 0

        for n in range(self.N):

            t_n = n*delta_t 
            if n > 0:
                self.u1_minus.x.array[:] =  u1_h.x.array[self.dofmaps_full[0][int_idx-1]]
            
            for l in range(self.q+1):
                self.u1_slab[l].x.array[:] = u1_h.x.array[ self.dofmaps_full[0][int_idx+l] ]

            # measuring L2-error in time of u_t 
            for tau_i,omega_i in zip(self.qr_ho.current_pts(0,1),self.qr_ho.t_weights(1)):
                time_ti = t_n + delta_t*tau_i 
                dt_ue_tni =  self.dt_sol(time_ti,self.x) 
                uh_prime_at_ti = sum([ (1/delta_t) * self.dt_phi_trial[j]((time_ti - t_n)/delta_t) * self.u1_slab[j] for j in range(self.q+1)  ] )
                if n > 0:
                    uh_prime_at_ti -= (2/delta_t) * ( self.u1_slab[0] -  self.u1_minus  ) * d_theta_ref( ( 2*time_ti - (t_n + delta_t + t_n  ) ) / delta_t )  
                
                L2_error_dt_tni = fem.form(  ufl.inner(  uh_prime_at_ti - dt_ue_tni, uh_prime_at_ti - dt_ue_tni  ) * dx)
                error_dt_tni_local = fem.assemble_scalar(L2_error_dt_tni)
                error_dt_tni = self.msh.comm.allreduce(error_dt_tni_local, op=MPI.SUM)
                total_error_L2L2_dt += omega_i * delta_t * error_dt_tni 

            for t_sample in self.sample_pts_error: 
                if t_sample >= t_n and t_sample <= (t_n + delta_t):
                    ue = self.sol(t_sample,self.x)
                    dt_ue = self.dt_sol(t_sample,self.x)
                    uh_at_ti = sum([  self.phi_trial[j]((t_sample - t_n)/delta_t) * self.u1_slab[j] for j in range(self.q+1)  ] )
                    uh_prime_at_ti = sum([ (1/delta_t) * self.dt_phi_trial[j]((t_sample - t_n)/delta_t) * self.u1_slab[j] for j in range(self.q+1)  ] )
                    if n > 0:
                        uh_at_ti -= theta_ref( ( 2*t_sample - (t_n + delta_t + t_n  ) ) / delta_t ) * ( self.u1_slab[0] -  self.u1_minus  )  
                        uh_prime_at_ti -= (2/delta_t) * ( self.u1_slab[0] -  self.u1_minus  ) * d_theta_ref( ( 2*t_sample - (t_n + delta_t + t_n  ) ) / delta_t )  

                    L2_error = fem.form(  ufl.inner( uh_at_ti - ue, uh_at_ti - ue ) * dx)
                    L2_error_dt = fem.form(  ufl.inner( uh_prime_at_ti - dt_ue, uh_prime_at_ti - dt_ue ) * dx)
                    
                    error_local = fem.assemble_scalar(L2_error)
                    error_local_dt = fem.assemble_scalar(L2_error_dt)
                    error_L2_abs = np.sqrt(self.msh.comm.allreduce(error_local, op=MPI.SUM))
                    error_dt_L2_abs = np.sqrt(self.msh.comm.allreduce(error_local_dt, op=MPI.SUM))
                    L2_error_at_samples.append(error_L2_abs)
                    L2_error_dt_at_samples.append(error_dt_L2_abs)
                    if verbose:
                        print("t = {0}, L2-error u = {1}, L2-error u_t = {2} ".format(t_sample, error_L2_abs, error_dt_L2_abs) ) 

            int_idx += (self.q+1)
        
        #compute dy error 
        if dy_error:
            dy_uh = grad(self.u1_0)[1]
            dy_ue = grad(ue)[1]
            dy_error = fem.form(  ufl.inner( dy_uh- dy_ue, dy_uh- dy_ue ) * dx)
            dy_error_local = fem.assemble_scalar(dy_error)
            error_dy_abs = np.sqrt(self.msh.comm.allreduce(dy_error_local, op=MPI.SUM))
            print("t = {0}, dy-L2-error = {1} ".format(self.T, error_dy_abs  ) )
            err_dict["dy-L2-u1-T"] = error_dy_abs



        print("L-infty-L2 error u = ", max( L2_error_at_samples ) )
        print("L-infty-L2 error u_t = ", max( L2_error_dt_at_samples ) )
        print("L2-L2 error u_t = ", sqrt( total_error_L2L2_dt ) )
        
        err_dict["L-infty-L2-error-u"] = max( L2_error_at_samples ) 
        err_dict["L-infty-L2-error-ut"] = max( L2_error_dt_at_samples )  
        err_dict["L2-L2-error-u_t"]  = sqrt( total_error_L2L2_dt ) 

        return err_dict 

    def MeasureL2Error(self,u_inp,verbose=False): 
        err_dict = {}
        dx = self.dx
        delta_t = self.delta_t
        
        self.uh.x.array[:] = u_inp.array[:]
        self.uh.x.scatter_forward()
        u1_h,u2_h,z1_h,z2_h = self.uh.split()

        self.u1_0.x.array[:] = u1_h.x.array[ self.dofmaps_full[0][-1] ]
        self.u2_0.x.array[:] = u2_h.x.array[ self.dofmaps_full[1][-1] ] 
        ue =  self.sol(self.T,self.x)  
        L2_error = fem.form(  ufl.inner( self.u1_0 - ue, self.u1_0 - ue ) * dx)
        error_local = fem.assemble_scalar(L2_error)
        error_L2_abs = np.sqrt(self.msh.comm.allreduce(error_local, op=MPI.SUM))
        print("t = {0}, L2-error = {1} ".format(self.T, error_L2_abs  ) )
        err_dict["L2-u1-T"] = error_L2_abs
        return err_dict

    def MeasureErrorsRestrict(self,u_inp,verbose=False,data_size=0.25):
        
        # increase quadrature points to account for intrgration of discontinuous integrand 
        dxR = ufl.Measure("dx", domain=self.msh, metadata=  {"quadrature_degree": 2*self.k+10 } ) 

        err_dict = {} 
        dx = self.dx
        delta_t = self.delta_t
        
        self.uh.x.array[:] = u_inp.array[:]
        self.uh.x.scatter_forward()
        u1_h,u2_h,z1_h,z2_h = self.uh.split()


        gcc_indicator = ufl.And(self.x[0] <= 3*data_size - self.T, self.x[0] >= 0.0)
        gcc_Ind = ufl.conditional(gcc_indicator, 1, 0)

        self.u1_0.x.array[:] = u1_h.x.array[ self.dofmaps_full[0][-1] ]
        self.u2_0.x.array[:] = u2_h.x.array[ self.dofmaps_full[1][-1] ] 
        ue =  self.sol(self.T,self.x)  
        

        L2_error = fem.form(  gcc_Ind * ufl.inner( self.u1_0 - ue, self.u1_0 - ue ) * dxR)
        error_local = fem.assemble_scalar(L2_error)
        error_L2_abs = np.sqrt(self.msh.comm.allreduce(error_local, op=MPI.SUM))
        print("t = {0}, L2-error = {1} ".format(self.T, error_L2_abs  ) )
        err_dict["L2-u1-T"] = error_L2_abs 

        L2_error_at_samples = []
        L2_error_dt_at_samples = []

        int_idx = 0

        total_error_L2L2_dt = 0
        total_error_L2_dt = 0

        for n in range(self.N):

            t_n = n*delta_t 
            #if t_n <= data_size:
            #    gcc_indicator = ufl.And(self.x[0] <= data_size + t_n, self.x[0] >= 0.0)
            #    gcc_Ind = ufl.conditional(gcc_indicator, 1, 0)
            #else:
            #    gcc_indicator = ufl.And(self.x[0] <= 3*data_size - t_n, self.x[0] >= 0.0)
            #    gcc_Ind = ufl.conditional(gcc_indicator, 1, 0)


            if n > 0:
                self.u1_minus.x.array[:] =  u1_h.x.array[self.dofmaps_full[0][int_idx-1]]
            
            for l in range(self.q+1):
                self.u1_slab[l].x.array[:] = u1_h.x.array[ self.dofmaps_full[0][int_idx+l] ]

            # measuring L2-error in time of u_t 
            for tau_i,omega_i in zip(self.qr_ho.current_pts(0,1),self.qr_ho.t_weights(1)):
                time_ti = t_n + delta_t*tau_i 

                if  time_ti <= data_size:
                    gcc_indicator = ufl.And(self.x[0] <= data_size + time_ti , self.x[0] >= 0.0)
                    gcc_Ind = ufl.conditional(gcc_indicator, 1, 0)
                else:
                    gcc_indicator = ufl.And(self.x[0] <= 3*data_size - time_ti , self.x[0] >= 0.0)
                    gcc_Ind = ufl.conditional(gcc_indicator, 1, 0)

                dt_ue_tni =  self.dt_sol(time_ti,self.x) 
                uh_prime_at_ti = sum([ (1/delta_t) * self.dt_phi_trial[j]((time_ti - t_n)/delta_t) * self.u1_slab[j] for j in range(self.q+1)  ] )
                if n > 0:
                    uh_prime_at_ti -= (2/delta_t) * ( self.u1_slab[0] -  self.u1_minus  ) * d_theta_ref( ( 2*time_ti - (t_n + delta_t + t_n  ) ) / delta_t )  
                
                L2_error_dt_tni = fem.form(  gcc_Ind * ufl.inner(  uh_prime_at_ti - dt_ue_tni, uh_prime_at_ti - dt_ue_tni  ) * dxR )
                error_dt_tni_local = fem.assemble_scalar(L2_error_dt_tni)
                error_dt_tni = self.msh.comm.allreduce(error_dt_tni_local, op=MPI.SUM)
                total_error_L2L2_dt += omega_i * delta_t * error_dt_tni 

            for t_sample in self.sample_pts_error: 
                if t_sample >= t_n and t_sample <= (t_n + delta_t):

                    if  t_sample <= data_size:

                        gcc_indicator = ufl.And(self.x[0] <= data_size + t_sample, self.x[0] >= 0.0)
                        gcc_Ind = ufl.conditional(gcc_indicator, 1, 0)
                    else:
                        gcc_indicator = ufl.And(self.x[0] <= 3*data_size - t_sample, self.x[0] >= 0.0)
                        gcc_Ind = ufl.conditional(gcc_indicator, 1, 0)

                    ue = self.sol(t_sample,self.x)
                    dt_ue = self.dt_sol(t_sample,self.x)
                    uh_at_ti = sum([  self.phi_trial[j]((t_sample - t_n)/delta_t) * self.u1_slab[j] for j in range(self.q+1)  ] )
                    uh_prime_at_ti = sum([ (1/delta_t) * self.dt_phi_trial[j]((t_sample - t_n)/delta_t) * self.u1_slab[j] for j in range(self.q+1)  ] )
                    if n > 0:
                        uh_at_ti -= theta_ref( ( 2*t_sample - (t_n + delta_t + t_n  ) ) / delta_t ) * ( self.u1_slab[0] -  self.u1_minus  )  
                        uh_prime_at_ti -= (2/delta_t) * ( self.u1_slab[0] -  self.u1_minus  ) * d_theta_ref( ( 2*t_sample - (t_n + delta_t + t_n  ) ) / delta_t )  

                    L2_error = fem.form( gcc_Ind * ufl.inner( uh_at_ti - ue, uh_at_ti - ue ) * dxR )
                    L2_error_dt = fem.form( gcc_Ind * ufl.inner( uh_prime_at_ti - dt_ue, uh_prime_at_ti - dt_ue ) * dxR )
                    
                    error_local = fem.assemble_scalar(L2_error)
                    error_local_dt = fem.assemble_scalar(L2_error_dt)
                    error_L2_abs = np.sqrt(self.msh.comm.allreduce(error_local, op=MPI.SUM))
                    error_dt_L2_abs = np.sqrt(self.msh.comm.allreduce(error_local_dt, op=MPI.SUM))
                    L2_error_at_samples.append(error_L2_abs)
                    L2_error_dt_at_samples.append(error_dt_L2_abs)
                    if verbose:
                        print("t = {0}, L2-error u = {1}, L2-error u_t = {2} ".format(t_sample, error_L2_abs, error_dt_L2_abs) ) 

            int_idx += (self.q+1)
        
        print("L-infty-L2 error u = ", max( L2_error_at_samples ) )
        print("L-infty-L2 error u_t = ", max( L2_error_dt_at_samples ) )
        print("L2-L2 error u_t = ", sqrt( total_error_L2L2_dt ) )
        
        err_dict["L-infty-L2-error-u"] = max( L2_error_at_samples ) 
        err_dict["L-infty-L2-error-ut"] = max( L2_error_dt_at_samples )  
        err_dict["L2-L2-error-u_t"]  = sqrt( total_error_L2L2_dt ) 

        return err_dict 




    
    def Plot(self,uh_inp,N_space=100,N_time_subdiv=10,abs_val=False,return_data=False):
        
        if abs_val:
            trafo = lambda x: abs(x)
        else:
            trafo = lambda x: x

        uh_plot  = fem.Function(self.fes)
        uh_plot.x.array[:] = uh_inp.array[:]
        u1_h,u2_h,z1_h,z2_h = uh_plot.split()
        
        u1_slab_plot = [fem.Function(self.fes_u1_0) for j in range(self.q+1)]
        z1_slab_plot = [fem.Function(self.fes_z1_0) for j in range(self.qstar+1)]

        delta_t = self.delta_t
        right_end = 1.0
        left_end = 0.0
        eval_pts_space = np.linspace(left_end,right_end,num=N_space).tolist()
        eval_pts_space_3d = np.array([ (xp ,0,0) for xp in  eval_pts_space] ) 

        eval_t_subdiv = np.linspace(0.0,1.0,num=N_time_subdiv,endpoint=False).tolist()

        ts = [ ]

        int_idx = 0
        time_idx = 0
           
        fun_val = np.zeros(( N_time_subdiv*self.N, N_space ))
        dt_fun_val = np.zeros(( N_time_subdiv*self.N, N_space ))
        space_val_ti = np.zeros((self.q+1,N_space))
        
        for n in range(self.N):
            t_n = n*delta_t

            for l in range(self.q+1):
                u1_slab_plot[l].x.array[:] = u1_h.x.array[ self.dofmaps_full[0][int_idx+l] ]

            for l in range(self.q+1):
                space_val_ti[l,:] = evaluate_function_at_points( u1_slab_plot[l] , eval_pts_space_3d )  

            # measuring L2-error in time of u_t 
            for j in range(N_space):
                for m,tau_m in enumerate(eval_t_subdiv):
                    time_ti = t_n + delta_t*tau_m 
                    if j == 0:
                        ts.append( time_ti)
                    uh_at_ti = trafo( sum([  self.phi_trial[l]((time_ti - t_n)/delta_t) * space_val_ti[l,j] for l in range(self.q+1)  ] ) )
                    dt_uh_at_ti = trafo( sum( [ (1/delta_t) * self.dt_phi_trial[l]((time_ti - t_n)/delta_t) * space_val_ti[l,j] for l in range(self.q+1)  ] ))
                    fun_val[time_idx+m,j] = uh_at_ti 
                    dt_fun_val[time_idx+m,j] = dt_uh_at_ti 

            int_idx += (self.q+1)
            time_idx += N_time_subdiv

        
        fun_val_z = np.zeros(( N_time_subdiv*self.N, N_space ))
        dt_fun_val_z = np.zeros(( N_time_subdiv*self.N, N_space ))
        space_val_ti_z = np.zeros((self.qstar+1,N_space))

        int_idx = 0
        time_idx = 0

        for n in range(self.N):
            t_n = n*delta_t

            for l in range(self.qstar+1):
                z1_slab_plot[l].x.array[:] = z1_h.x.array[ self.dofmaps_full[2][int_idx+l] ]

            for l in range(self.qstar+1):
                space_val_ti_z[l,:] = evaluate_function_at_points( z1_slab_plot[l] , eval_pts_space_3d )  

            # measuring L2-error in time of u_t 
            for j in range(N_space):
                for m,tau_m in enumerate(eval_t_subdiv):
                    time_ti = t_n + delta_t*tau_m 
                    zh_at_ti = trafo( sum([  self.phi_trial[l]((time_ti - t_n)/delta_t) * space_val_ti_z[l,j] for l in range(self.qstar+1)  ] ) )
                    dt_zh_at_ti = trafo( sum( [ (1/delta_t) * self.dt_phi_trial[l]((time_ti - t_n)/delta_t) * space_val_ti_z[l,j] for l in range(self.qstar+1)  ] ))
                    fun_val_z[time_idx+m,j] = zh_at_ti 
                    dt_fun_val_z[time_idx+m,j] = dt_zh_at_ti 

            int_idx += (self.qstar+1)
            time_idx += N_time_subdiv
        
        
        if return_data:
            data = [eval_pts_space,ts,fun_val,dt_fun_val]
            #np.savetxt("../data/eval_pts_space.csv", eval_pts_space, delimiter=",")
            #np.savetxt("../data/ts.csv", ts, delimiter=",")
            #np.savetxt("../data/fun_val.csv", fun_val, delimiter=",")
            #np.savetxt("../data/dt_fun_val.csv", dt_fun_val, delimiter=",")
            #print('Data saved!')
            return data 
        else:
            if False:
                vmin = np.min( fun_val )*1e1
                vmax = np.max( fun_val )*1e1

                im = plt.pcolormesh(eval_pts_space,ts, fun_val,
                            cmap=plt.get_cmap('hot'),
                            norm=colors.LogNorm( vmin= vmin  ,vmax= vmax ) 
                            )
            elif False:
                cmap = plt.get_cmap('PiYG')
                #levels = MaxNLocator(nbins=15).tick_values(1e4*fun_val.min(), 1e-1*fun_val.max())
                levels = MaxNLocator(nbins=15).tick_values(fun_val.min(), fun_val.max())
                norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
                im = plt.pcolormesh(eval_pts_space,ts, fun_val,cmap=cmap,norm=norm)
                plt.colorbar(im)
                plt.xlabel("x")
                plt.ylabel("t")
                plt.show()

            else:
                fig = plt.figure()
                #ax = fig.gca(projection='3d')
                ax = fig.add_subplot(projection='3d')
                xx, tt = np.meshgrid(eval_pts_space, ts)
                
                #print("xx.shape = ", xx.shape)
                #print("tt.shape = ", tt.shape)
                #print("fun_val.shape = ", fun_val.shape )

                surf = ax.plot_surface(xx, tt, fun_val, cmap=cm.viridis,
                                    linewidth=0, antialiased=False )
                # Customize the z axis.
                ax.set_zlim(  fun_val.min() ,  fun_val.max() )
                ax.zaxis.set_major_locator(LinearLocator(5))
                #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                ax.zaxis.set_major_formatter(NullFormatter())

                # Add a color bar which maps values to colors.
                ax.set_xlabel("x")
                ax.set_ylabel("t")
                fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.savefig("height-map",transparent=True)
                plt.show()
                
                idx = 80
                print("ts[ .. ] = ", ts[ int(len(ts)/2) ] )
                plt.plot( eval_pts_space, fun_val[ int(len(ts)/2) ,: ] )
                plt.show()
                  

                ####  for z 
                fig = plt.figure()
                #ax = fig.gca(projection='3d')
                ax = fig.add_subplot(projection='3d')
                xx, tt = np.meshgrid(eval_pts_space, ts)
                
                #print("xx.shape = ", xx.shape)
                #print("tt.shape = ", tt.shape)
                #print("fun_val.shape = ", fun_val.shape )

                surf = ax.plot_surface(xx, tt, fun_val_z , cmap=cm.viridis,
                                    linewidth=0, antialiased=False )
                # Customize the z axis.
                ax.set_zlim(  fun_val_z.min() ,  fun_val_z.max() )
                ax.zaxis.set_major_locator(LinearLocator(5))
                #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                ax.zaxis.set_major_formatter(NullFormatter())

                # Add a color bar which maps values to colors.
                ax.set_xlabel("x")
                ax.set_ylabel("t")
                fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.savefig("height-map",transparent=True)
                plt.show()
                
                idx = 80
                print("ts[ .. ] = ", ts[ int(len(ts)/2) ] )
                plt.plot( eval_pts_space, fun_val_z[ int(len(ts)/2) ,: ] )
                plt.show()



        if False:
            vmin = np.min( fun_val )*1e2
            vmax = np.max( fun_val )*1e2
            #vmin = 1e-4
            #vmax = 1e-1
            im = plt.pcolormesh(eval_pts_space,ts, dt_fun_val,
                           cmap=plt.get_cmap('hot'),
                           norm=colors.LogNorm( vmin= vmin  ,vmax= vmax ) 
                          )
        else:
            #cmap = plt.get_cmap('PiYG')
            cmap = plt.get_cmap('jet')
            levels = MaxNLocator(nbins=15).tick_values(dt_fun_val.min(), dt_fun_val.max())
            norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            im = plt.pcolormesh(eval_pts_space,ts, dt_fun_val,cmap=cmap,norm=norm)
            #im = plt.pcolormesh(eval_pts_space,ts, dt_fun_val)
        plt.colorbar(im)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.show()

    def PlotError(self,uh_inp,N_space=100,N_time_subdiv=10,return_data=False):
        
        uh_plot  = fem.Function(self.fes)
        uh_plot.x.array[:] = uh_inp.array[:]
        u1_h,u2_h,z1_h,z2_h = uh_plot.split()
        
        u1_slab_plot = [fem.Function(self.fes_u1_0) for j in range(self.q+1)]

        delta_t = self.delta_t
        right_end = 1.0
        left_end = 0.0
        eval_pts_space = np.linspace(left_end,right_end,num=N_space).tolist()
        eval_pts_space_3d = np.array([ (xp ,0,0) for xp in  eval_pts_space] ) 

        eval_t_subdiv = np.linspace(0.0,1.0,num=N_time_subdiv,endpoint=False).tolist()

        ts = [ ]

        int_idx = 0
        time_idx = 0
           
        fun_val = np.zeros(( N_time_subdiv*self.N, N_space ))
        dt_fun_val = np.zeros(( N_time_subdiv*self.N, N_space ))
        err_val = np.zeros(( N_time_subdiv*self.N, N_space ))
        dt_err_val = np.zeros(( N_time_subdiv*self.N, N_space ))
        space_val_ti = np.zeros((self.q+1,N_space))


        
        for n in range(self.N):
            t_n = n*delta_t

            for l in range(self.q+1):
                u1_slab_plot[l].x.array[:] = u1_h.x.array[ self.dofmaps_full[0][int_idx+l] ]

            for l in range(self.q+1):
                space_val_ti[l,:] = evaluate_function_at_points( u1_slab_plot[l] , eval_pts_space_3d )  

            # measuring L2-error in time of u_t 
            for j in range(N_space):
                for m,tau_m in enumerate(eval_t_subdiv):
                    time_ti = t_n + delta_t*tau_m 
                    if j == 0:
                        ts.append( time_ti)
                    uh_at_ti = sum([  self.phi_trial[l]((time_ti - t_n)/delta_t) * space_val_ti[l,j] for l in range(self.q+1)  ] )
                    u_at_ti = self.sol( time_ti , [eval_pts_space[j]] )
                    
                    dt_uh_at_ti = sum( [ (1/delta_t) * self.dt_phi_trial[l]((time_ti - t_n)/delta_t) * space_val_ti[l,j] for l in range(self.q+1)  ] )
                    dt_u_at_ti = self.dt_sol( time_ti , [eval_pts_space[j]] )

                    fun_val[time_idx+m,j] = uh_at_ti 
                    err_val[time_idx+m,j] = abs( uh_at_ti - u_at_ti) 
                    
                    dt_fun_val[time_idx+m,j] = dt_uh_at_ti 
                    dt_err_val[time_idx+m,j] = abs( dt_uh_at_ti - dt_u_at_ti ) 


            int_idx += (self.q+1)
            time_idx += N_time_subdiv

        if return_data:
            data = [eval_pts_space,ts,err_val,dt_err_val]
            return data 
        else: 
            #plt.pcolormesh(ts, eval_pts_space, fun_val )
            #plt.pcolormesh(eval_pts_space,ts, fun_val )
            plt.pcolormesh(eval_pts_space,ts, dt_fun_val  )
            plt.show()
            
            vmin = 1e-5
            vmax = 1e-1

            SMALL_SIZE = 8
            MEDIUM_SIZE = 13
            BIGGER_SIZE = 15

            plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            im = plt.pcolormesh(eval_pts_space,ts, err_val, 
                        cmap=plt.get_cmap('hot'),
                        norm=colors.LogNorm( vmin= vmin  ,vmax= vmax ) 
                        )

            plt.colorbar(im)
            #plt.xlabel("$x$")
            #plt.ylabel("$t$")
            plt.title("$|u(t,x) - u_h(t,x)|$")
            
            plt.savefig("abs-err",transparent=True)
            plt.show()

            vmin = 1e-4
            vmax = 1e-1
            #vmin = 1e-6
            #vmax = 1e-2

            im = plt.pcolormesh(eval_pts_space,ts, dt_err_val, 
                        cmap=plt.get_cmap('hot'),
                        norm=colors.LogNorm( vmin= vmin  ,vmax= vmax ) 
                        )
            plt.colorbar(im)
            plt.xlabel("x")
            plt.ylabel("t")

            plt.show()


    def PlotParaview(self,uh_inp,name="abserr"):
        
        self.uh.x.array[:] = uh_inp.array[:]
        self.uh.x.scatter_forward()
        u1_h,u2_h,z1_h,z2_h = self.uh.split()
        self.u1_0.x.array[:] = u1_h.x.array[ self.dofmaps_full[0][0] ]
        self.u2_0.x.array[:] = u2_h.x.array[ self.dofmaps_full[1][0] ]

        u_exact = fem.Function(self.fes_u1_0)
        u_diff = fem.Function(self.fes_u1_0)

        ut_exact = fem.Function(self.fes_u2_0)
        ut_diff = fem.Function(self.fes_u2_0)

        t0 = 0
        ue =  self.sol(t0,self.x)  
        dt_ue =  self.dt_sol(t0,self.x)  
        
        if isinstance(ue, ufl.classes.Zero):
            u_diff.x.array[:] = np.abs( self.u1_0.x.array[:] )
        else:
            u_expr_exact = fem.Expression(ue, self.fes_u1_0.element.interpolation_points() )
            u_exact.interpolate(u_expr_exact) 
            u_diff.x.array[:] = np.abs( u_exact.x.array[:] - self.u1_0.x.array[:] )
        
        if isinstance(dt_ue, ufl.classes.Zero):
            ut_diff.x.array[:] = np.abs(self.u2_0.x.array[:] )
        else:
            ut_expr_exact = fem.Expression(dt_ue, self.fes_u2_0.element.interpolation_points() )
            ut_exact.interpolate(ut_expr_exact) 
            ut_diff.x.array[:] = np.abs( ut_exact.x.array[:] - self.u2_0.x.array[:] )
        #print(" u_diff.x.array =",u_diff.x.array )

        u_diff.name ="u-abserr"
        ut_diff.name ="ut-abserr"
        with io.XDMFFile(self.msh.comm,"{0}-u.xdmf".format(name), "w") as xdmf:
            u1 = fem.Function(fem.FunctionSpace(self.msh,('Lagrange',1)))
            u1.interpolate(u_diff)
            xdmf.write_mesh(self.msh)
            xdmf.write_function(u1)
        with io.XDMFFile(self.msh.comm,"{0}-ut.xdmf".format(name), "w") as xdmf:
            ut1 = fem.Function(fem.FunctionSpace(self.msh,('Lagrange',1)))
            ut1.interpolate(ut_diff)
            xdmf.write_mesh(self.msh)
            xdmf.write_function(ut1)
        with io.XDMFFile(self.msh.comm,"{0}-exact.xdmf".format(name), "w") as xdmf:
            u1 = fem.Function(fem.FunctionSpace(self.msh,('Lagrange',1)))
            u1.interpolate(u_exact)
            xdmf.write_mesh(self.msh)
            xdmf.write_function(u1)
        with io.XDMFFile(self.msh.comm,"{0}-approx.xdmf".format(name), "w") as xdmf:
            u1 = fem.Function(fem.FunctionSpace(self.msh,('Lagrange',1)))
            u1.interpolate(self.u1_0)
            xdmf.write_mesh(self.msh)
            xdmf.write_function(u1)

    # plots time series
    def PlotParaviewTS(self,uh_inp,name="abserr"):
        
        self.uh.x.array[:] = uh_inp.array[:]
        self.uh.x.scatter_forward()
        u1_h,u2_h,z1_h,z2_h = self.uh.split()
        self.u1_0.x.array[:] = u1_h.x.array[ self.dofmaps_full[0][0] ]
        self.u2_0.x.array[:] = u2_h.x.array[ self.dofmaps_full[1][0] ]

        delta_t = self.delta_t
        int_idx = 0
        
        with io.XDMFFile(self.msh.comm,"{0}-approx_at_t.xdmf".format(name), "w") as xdmf:
            xdmf.write_mesh(self.msh)

            for n in range(self.N):
                t_n = n*delta_t   
                if n > 0:
                    self.u1_minus.x.array[:] =  u1_h.x.array[self.dofmaps_full[0][int_idx-1]]



                for l in range(self.q+1):
                    self.u1_slab[l].x.array[:] = u1_h.x.array[ self.dofmaps_full[0][int_idx+l] ]

                for t_sample in self.sample_pts_error: 
                    if t_sample >= t_n and t_sample <= (t_n + delta_t):
                        uh_at_ti = sum([  self.phi_trial[j]((t_sample - t_n)/delta_t) * self.u1_slab[j] for j in range(self.q+1)  ] )
                        if n > 0:
                            uh_at_ti -= theta_ref( ( 2*t_sample - (t_n + delta_t + t_n  ) ) / delta_t ) * ( self.u1_slab[0] -  self.u1_minus  )  
                        
                        uh_ti = fem.Function(self.fes_u1_0)
                        uh_at_ti_expr = fem.Expression(uh_at_ti, self.fes_u1_0.element.interpolation_points() )
                        uh_ti.interpolate(uh_at_ti_expr)
                        #u1 = fem.Function(fem.functionspace(self.msh,('Lagrange',1)))
                        u1 = fem.Function(fem.FunctionSpace(self.msh,('Lagrange',1)))
                        u1.interpolate(uh_ti)
                        xdmf.write_function(u1,t_sample)
                        
                

                            
                int_idx += (self.q+1)
            xdmf.close()

        u_exact_at_t = fem.Function(self.fes_u1_0)

        with io.XDMFFile(self.msh.comm,"{0}-exact_at_t.xdmf".format(name), "w") as xdmf:
            xdmf.write_mesh(self.msh)
            for t_sample in self.sample_pts_error: 
                ue_at_t = self.sol(t_sample,self.x) 
                u_expr_exact_at_t = fem.Expression(ue_at_t, self.fes_u1_0.element.interpolation_points() )
                u_exact_at_t.interpolate(u_expr_exact_at_t)                
                #u1 = fem.Function(fem.functionspace(self.msh,('Lagrange',1)))
                u1 = fem.Function(fem.FunctionSpace(self.msh,('Lagrange',1)))
                u1.interpolate(u_exact_at_t)
                xdmf.write_function(u1,t_sample)
        xdmf.close()

        u_diff_at_t = fem.Function(self.fes_u1_0)

        #not efficient to compute this again...
        with io.XDMFFile(self.msh.comm,"{0}-u_at_t.xdmf".format(name), "w") as xdmf:
            xdmf.write_mesh(self.msh)
            int_idx = 0
            for n in range(self.N):
                t_n = n*delta_t   
                if n > 0:
                    self.u1_minus.x.array[:] =  u1_h.x.array[self.dofmaps_full[0][int_idx-1]]



                for l in range(self.q+1):
                    self.u1_slab[l].x.array[:] = u1_h.x.array[ self.dofmaps_full[0][int_idx+l] ]

                for t_sample in self.sample_pts_error: 
                    if t_sample >= t_n and t_sample <= (t_n + delta_t):
                        uh_at_ti = sum([  self.phi_trial[j]((t_sample - t_n)/delta_t) * self.u1_slab[j] for j in range(self.q+1)  ] )
                        if n > 0:
                            uh_at_ti -= theta_ref( ( 2*t_sample - (t_n + delta_t + t_n  ) ) / delta_t ) * ( self.u1_slab[0] -  self.u1_minus  )  

                        uh_ti = fem.Function(self.fes_u1_0)
                        uh_at_ti_expr = fem.Expression(uh_at_ti, self.fes_u1_0.element.interpolation_points() )
                        uh_ti.interpolate(uh_at_ti_expr)
                        
                        ue_at_t =  self.sol(t_sample,self.x)    
                        
                        if isinstance(ue_at_t, ufl.classes.Zero):
                            u_diff_at_t.x.array[:] = np.abs( self.u1_0.x.array[:] )
                        else:
                            u_expr_exact_at_ = fem.Expression(ue_at_t, self.fes_u1_0.element.interpolation_points() )
                            u_exact_at_t.interpolate(u_expr_exact_at_t) 
                            u_diff_at_t.x.array[:] = np.abs( u_exact_at_t.x.array[:] - uh_ti.x.array[:] )

                        u1 = fem.Function(fem.FunctionSpace(self.msh,('Lagrange',1)))
                        u1.interpolate(u_diff_at_t)
                        xdmf.write_function(u1,t_sample)

                            
                int_idx += (self.q+1)
            xdmf.close()
            
                    

    def compute_best_approx(self,x_sol):

        u1_p,u2_p,z1_p,z2_p = ufl.TrialFunctions(self.fes_p)
        w1_p,w2_p,y1_p,y2_p = ufl.TestFunctions(self.fes_p)
        
        dx = self.dx
        n_facet = self.n_facet
        h = self.h
        delta_t = self.delta_t
        elmat_time = self.elmat_time
         
        m_mass = 1e-20*u1_p[0]*y1_p[0]*dx 

        # (u_1,w_1) 
        for j in range(self.q+1):
            for k in range(self.q+1):
                m_mass += delta_t *  elmat_time["M_time_q_q"][k,j] * inner( u1_p[j], w1_p[k] ) * dx 
                m_mass += delta_t *  elmat_time["M_time_q_q"][k,j] * inner( u2_p[j], w2_p[k] ) * dx 


        for j in range(self.qstar+1):
            for k in range(self.qstar+1):
                m_mass += delta_t *  elmat_time["M_time_q_q"][k,j] * inner( y1_p[j], z1_p[k] ) * dx 
                m_mass += delta_t *  elmat_time["M_time_q_q"][k,j] * inner( y2_p[j], z2_p[k] ) * dx 

        bfi_m = fem.form(m_mass)
        M_mass = assemble_matrix(bfi_m, bcs=[])
        M_mass.assemble()

        solver_m = PETSc.KSP().create(self.msh.comm)
        solver_m.setOperators(M_mass)
        solver_m.setType(PETSc.KSP.Type.PREONLY)
        solver_m.getPC().setType(PETSc.PC.Type.LU)

         
        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1 
        
        for n in range(self.N):

            t_n = n*delta_t
     
            L_mass = 1e-20*inner(1.0, self.y1_p[0] ) * dx

            if n > 0:
                L_mass +=  inner(self.u2_0_pre,self.w2_p[0]) * dx

            for tau_i,omega_i in zip(self.qr.current_pts(0,1),self.qr.t_weights(1)):
                time_ti = t_n + delta_t*tau_i                 
                sol_ti = self.sol(time_ti,self.x)   
                L_mass +=  delta_t * omega_i * sol_ti *  sum([  w1_p[k]*self.phi_test[k](tau_i) for k in range(self.q+1) ]) *dx

            lfi_m = fem.form(L_mass)
            b_rhs_m = fem.petsc.create_vector(lfi_m)
            fem.petsc.assemble_vector(b_rhs_m, lfi_m)

            # copy rhs entries from global to local vector on the slice
            for j in range(4): 
                if j in [0,1]: 
                    shift_idx = int_idx_q 
                else:
                    shift_idx = int_idx_qstar 
                #for l in range(self.subspace_time_order[j]+1):
                #    b_rhs_m.array[self.dofmaps_pre[j][l]] += b.array[self.dofmaps_full[j][shift_idx+l]] 

            solver_m.solve(b_rhs_m, self.uh_pre.vector)
            self.uh_pre.x.scatter_forward()
            
            u1_h_p,u2_h_p,z1_h_p,z2_h_p = self.uh_pre.split()
            self.u1_0_pre.x.array[:] = u1_h_p.x.array[ self.dofmaps_pre[0][self.q] ]
            self.u2_0_pre.x.array[:] = u2_h_p.x.array[ self.dofmaps_pre[1][self.q] ]
            
            for j in range(4):  
                if j in [0,1]: 
                    shift_idx = int_idx_q 
                else:
                    shift_idx = int_idx_qstar 
                for l in range(self.subspace_time_order[j]+1):
                    x_sol.array[ self.dofmaps_full[j][shift_idx+l] ] =  self.uh_pre.x.array[self.dofmaps_pre[j][l]] 

            int_idx_q += (self.q+1)
            coupling_idx_q += (self.q+1)
            int_idx_qstar += (self.qstar+1)
            coupling_idx_qstar += (self.qstar+1)

