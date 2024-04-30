import numpy as np
import scipy.sparse as sp
from math import pi,sqrt  
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, dS,jump,div
from mpi4py import MPI
from petsc4py import PETSc
import pypardiso

def GetSpMat(mat):
    ai, aj, av = mat.getValuesCSR()
    Asp = sp.csr_matrix((av, aj, ai))
    return Asp

def GetLuSolver(msh,mat):
    solver = PETSc.KSP().create(msh.comm) 
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    #solver.getPC().setFactorSolverType("mumps")
    return solver

class PySolver:
    def __init__(self,Asp,psolver):
        self.Asp = Asp
        self.solver = psolver
    def solve(self,b_inp,x_out): 
        self.solver._check_A(self.Asp)
        b = self.solver._check_b(self.Asp, b_inp.array)
        self.solver.set_phase(33)
        x_out.array[:] = self.solver._call_pardiso(self.Asp , b )[:]

def GetPyPardisoSolver(mat):
    solver_instance = pypardiso.PyPardisoSolver()
    matrix_A  = GetSpMat(mat)
    solver_instance.factorize(matrix_A)
    return PySolver( matrix_A, solver_instance)
