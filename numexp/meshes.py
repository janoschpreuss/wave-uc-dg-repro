import gmsh
import numpy as np

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities, refine, compute_incident_entities, GhostMode
#from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner,And,Not,conditional)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from math import pi
from dolfinx.io import XDMFFile,gmshio
from dolfinx.cpp.io import perm_gmsh
from dolfinx.mesh import CellType, create_mesh
from dolfinx.cpp import mesh as cmesh
#help(CellType)
import itertools
GM = GhostMode.shared_facet


def get_1Dmesh(xcoord = [0.0,0.25,0.5,0.75,1.0]): 


    # points to connect
    points = [(xc, 0, 0) for xc in xcoord]

    # assign node numbers
    nodes = [x for x in range(1, len(points)+1)]
    # get a flat list of coordinates
    coords = list(itertools.chain.from_iterable(points))

    gmsh.initialize()
    # add a dummy line to add nodes to
    dummy_line = 1
    gmsh.model.addDiscreteEntity(1, dummy_line)

    # add nodes
    gmsh.model.mesh.addNodes(1, dummy_line, nodes, coords)

    # add line elements between each node
    line_elt = 1
    for i in range(len(nodes) - 1):
        gmsh.model.mesh.addElementsByType(
            dummy_line, line_elt, [], [nodes[i], nodes[i+1]])

    gdim = 1  # 1D mesh 
    proc = MPI.COMM_WORLD.rank 

    its = 1 
    #print( " len(gmsh.model.getEntities(dim=1)) = ", len(gmsh.model.getEntities(dim=1)) )  
    for surface in gmsh.model.getEntities(dim=1):
        gmsh.model.addPhysicalGroup(1, [surface[0]], its )
        its +=1
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(gdim)

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
    
    return mesh




def get_2Dmesh_data_all_around(n_ref,init_h_scale=1.0,data_size=0.2): 

    gmsh.initialize()
    gdim = 2
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale)
    proc = MPI.COMM_WORLD.rank
    top_marker = 2
    bottom_marker = 1
    bnd_marker = 1
    omega_marker = 1
    Bwithoutomega_marker = 2
    rest_marker = 3 

    r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1,tag=1)
    r2 = gmsh.model.occ.addRectangle(data_size, data_size, 0, 1-2*data_size, 1-2*data_size,tag=2)
    r3 = gmsh.model.occ.cut( [(2,r1)], [(2,r2)],tag=3)

    print("r3 = ", r3)
    #gmsh.model.occ.addRectangle(0.1, 0.1, 0, 0.8, 0.9,tag=4)
    gmsh.model.occ.addRectangle(data_size, data_size, 0, 1-2*data_size, 1-2*data_size,tag=4)

    # We fuse the two rectangles and keep the interface between them
    gmsh.model.occ.fragment([(2,3)],[(2,4)])
    gmsh.model.occ.synchronize()

    its = 1 
    
    for surface in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        print(com)
        gmsh.model.addPhysicalGroup(2, [surface[1]], its )
        its +=1
    
    gmsh.model.mesh.generate(gdim)

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

    mesh_hierarchy = [] 
    mesh_hierarchy.append(mesh) 

    for i in range(n_ref):
        mesh.topology.create_entities(2)
        mesh.topology.create_entities(1)
        mesh = refine(mesh)
        mesh_hierarchy.append(mesh) 

    return mesh_hierarchy


def get_3Dmesh_data_all_around(n_ref,init_h_scale=1.0): 


    proc = MPI.COMM_WORLD.rank
    #if proc == 0:
    gdim = 3
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale)
    top_marker = 2
    bottom_marker = 1
    bnd_marker = 1
    omega_marker = 1
    Bwithoutomega_marker = 2
    rest_marker = 3
    # We create one cube for each subdomain

    r1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1,1,tag=1)
    
    r2 = gmsh.model.occ.addBox(0.2, 0.2,0.2, 0.6, 0.6, 0.6,tag=2)
    r3 = gmsh.model.occ.cut( [(3,r1)], [(3,r2)],tag=3)
    r4 = gmsh.model.occ.addBox(0.2, 0.2,0.2, 0.6, 0.6, 0.6,tag=4)

    # We fuse the two boxes and keep the interface between them
    ov, ovv = gmsh.model.occ.fragment([(3,3)],[(3,4)] )
    ov, ovv = gmsh.model.occ.fragment([(2,3)],[(2,4)] )

    print("fragment produced volumes:")
    for e in ov:
        print(e)

    # ovv contains the parent-child relationships for all the input entities:
    print("before/after fragment relations:")
    for e in ovv:
        print(e)


    gmsh.model.occ.synchronize()
    
    for e in ov:
        gmsh.model.add_physical_group(dim=3, tags=[e[1]])
        print(e)
    its = 1
    for surface in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        print(com)
        gmsh.model.addPhysicalGroup(2, [surface[1]], its )
        its +=1
    #input("")
    #gmsh.model.add_physical_group(dim=3, tags=[3])
    #gmsh.model.add_physical_group(dim=3, tags=[4])

    #its = 1 
    #for surface in gmsh.model.getEntities(dim=3):
    #    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    #    print(com)
    #    gmsh.model.addPhysicalGroup(2, [surface[1]], its )
    #    its +=1

    # Tag the left boundary
    #bnd_square = []
    #for line in gmsh.model.getEntities(dim=1):
    #    com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
    #    if np.isclose(com[0], 0) or np.isclose(com[0], 1) or np.isclose(com[1], 0) or  np.isclose(com[1], 1): 
    #        bnd_square.append(line[1])
    #gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
    gmsh.model.mesh.generate(gdim)
    #gmsh.write("mesh.msh")
    #gmsh.finalize()

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

    mesh_hierarchy = [] 
    mesh_hierarchy.append(mesh) 

    #def refine_all(x):
    #    return x[0] >= 0
    for i in range(n_ref):
        mesh.topology.create_entities(2)
        mesh.topology.create_entities(1)
        #cells = locate_entities(mesh, mesh.topology.dim, refine_all)
        #edges = compute_incident_entities(mesh, cells, 2, 1)
        #print(edges)
        #help(refine)
        mesh = refine(mesh)
        mesh_hierarchy.append(mesh) 
    return mesh_hierarchy


def get_mesh_hierarchy_fitted_disc(n_ref,x_IF=0.5,height=0.5,x_data_left=0.75, h_init=1.25):

    gdim = 2
    xi = (1.0-height)/2

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h_init)
    proc = MPI.COMM_WORLD.rank

    # We create one rectangle for each subdomain

    r1 = gmsh.model.occ.addRectangle(x_IF, 0,0, 1.0-x_IF, 1 ,tag=1)
    r2 = gmsh.model.occ.addRectangle(x_IF, xi, 0, x_data_left-x_IF ,1-2*xi ,tag=2)
    r3 = gmsh.model.occ.cut( [(2,r1)], [(2,r2)],tag=3)

    r4 = gmsh.model.occ.addRectangle(0.0, 1-xi, 0, x_IF, xi,tag=4)
    r5 = gmsh.model.occ.addRectangle(0.0, 0.0, 0, x_IF, xi,tag=5)
    r6 = gmsh.model.occ.addRectangle(0.0, xi , 0, x_IF, 1-2*xi ,tag=6)

    r7 = gmsh.model.occ.addRectangle(x_IF, xi, 0, x_data_left-x_IF ,1-2*xi ,tag=7)


    gmsh.model.occ.fragment([(2,3)],[(2,7),(2,3),(2,4), (2,5),(2,6)] )
    gmsh.model.occ.synchronize()
    
    its = 1
    for surface in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        print(com)
        gmsh.model.addPhysicalGroup(2, [surface[1]], its )
        its +=1

    gmsh.model.mesh.generate(gdim)

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

    mesh_hierarchy = [] 
    mesh_hierarchy.append(mesh) 

    for i in range(n_ref):
        mesh.topology.create_entities(2)
        mesh.topology.create_entities(1)
        mesh = refine(mesh)
        mesh_hierarchy.append(mesh) 
    return mesh_hierarchy

#ls = get_mesh_hierarchy_fitted_disc(n_ref=1,x_IF=0.5,height=0.5,x_data_left=0.75, h_init=0.25)
#msh = ls[0]

#with XDMFFile(msh.comm, "mesh.xdmf", "w") as xdmf:
#        xdmf.write_mesh(msh)
