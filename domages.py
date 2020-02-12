from __future__ import division # Allow the division of integer 
from dolfin import * 
import numpy as np
import sympy # Library for symbolic mathematics
import matplotlib.pyplot as plt
import random as rd
#set_log_level(PROGRESS)
import time

##############################
### Begining of Parameters ###
##############################

# Save the values in "output.txt
Output_latex = True

# Create flat cells
Damage = True

# Initiallisation of the iterations
init_Iter = 1

# Number refienement of the mesh
Iter = 4

# number of flatted cells
#1) 10 patches
#2) 2 patches (test) mesh 8X8
#3) full
choice_num_patch = 2

# Activate the ghost penalty
ghost = True

# Print the conditionong of the matrix
conditioning = False


Plot = False

#########################
### End of Parameters ###
#########################


# Dirichlet boundary condition
class Dirich(SubDomain):
    def inside(self, x, on_boundary):
        return True if on_boundary else False

dirich = Dirich()


# Coefficient of flattness
def delta(h):
	return h**2/np.sqrt(2)


def modif_mesh(mesh):
	for ii in range(int(N/3)):
		for jj in range(int(N/3)):
			x_ref = (3*ii+2)*h
			y_ref = (3*jj+1)*h
			p_ref = Point(x_ref,y_ref)
			d, ver = min((p_ref.distance(v.point()), v) for v in vertices(mesh))
			node = ver.index()
			mesh.coordinates()[node][0] = mesh.coordinates()[node][0] - h/2+Delta
			mesh.coordinates()[node][1] = mesh.coordinates()[node][1] + h/2-Delta

	return mesh



# Compute the area of the cell with index ind
def area_cell(ind):
	v1x,v1y,v2x,v2y,v3x,v3y = Cell(mesh,ind).get_vertex_coordinates()
	return 0.5*abs((v2x-v1x)*(v3y-v1y)-(v3x-v1x)*(v2y-v1y))


# Give thevertices of the cell "mycell" with x0 as the first one
def vertex_d(mycell):
	v1,v2,v3=vertices(mycell)
	v1x,v1y,v2x,v2y,v3x,v3y = mycell.get_vertex_coordinates()
	d1 = np.sqrt((v3x-v2x)**2+(v3y-v2y)**2)
	d2 = np.sqrt((v3x-v1x)**2+(v3y-v1y)**2)
	d3 = np.sqrt((v1x-v2x)**2+(v1y-v2y)**2)
	sup = max(d1,d2,d3)
	if sup==d2:
		v1,v2,v3 = v2,v1,v3
	if sup==d3:
		v1,v2,v3 = v3,v1,v2
	return v1,v2,v3


# Give the coefficient before the non degenerate cells in the ghost penalty formulation
def coeff_weight_nd(ind_cell_nd,ind_cell_d):
	area_nd = area_cell(ind_cell_nd)
	area_d = area_cell(ind_cell_d)
	#print("area_nd",area_nd)
	#print("area_d",area_d)
	area_P = area_nd + area_d
	return area_P/area_nd


# Give the coefficient before the jump in the ghost penalty formulation
def coeff_weight_jump(ind_cell_nd,ind_cell_d):
	area_d = area_cell(ind_cell_d)
	length_f = np.sqrt(2)/N
	return (2/3)*area_d**3/length_f**2


# Say if a cell belongs to the boundary
def cell_have_vertex_on_boundary(mycell,bbtree):
	res = False
	for myvertex in vertices(mycell):
		_, distance = bbtree.compute_closest_entity(myvertex.point())
		if near(distance,0.0)==True:
			res = True
	return res


# Flatt the cell with index cell_d
def flatt(cell_d):
	v1,v2,v3 = vertex_d(cell_d) #v1=x0
	v1x = v1.point().x()
	v1y = v1.point().y()
	v2x = v2.point().x()
	v2y = v2.point().y()
	v3x = v3.point().x()
	v3y = v3.point().y()
	dirx = (v3x-v2x)/np.sqrt((v2x-v3x)**2+(v2y-v3y)**2)
	diry = (v3y-v2y)/np.sqrt((v2x-v3x)**2+(v2y-v3y)**2)
	vec21x = v1x-v2x
	vec21y = v1y-v2y
	ps = dirx*vec21x + diry*vec21y
	hx = v2x+ps*dirx
	hy = v2y+ps*diry
	hv1 = np.sqrt((hx-v1x)**2+(hy-v1y)**2)
	mesh.coordinates()[v1.index()][0] = hx+Delta*(v1x-hx)/hv1
	mesh.coordinates()[v1.index()][1] = hy+Delta*(v1y-hy)/hv1
	return


def bilin_ghost_full(mesh):
	domains = CellFunction("size_t",mesh)
	patches = CellFunction("size_t",mesh)
	weight_nd = interpolate(Expression("0.0",degree=1,domain=mesh),DG0)
	weight_jump = interpolate(Expression("0.0",degree=1,domain=mesh),DG0)
	domains.set_all(0)

	ind_cell_d_array=np.zeros(num_patch)
	ind_cell_nd_array=np.zeros(num_patch)
	ind_x0_array=np.zeros(num_patch)

	D = mesh.topology().dim()
	mesh.init(D-1,D)
	bmesh = BoundaryMesh(mesh, "exterior")
	bbtree = BoundingBoxTree()
	bbtree.build(bmesh)
	tree = mesh.bounding_box_tree()
	for ii in range(int(N/3)):
		for jj in range(int(N/3)):
			x_ref = (3*ii+5/4)*h
			y_ref = (3*jj+7/4)*h
			p_ref = Point(x_ref,y_ref)
			ind_cell_nd, d =tree.compute_closest_entity(p_ref)
			x_ref = (3*ii+3/2)*h+Delta/2
			y_ref = (3*jj+3/2)*h-Delta/2
			p_ref = Point(x_ref,y_ref)
			ind_cell_d, d =tree.compute_closest_entity(p_ref)
			domains[ind_cell_nd] = 1
			domains[ind_cell_d] = 2
			#### patches
			cell_d = Cell(mesh,ind_cell_d)
			cell_nd = Cell(mesh,ind_cell_nd)
			ver_d1,ver_d2,ver_d3 = vertex_d(cell_d) #ver_d1=x0
			ind_d1, ind_d2, ind_d3 = ver_d1.index(),ver_d2.index(),ver_d3.index()
			ver_nd1,ver_nd2,ver_nd3 = vertices(cell_nd) #v1=x0
			ind_nd1, ind_nd2, ind_nd3 = ver_nd1.index(),ver_nd2.index(),ver_nd3.index()
			for mycell in cells(mesh):
				ind_ver = mycell.entities(0)
				if (ind_d1 in ind_ver) or (ind_d2 in ind_ver) or (ind_d3 in ind_ver) or (ind_nd1 in ind_ver) or (ind_nd2 in ind_ver) or (ind_nd3 in ind_ver):
					patches[mycell.index()] = 1
			patches[ind_cell_nd] = 2
			patches[ind_cell_d] = 3
			#### patches
			weight_nd.vector()[ind_cell_nd] = coeff_weight_nd(ind_cell_nd,ind_cell_d)
			weight_jump.vector()[ind_cell_d] = coeff_weight_jump(ind_cell_nd,ind_cell_d)
			ind_cell_d_array[int(ii*N/3+jj)]=ind_cell_d
			ind_cell_nd_array[int(ii*N/3+jj)]=ind_cell_nd
			ver_d1,ver_d2,ver_d3 = vertex_d(Cell(mesh,ind_cell_d))
			ind_d1, ind_d2, ind_d3 = ver_d1.index(),ver_d2.index(),ver_d3.index()
			ind_x0_array[int(ii*N/3+jj)]=ind_d1
	return patches, ind_cell_d_array, ind_cell_nd_array, ind_x0_array, domains, weight_nd, weight_jump


def bilin_ghost(mesh):
	domains = MeshFunction("size_t", mesh, mesh.topology().dim())
	cells_dispo = MeshFunction("size_t", mesh, mesh.topology().dim())
	weight_nd = interpolate(Expression("0.0",degree=1,domain=mesh),DG0)
	weight_jump = interpolate(Expression("0.0",degree=1,domain=mesh),DG0)
	domains.set_all(0)
	cells_dispo.set_all(0) # 0 : dispo

	ind_cell_d_array=np.zeros(num_patch)
	ind_cell_nd_array=np.zeros(num_patch)
	ind_x0_array=np.zeros(num_patch)

	D = mesh.topology().dim()
	mesh.init(D-1,D)
	bmesh = BoundaryMesh(mesh, "exterior")
	bbtree = BoundingBoxTree()
	bbtree.build(bmesh)

	for iii in range(num_patch):
		dispo = False
		while dispo == False:
			ind_cell_d = rd.randint(0,mesh.num_cells()-1)
			if cells_dispo[ind_cell_d] == 0:
				cell_d = Cell(mesh,ind_cell_d)
				if cell_have_vertex_on_boundary(cell_d,bbtree) == False:
					ver_d1,ver_d2,ver_d3 = vertex_d(cell_d) #ver_d1=x0
					ind_d1, ind_d2, ind_d3 = ver_d1.index(),ver_d2.index(),ver_d3.index()
					for mycell in cells(mesh):
						inid_ver = mycell.entities(0)
						if (ind_d1 not in inid_ver) and (ind_d2 in inid_ver) and (ind_d3 in inid_ver):
							ind_cell_nd = mycell.index()
					cell_nd = Cell(mesh,ind_cell_nd)
					ver_nd1,ver_nd2,ver_nd3 = vertices(cell_nd) #v1=x0
					ind_nd1, ind_nd2, ind_nd3 = ver_nd1.index(),ver_nd2.index(),ver_nd3.index()
					if cell_have_vertex_on_boundary(cell_nd,bbtree) == False:
						dispo = True
						for mycell in cells(mesh):
							ind_ver = mycell.entities(0)
							if (ind_d1 in ind_ver) or (ind_d2 in ind_ver) or (ind_d3 in ind_ver) or (ind_nd1 in ind_ver) or (ind_nd2 in ind_ver) or (ind_nd3 in ind_ver):
								if cells_dispo[mycell.index()] == 1 or cells_dispo[mycell.index()] == 2 or cells_dispo[mycell.index()] == 3:
									dispo = False

		for mycell in cells(mesh):
			ind_ver = mycell.entities(0)
			if (ind_d1 in ind_ver) or (ind_d2 in ind_ver) or (ind_d3 in ind_ver) or (ind_nd1 in ind_ver) or (ind_nd2 in ind_ver) or (ind_nd3 in ind_ver):
				cells_dispo[mycell.index()] = 1


		cells_dispo[ind_cell_nd] = 2
		cells_dispo[ind_cell_d] = 3
		domains[ind_cell_nd] = 1
		domains[ind_cell_d] = 2
		flatt(cell_d) # Applatissement
		weight_nd.vector()[ind_cell_nd] = coeff_weight_nd(ind_cell_nd,ind_cell_d)
		weight_jump.vector()[ind_cell_d] = coeff_weight_jump(ind_cell_nd,ind_cell_d)
		ind_cell_d_array[iii]=ind_cell_d
		ind_cell_nd_array[iii]=ind_cell_nd
		ind_x0_array[iii]=ind_d1
	return cells_dispo, ind_cell_d_array, ind_cell_nd_array, ind_x0_array, domains, weight_nd, weight_jump


# Compute the local interpolation of "func"
def Ih_tilde(func):
	res=func
	for iii in range(num_patch):
		v1x,v1y,v2x,v2y,v3x,v3y = Cell(mesh,int(ind_cell_nd_array[iii])).get_vertex_coordinates()
		A = np.matrix([[v1x,v1y,1],[v2x,v2y,1],[v3x,v3y,1]])
		B = np.matrix([[func(v1x,v1y)],[func(v2x,v2y)],[func(v3x,v3y)]])
		C = np.dot(np.linalg.inv(A),B)
		a, b, c = C[0], C[1], C[2]
		x0 = Vertex(mesh,int(ind_x0_array[iii]))
		vertex_values = np.zeros(mesh.num_vertices())
		for jjj in range(mesh.num_vertices()):
			vertex_values[jjj]=res(Vertex(mesh,jjj).x(0),Vertex(mesh,jjj).x(1))
		#print(ind_x0_array)
		vertex_values[np.int(ind_x0_array[iii])]=a*x0.x(0)+b*x0.x(1)+c
		res.vector()[:] = vertex_values[dof_to_vertex_map(V)]
	return res


####################################
### Begining of the computations ###
####################################

#if choice_num_patch == 2:
#	init_Iter = 1
#	Iter = 1

size_mesh_vec = np.zeros(Iter)
error_L2_vec = np.zeros(Iter)
error_H1_vec = np.zeros(Iter)
if ghost == True:
	error_triple_vec = np.zeros(Iter)
cond_vec = np.zeros(Iter)
time_vec = np.zeros(Iter)

for i in range(init_Iter-1,Iter):
	print("################")
	print("################")
	print("Iteration : ",i+1)

	#construction f the mesh
	if choice_num_patch == 1 or choice_num_patch == 1:
		N = int(10*e**((i+1)/2))
	if choice_num_patch == 2:
		N = 7
	if choice_num_patch == 3:
		N = 9*int(e**((i+1)/2))
	mesh = UnitSquareMesh(N,N)
	h = 1/N
	Delta = delta(h)
	if choice_num_patch == 3:
		mesh = modif_mesh(mesh)
		num_patch = int(N*N/9)

	print('Number of cells',mesh.num_cells())


	# Number of flatted cells
	if choice_num_patch == 1:
		num_patch = 10
	if choice_num_patch == 2:
		num_patch = 2

	DG0 = FunctionSpace(mesh,'DG',0)
	if choice_num_patch == 1 or choice_num_patch == 2:
		cells_dispo, ind_cell_d_array, ind_cell_nd_array, ind_x0_array, domains, weight_nd, weight_jump = bilin_ghost(mesh)
	if choice_num_patch == 3:
		DG0 = FunctionSpace(mesh,'DG',0)
		patches, ind_cell_d_array, ind_cell_nd_array, ind_x0_array, domains, weight_nd, weight_jump = bilin_ghost_full(mesh)


	print('mesh : ok')

	# Initialize cell function for domains
	dx = Measure("dx")(subdomain_data = domains)
	dS = Measure("dS")(subdomain_data = domains)



	V = FunctionSpace(mesh, "CG", 1)

	# Computation of the Exact solution
	x, y = sympy.symbols('xx yy')
	u1 = sympy.sin(pi*x)*sympy.sin(pi*y)
	#plot(u_expr,mesh=mesh)
	#mesh_exact = UnitSquareMesh(10*3*3*(Iter+1),10*3*3*(Iter+1))
	u_expr = Expression(sympy.ccode(u1).replace('xx', 'x[0]').replace('yy', 'x[1]'),degree=4,domain=mesh)
	Iu_expr = interpolate(u_expr,V)
	Iu_expr = Ih_tilde(Iu_expr)


	# Compute grad u
	dxu1 = sympy.diff(u1, x)
	dyu1 = sympy.diff(u1, y)
	dxu2 = sympy.ccode(dxu1).replace('xx', 'x[0]').replace('yy', 'x[1]')
	dyu2 = sympy.ccode(dyu1).replace('xx', 'x[0]').replace('yy', 'x[1]')
	grad_u = Expression((dxu2,dyu2),degree =4,domain=mesh)


	# Compute source term
	f1 = -sympy.diff(sympy.diff(u1, x),x)-sympy.diff(sympy.diff(u1, y),y)
	f2 = sympy.ccode(f1).replace('xx', 'x[0]').replace('yy', 'x[1]')

	f = Expression(f2,degree =4,domain=mesh)

	# Computation of the solution

	bcs = DirichletBC(V, Constant(0.0), dirich)
	u = TrialFunction(V)
	v = TestFunction(V)
	if ghost == False:
		a = inner(grad(u),grad(v))*dx
	else:
		a = inner(grad(u),grad(v))*dx(0) + weight_nd*inner(grad(u),grad(v))*dx(1) + (1/h**2)*2.0*avg(weight_jump)*inner(jump(grad(u)),jump(grad(v)))*dS(1)
		#a = inner(grad(u),grad(v))*dx(0) + weight_nd*inner(grad(u),grad(v))*dx(1)
	L = f*v*dx
	u_h = Function(V)
	initial_time = time.time()
	solve(a == L, u_h, bcs=bcs)
	compute_time = time.time()-initial_time
	#problem = LinearVariationalProblem(a, L, u_h, bcs)
	#solver = LinearVariationalSolver(problem)
	#solver.parameters['linear_solver']='cg'
	#solver.parameters["linear_solver"]["monitor_convergence"] = True
	#solver.solve()


	#error = project(u_h-Iu_expr,V)
	#error_L2 = norm(error,'L2')
	error = u_h-u_expr

	error_L2 = assemble(error*error*dx)**0.5

	error_L2_vec[i] = error_L2
	print("L2 error",error_L2)
	#error_H1 = norm(error,'H1')
	grad_error = u_h-u_expr
	error_H1 = assemble(inner(grad(u_h)-grad_u,grad(u_h)-grad_u)*dx)**0.5

	error_H1_vec[i] = error_H1
	print("H1 error",error_H1)
	size_mesh_vec[i] = h
	print("h",h)

	if ghost == True:
		#I_h_error = project(u_h-u_expr,V)
		#error_triple = assemble(inner(grad(error),grad(error))*dx(0) + weight_nd*inner(grad(error),grad(error))*dx(1))**0.5 #+ (1/h**2)*2.0*avg(weight_jump)*inner(jump(grad(error)),jump(grad(error)))*dS(1)
		error2 = Ih_tilde(u_h) - u_expr
		#error2 = Ih_tilde(u_h) - Ih_tilde(project(u_expr,V))
		#error3 = u_h - Ih_tilde(u_h) - u_expr + Ih_tilde(project(u_expr,V))
		error_triple = assemble(inner(grad(error),grad(error))*dx(0) + inner(grad(error2),grad(error2))*dx(1)+ inner(grad(error2),grad(error2))*dx(2))**0.5
		#error_triple = assemble(inner(grad(error),grad(error))*dx(0) + inner(grad(error2),grad(error2))*dx(1)+ inner(grad(error2),grad(error2))*dx(2)+ (1/h**2)*inner(grad(error3),grad(error3))*dx(1)+ (1/h**2)*inner(grad(error3),grad(error3))*dx(2))**0.5
		print("|u-u_h|_1 + sum_i |u-I_hu_h|_1",error_triple)
		error_triple_vec[i] = error_triple
	if conditioning == True:
		A = assemble(a)
		B = assemble(L)

		bcs.apply(A,B)
		A = A.array()
		#print(A)
		#A = A[1:-1,1:-1]
		#ev, eV = np.linalg.eig(A)
		#ev = np.sort(ev)
		#print(ev)
		#ev = ev[1:]
		#cond = h**2*np.max(ev)/np.min(ev)
		#print("min ev",np.min(ev))
		cond = h**2*np.linalg.cond(A)
		cond_vec[i] = cond
		print("conditioning number x h^2",cond)
	time_vec[i] = compute_time
	print("time of computation",compute_time)
	#plot(mesh,interactive=True)


# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')


# print the different arrays in the outputs files
if Output_latex == True:
	if ghost == True:
		f = open('output_ghost.txt','w')
	if ghost == False:
		f = open('output_no_ghost.txt','w')
	if Damage == True:
		f.write('Mesh with damage \n \n')
	else:
		f.write('Mesh with no damage \n \n')
	f.write('L2 norm : \n')	
	output_latex(f,size_mesh_vec,error_L2_vec)
	f.write('H1 norm : \n')	
	output_latex(f,size_mesh_vec,error_H1_vec)
	if ghost == True:
		f.write('|u-u_h|_1 + sum_i |u-I_hu_h|_1 : \n')	
		output_latex(f,size_mesh_vec,error_triple_vec)
	f.write('conditioning number x h^2 : \n')	
	output_latex(f,size_mesh_vec,cond_vec)
	f.write('time of computation : \n')	
	output_latex(f,size_mesh_vec,time_vec)
	f.close()



print(error_L2_vec)
print(error_H1_vec)
if ghost == True:
	print(error_triple_vec)
print(cond_vec)


if Plot == True:

	if ghost == True:
		plot_domains = plot(domains,title='Dommains')
	plot_mesh = plot(mesh,title = 'Final mesh')
	if choice_num_patch == 1 or choice_num_patch == 2:
		plot_cells_dispo = plot(cells_dispo,title='Cells dispo')
	if choice_num_patch == 3:
		plot_patches = plot(patches,title='Patches')
	plot(weight_nd ,title= 'Coeff non degenerate')
	plot(weight_jump,title='Coeff jump')
	#interactive()


	if ghost == True:
		plot_domains.savefig('domains.png')
	plot_mesh.write_png('mesh')
	if choice_num_patch == 1 or choice_num_patch == 2:
		plot_cells_dispo.savefig('cell_dispo.png')
	if choice_num_patch == 3:
		plot_patches.wsavefig('patches.png')




