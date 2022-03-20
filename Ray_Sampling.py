import numpy as np
import matplotlib.pyplot as plt
import json


with open('/content/extrinsic.camera.01.json', 'r') as f:
  Extrinsic01 = json.load(f)

Extrinsic01 = Extrinsic01.get('camera')

R_01 = Extrinsic01.get("r")
T_01 = Extrinsic01.get("t")
R_01 = np.reshape(np.asarray(R_01,dtype = np.float64),(3,3))
T_01 = np.asarray(T_01,dtype = np.float64)

#Pixel to Camera coordinates

W_01 = float(Extrinsic01.get("width")) 
H_01 = float(Extrinsic01.get("height"))
fx_01 = float(Extrinsic01.get("fx"))
fy_01 = float(Extrinsic01.get("fy"))

#Pixel grid of size imageWidth x imageHeight
x_01, y_01 = np.meshgrid(np.arange(W_01+1),np.arange(H_01+1)) #Not a tensorflow tensor

X_camera01 = (x_01-W_01/2)/fx_01 # Assume z = 1
Y_camera01 = (y_01-H_01/2)/fy_01

C_01 = np.matmul(np.transpose(R_01),T_01) # Position of camera in world coordinates

# Building the extrinsic camera matrix
C_ex_01 = np.column_stack((R_01,T_01))
C_ex_01 = np.row_stack((C_ex_01,np.array([0,0,0,1])))
print(C_ex_01)

cam_to_world01 = np.linalg.inv(C_ex_01) # Inverse of extrinsic camera matrix -> goes from camera to world
print(cam_to_world01)
R_inv_01 = cam_to_world01[:3,:3]
T_inv_01 = cam_to_world01[:3,3]

homogeneous = np.stack((X_camera01,-Y_camera01,-np.ones(np.shape(X_camera01))),-1) #Homogeneous coordinates - stack camera coordinates
homogeneous_expanded = homogeneous[..., None, :]

# RAY SAMPLING

# Ray direction unit vectors
unit_vectors01 = np.matmul(R_inv_01, np.transpose(homogeneous_expanded[0][0]))/np.linalg.norm(np.matmul(R_inv_01, np.transpose(homogeneous_expanded[0][0])))
for i in range(0,101):
  for j in range(1,101):
    insert = np.matmul(R_inv_01, np.transpose(homogeneous_expanded[i][j]))
    insert = insert/np.linalg.norm(insert)
    unit_vectors01 = np.row_stack((unit_vectors01,insert))

# Ray origin - shoots into scene through each pixel
origin01 = T_inv_01 # camera to world translation vector


# Sampled points from along ray:

# r(t) = x_0 + t*d with near and far bounds, t_n and t_f

points_uniform = 100
t_f = 10 # Far Bound
t_n = 60 # Near Bound
point_locations_uniform = (t_f-t_n)/points_uniform
sample_points_uniform = []
for i in range(0,points_uniform+1):
  sample_points_uniform.append(i*(t_f-t_n)/points_uniform) #Points along ray to be sampled. Using this method, the same places along every ray sampled

coords_uniform01 = []
for j in range(0,len(unit_vectors01)): #Go through each of the direction unit vectors, each corresponding to a pixel
  sample_locations_uniform = []
  for i in range(0,len(sample_points_uniform)): #Go through all points along a ray and store the coordinates
    sample_locations_uniform.append(origin01+sample_points_uniform[i]*unit_vectors01[j])
  coords_uniform01.append(sample_locations_uniform)
#coords_uniform01 structure contains sampled points in 3D space for all pixel rays


#Stratified Sampling: divide the population into bins and sample one point uniformly from them

points_stratified = 100
t_f = 10 # Far Bound
t_n = 60 # Near Bound
bins = (t_f-t_n)/points_stratified # Split rays into a number (points_stratified) of bins

sample_points_stratified = []
for i in range(1,points_stratified):
  sample_points_stratified.append(np.random.uniform(t_n+(i-1)*bins,t_n+(i)*bins)) #Points along ray to be sampled. Using this method, the same places along every ray sampled

coords_stratified01 = []
for j in range(0,len(unit_vectors01)): #Go through each of the direction unit vectors, each corresponding to a pixel
  sample_locations_stratified = []
  for i in range(0,len(sample_points_stratified)): #Go through all points along a ray and store the coordinates
    sample_locations_stratified.append(origin01+sample_points_stratified[i]*unit_vectors01[j])
  coords_stratified01.append(sample_locations_stratified)