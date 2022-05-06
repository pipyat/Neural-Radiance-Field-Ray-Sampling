def get_rays(H, W, focal_x, focal_y, cam_to_world, Distortion): #cam_to_world is the inverse of extrinsic camera matrix, distortion contains the parameters k1, k2, p1, p2, k3
    x_01, y_01 = np.meshgrid(np.arange(W),np.arange(H)) # Pixel space 

    k1 = float(Distortion[0])
    k2 = float(Distortion[1])
    p1 = float(Distortion[2])
    p2 = float(Distortion[3])
    k3 = float(Distortion[4])
    r = (x_01**2 + y_01**2)**0.5
    x_01 = x_01*((1+k1*r**2+k2*r**4+k3*r**6)/(1)) + 2*p1*x_01*y_01 + p2*(r**2+2*x_01**2)
    y_01 = y_01*((1+k1*r**2+k2*r**4+k3*r**6)/(1)) + 2*p2*x_01*y_01 + p1*(r**2+2*y_01**2)

    # Convert to camera reference frame
    X_camera01 = (x_01-W/2)/focal_x
    Y_camera01 = (y_01-H/2)/focal_y

    R_inv_01 = cam_to_world[:3,:3]
    T_inv_01 = cam_to_world[:3,3]

    homogeneous = np.stack((X_camera01,-Y_camera01,-np.ones(np.shape(X_camera01))),-1) #Homogeneous coordinates - stack camera coordinates
    homogeneous_expanded = homogeneous[..., None, :] # One entry for each of the height pixels, each of those has an entry for each of the width pixels so together we have every combination of coordinates

    # Ray direction unit vectors
    unit_vectors = np.matmul(R_inv_01, np.transpose(homogeneous_expanded[0][0]))/np.linalg.norm(np.matmul(R_inv_01, np.transpose(homogeneous_expanded[0][0]))) # This needs to be removed at the end
    for i in range(0,len(homogeneous_expanded)): 
      for j in range(0,len(homogeneous_expanded[0])): 
        insert = np.matmul(R_inv_01, np.transpose(homogeneous_expanded[i][j]))
        insert = insert/np.linalg.norm(insert)
        unit_vectors = np.row_stack((unit_vectors,insert)) # As values are stacked, we have x,y,z,x,y,z,.... so the length is 3 times the number of vectors we need

    origin = T_inv_01 # The ray origin is the camera to world translation vector, be careful how this is defined since X_c = R*X_w + R*t = R*X_w + T 

    dir = []
    for i in range(len(unit_vectors[3:])//3):
      start = i*3
      dir.append(unit_vectors[3:][start:start+3])

    return origin, dir 
  
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def sample_points(origin, unit_vectors, t_n, t_f, points_stratified):  # r(t) = x_0 + t*d with near and far bounds, t_n and t_f

    # Compute 3D query points
    bins = (t_f-t_n)/points_stratified # Split rays into a number (points_stratified) of bins

    sample_points_stratified = []
    for i in range(0,points_stratified):
      sample_points_stratified.append(np.random.uniform(t_n+(i-1)*bins,t_n+(i)*bins)) #Points along ray to be sampled. Using this method, the same places along every ray sampled

    coords_stratified01 = []
    for j in range(0,len(unit_vectors)): #Go through each of the direction unit vectors, each corresponding to a pixel
      sample_locations_stratified = []
      for i in range(0,len(sample_points_stratified)): #Go through all points along a ray and store the coordinates
        sample_locations_stratified.append(np.asarray(origin)+sample_points_stratified[i]*np.asarray(np.transpose(unit_vectors[0]))) 
      coords_stratified01.append(np.squeeze(sample_locations_stratified)) # 3D array for every point sampled, e.g. the first entry is the first sampled point via the first ray, second entry is the second sampled point via the first ray etc

    return coords_stratified01, sample_points_stratified #Returns samples coordinates for every ray and t values 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
# POSITIONAL ENCODING - allows network to deal with high frequency signals 

def encoding(vector, L):
  enc = []
  for j in vector:
    component = []
    for i in range(L):
      component.append(np.sin(2**i*j))
      component.append(np.cos(2**i*j))
    enc.append(component)
  return enc
