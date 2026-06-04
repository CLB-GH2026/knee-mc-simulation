import numpy as np
import pmcx
import jdata as jd
from matplotlib import pyplot as plt
import trimesh

vol_shape = (64,64,64)  # define your grid size
volume = np.zeros(vol_shape, dtype=np.uint8)
voxel_size = 1.0

# Load STL file
meshes = {
    1: 'acl.stl',
    2: 'lcl.stl',
    3: 'mcl.stl',
    4: 'pcl.stl',
}

for label, path in meshes.items():
    mesh = trimesh.load(path)
    # Voxelize at desired resolution (pitch = voxel size in mesh units)
    voxels = mesh.voxelized(pitch=voxel_size).fill()  # adjust pitch to control resolution
    mat = voxels.matrix.astype(np.uint8) * label
    # Place into volume at correct offset
    origin = voxels.translation.astype(int)
    slices = tuple(slice(o, o + s) for o, s in zip(origin, mat.shape))
    volume[slices] = np.where(mat > 0, label, volume[slices])

voxel_grid = voxels.matrix  # shape: (X, Y, Z), dtype: bool

# Convert to uint8 for use as tissue labels
volume = voxel_grid.astype(np.uint8)

# Save as .npy for later use
np.save('volume.npy', volume)

cfg = {}
cfg['nphoton']=1e6
cfg['vol'] = volume
cfg['dim'] = [64, 64, 64]
cfg['tstart']=0
cfg['tend']=5e-9
cfg['tstep']=5e-9
cfg['srcpos']=[32,32,0],
cfg['srcdir']=[0,0,1]
cfg['prop']=[[0,0,1,1],[0.1, 10, 0.9, 1.37]]
#cfg['detpos']=[[30,27,0,1], [30,25,0,1]]    # to detect photons, one must first define detectors
#cfg['issavedet']=1      # cfg.issavedet must be set to 1 or True in order to save detected photons
#cfg['issrcfrom0']=1     # set this flag to ensure src/det coordinates align with voxel space
#cfg['savedetpos']=[[30,27,0,1]]
#cfg['savedetflag']='dpx'
cfg.keys()

res = pmcx.mcxlab(cfg)
#res['detp'].keys()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each tissue label with a different color
colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'grey'}  # map label -> color
color_vol = np.empty(volume.shape, dtype=object)

for label, color in colors.items():
    color_vol[volume == label] = color

filled = volume > 0  # only show non-background voxels
ax.voxels(filled, facecolors=color_vol, edgecolor='none', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Voxel Volume')
plt.show()

#plt.imshow(np.log10(res['flux'][10,:, :]))   # Based on error modifying 30 to 18 Axis 0
#plt.show()

#plt.hist(res['detp']['ppath'][:,0], bins=100, range=[0,200]);
#plt.show()

# plot photon existing position for Det#1 in red
#plt.scatter(res['detp']['p'][res['detp']['detid']==1,0], res['detp']['p'][res['detp']['detid']==1,1],
#            marker='.',color='red');
# plot photon existing position for Det#2 in blue
#plt.scatter(res['detp']['p'][res['detp']['detid']==2,0], res['detp']['p'][res['detp']['detid']==2,1],
#            marker='.',color='blue');
#plt.axis('equal');
#plt.show()

#jd.save(res, 'mcx_flux_dept.json', {'compression':'zlib'})

