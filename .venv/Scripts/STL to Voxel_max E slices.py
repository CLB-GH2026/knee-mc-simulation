import numpy as np
import pmcx
import jdata as jd
from matplotlib import pyplot as plt
import trimesh

mesh = trimesh.load('femur_cartilage.stl')
# Voxelize at desired resolution (pitch = voxel size in mesh units)
voxels = mesh.voxelized(pitch=0.5).fill() # adjust pitch to control resolution

# Convert to dense numpy boolean array
voxel_grid = voxels.matrix  # shape: (X, Y, Z), dtype: bool

# Convert to uint8 for use as tissue labels
volume = voxel_grid.astype(np.uint8) * 1  # label = 1 for this tissue

# Save as .npy for later use
np.save('volume.npy', volume)

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

cfg = {}
cfg['nphoton']=1e3
cfg['vol'] = volume
cfg['dim'] = [64, 64, 64]
cfg['tstart']=0
cfg['tend']=5e-9
cfg['tstep']=5e-9
cfg['srcpos']=[12,32,0],
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

flux = res['flux']    # shape: (X,Y,Z,T)

flux = np.squeeze(flux)   # now (X, Y, Z)

z_slice = np.argmax(flux.sum(axis=(0,1)))  # Z slice with most energy
flux_slice = np.log10(flux[:, :, z_slice] + 1e-10)

plt.figure(figsize=(6, 5))
plt.imshow(flux_slice.T, cmap='hot', origin='lower')
plt.colorbar(label='log10(fluence)')
plt.title('Fluence Map (Z max E-slice)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

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

