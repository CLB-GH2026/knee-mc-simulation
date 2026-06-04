import numpy as np
import pmcx
import jdata as jd
from matplotlib import pyplot as plt
import trimesh
import plotly.graph_objects as go

mesh = trimesh.load('patella_raw.stl')
# Voxelize at desired resolution (pitch = voxel size in mesh units)
voxel_size = 1.0
voxels = mesh.voxelized(pitch= voxel_size).fill() # adjust pitch to control resolution

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
cfg['nphoton']=1e6
cfg['vol'] = volume
cfg['dim'] = [volume.shape[0], volume.shape[1], volume.shape[2]]   #removed voxel_size scaler
cfg['tstart']=0
cfg['tend']=5e-9
cfg['tstep']=5e-9
cfg['srcpos']=[volume.shape[0],volume.shape[1]*0.7,volume.shape[2]*0.19],
cfg['srcdir']=[-1,0,0]
cfg['prop']=[[0,0,1,1],[0.1, 10, 0.9, 1.37]]
#cfg['detpos']=[[30,27,0,1], [30,25,0,1]]    # to detect photons, one must first define detectors
#cfg['issavedet']=1      # cfg.issavedet must be set to 1 or True in order to save detected photons
#cfg['issrcfrom0']=1     # set this flag to ensure src/det coordinates align with voxel space
#cfg['savedetpos']=[[30,27,0,1]]
#cfg['savedetflag']='dpx'
cfg.keys()
print('dim:', cfg['dim'])
print('srcpos:', cfg['srcpos'])

res = pmcx.mcxlab(cfg)
#res['detp'].keys()

flux = res['flux']    # shape: (X,Y,Z,T)

flux = np.squeeze(res['flux'])   # now (X, Y, Z)
flux_log = np.log10(flux + 1e-10)

volume = cfg['vol']    #your tissue label array
x, y, z = np.mgrid[0:flux.shape[0], 0:flux.shape[1], 0:flux.shape[2]]

# Tissue boundary as isosurface
fig = go.Figure()

fig.add_trace(go.Isosurface(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=volume.flatten().astype(float),
    isomin=0.5, isomax=1.5,
    surface_count=2,
    colorscale=[[0,'darkblue'],[1,'darkblue']],
    opacity=0.5,
    showscale=False,
    name='Tissue'
))

# Fluence volume
fig.add_trace(go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=flux_log.flatten(),
    isomin=flux_log.max() - 3,
    isomax=flux_log.max(),
    opacity=0.5,
    surface_count=15,
    colorscale='Hot',
    colorbar=dict(title='log10(fluence)'),
    name='Fluence'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (mm)', range = [0, flux.shape[0]]),
        yaxis=dict(title='Y (mm)', range = [0, flux.shape[1]]),
        zaxis=dict(title='Z (mm)', range = [0, flux.shape[2]]),
        aspectmode='data',
        aspectratio=dict(
            x=flux.shape[0],
            y=flux.shape[1],
            z=flux.shape[2]
        )
    ),
    autosize=True,
    width=1000,
    height=1000,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",
    title='3D Fluence Map'
)
fig.show()

#plt.hist(res['detp']['ppath'][:,0], bins=100, range=[0,200]);
#plt.show()

#jd.save(res, 'mcx_flux_dept.json', {'compression':'zlib'})

