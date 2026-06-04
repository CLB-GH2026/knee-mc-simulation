import numpy as np
import pmcx
import jdata as jd
from matplotlib import pyplot as plt

cfg = {}
cfg['nphoton']=1e7
cfg['vol'] =np.ones([60,60,60],dtype='uint8')
cfg['vol'] [20:40, 30:40, 20:30]=2
cfg['tstart']=0
cfg['tend']=5e-9
cfg['tstep']=5e-9
cfg['srcpos']=[30,30,0],
cfg['srcdir']=[0,0,1]
cfg['prop']=[[0,0,1,1],[0.005,0.1,0.01,1.37],[0.1, 10, 0.9, 1]]
cfg['detpos']=[[30,27,0,1], [30,25,0,1]]    # to detect photons, one must first define detectors
cfg['issavedet']=1      # cfg.issavedet must be set to 1 or True in order to save detected photons
cfg['issrcfrom0']=1     # set this flag to ensure src/det coordinates align with voxel space
cfg['savedetpos']=[[30,27,0,1]]
cfg['savedetflag']='dpx'
cfg.keys()

res = pmcx.mcxlab(cfg)
res['detp'].keys()

plt.imshow(np.log10(res['flux'][30,:, :]))
plt.show()

plt.hist(res['detp']['ppath'][:,0], bins=100, range=[0,200])
plt.show()

# plot photon existing position for Det#1 in red
plt.scatter(res['detp']['p'][res['detp']['detid']==1,0], res['detp']['p'][res['detp']['detid']==1,1],
            marker='.',color='red')
# plot photon existing position for Det#2 in blue
plt.scatter(res['detp']['p'][res['detp']['detid']==2,0], res['detp']['p'][res['detp']['detid']==2,1],
            marker='.',color='blue')
plt.axis('equal')
plt.show()

jd.save(res, 'mcx_flux_dept.json', {'compression':'zlib'})

#Diffuse Reflectance (Photon's remaining weight upon exiting domain surface ('0-Valued voxels')
cfg['vol'][:,:,0]=0
cfg['issaveref']=1
res=pmcx.mcxlab(cfg)
res.keys()
plt.imshow(np.log10(res['dref'][:,:,0]))
plt.colorbar()
plt.show()
