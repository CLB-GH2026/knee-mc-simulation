import numpy as np
import pmcx
import jdata as jd
from matplotlib import pyplot as plt
import iso2mesh as i2m

shape=[]
shape.append({'Grid':{'Tag':1, 'Size':[40,60,30]}})
shape.append({'Sphere':{'Tag':2, 'O':[20,30,10], 'R':10}})

cfg = {}
cfg['shapes']=jd.show({'Shapes':shape}, {'string':True})
cfg['prop']=[[0,0,1,1],[0.005,0.1,0.01,1.37],[0.1, 10, 0.9, 1]]
cfg['nphoton']=1e7
cfg['vol']=np.ones([60, 60, 60], dtype='uint8')
#cfg['vol'][20:40, 30:40, 20:30]=2
cfg['tstart']=0
cfg['tend']=5e-9
cfg['tstep']=5e-9
cfg['srcpos']=[30,30,0]
cfg['srcdir']=[0,0,1]


cfg['shapes']
res = pmcx.mcxlab(cfg)

plt.imshow(np.log10(res['flux'][:,:,3]))
plt.show()
