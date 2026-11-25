"""PS04 evidence only: no production source edits."""
from copy import deepcopy
import numpy as np
from pyspedas.particles.spd_part_products.spd_pgs_make_e_spec import spd_pgs_make_e_spec
from pyspedas.projects.mms.particles.mms_pgs_make_e_spec import mms_pgs_make_e_spec
from pyspedas.projects.erg.satellite.erg.particle.erg_pgs_make_e_spec import erg_pgs_make_e_spec
from pyspedas.particles.spd_part_products.spd_pgs_limit_range import spd_pgs_limit_range
from pyspedas.particles.spd_part_products.spd_pgs_do_fac import spd_pgs_do_fac
from pyspedas.particles.spd_part_products.spd_pgs_regrid import spd_pgs_regrid
from pyspedas.particles.moments.spd_pgs_moments import spd_pgs_moments
import sys

def counts(before, after, bins):
    changed = ~((before == after) | (np.isnan(before) & np.isnan(after)))
    return int(changed.sum()), int((changed & (bins != 0)).sum())

def compare(a,b):
    return np.allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True)

def synthetic():
    data = dict(data=np.array([[2.,6.],[4.,8.]]),
                bins=np.array([[1,0],[1,1]]),
                energy=np.array([[10.,10.],[20.,20.]]),orig_energy=np.array([10.,20.]))
    for f in [spd_pgs_make_e_spec,mms_pgs_make_e_spec,erg_pgs_make_e_spec]:
        d=deepcopy(data)
        _,out=f(d)
        print(f.__name__, "output", out.tolist(), "changed/all-active", counts(data['data'],d['data'],d['bins']))
    # Model the actual four-dimensional shape passed by upstream's ERG test.
    d=dict(data=np.arange(1.,17.).reshape(2,2,2,2),
           bins=np.ones((2,2,2,2),int),
           energy=np.broadcast_to(np.array([10.,20.])[:,None,None,None],(2,2,2,2)).copy())
    d['bins'][0,0,0,0]=0
    before=d['data'].copy()
    spd_pgs_make_e_spec(d)
    print("generic 4D changed/all-active",counts(before,d['data'],d['bins']))
    # Run MMS helpers in their driver order, with a fixed mask.
    shape=(2,4)
    d=dict(data=np.array([[2.,6.,10.,14.],[4.,8.,12.,16.]]),
           bins=np.ones(shape),energy=np.broadcast_to(np.array([10.,20.])[:,None],shape).copy(),
           orig_energy=np.array([10.,20.]),theta=np.zeros(shape),
           phi=np.tile([45.,135.,225.,315.],(2,1)),dtheta=np.full(shape,180.),
           dphi=np.full(shape,90.),denergy=np.full(shape,10.),
           mass=.0104535,charge=1.,magf=np.array([0.,0.,1.]),species='i',units_name='eflux')
    spd_pgs_limit_range(d,phi=[0.,180.])
    a,b=deepcopy(d),deepcopy(d)
    mms_pgs_make_e_spec(a)
    print("MMS fixed-mask changed/all-active",counts(b['data'],a['data'],a['bins']))
    ma,mb=spd_pgs_moments(a),spd_pgs_moments(b)
    print("MMS fixed-mask moments identical",all(compare(ma[k],mb[k]) for k in ['density','velocity','avgtemp','ptens']))
    ra,rb=[spd_pgs_regrid(spd_pgs_do_fac(z,np.eye(3)),[4,1]) for z in (a,b)]
    print("MMS FAC active regrid values identical",compare(ra['data'][ra['bins']!=0],rb['data'][rb['bins']!=0]))
    print("MMS FAC spectra identical",compare(mms_pgs_make_e_spec(ra)[1],mms_pgs_make_e_spec(rb)[1]))

def erg_real():
    import pyspedas
    from pyspedas.projects.erg.satellite.erg.particle.erg_lepe_get_dist import erg_lepe_get_dist
    from pyspedas.projects.erg.config import set_attitude_source,set_satellite_source
    set_attitude_source('jaxa')
    set_satellite_source('jaxa')
    trange=['2017-04-05 21:45:00','2017-04-05 22:45:00']
    pyspedas.projects.erg.lepe(trange=trange,datatype='3dflux')
    d=erg_lepe_get_dist('erg_lepe_l2_3dflux_FEDU',trange=trange)
    if not isinstance(d,dict):
        raise RuntimeError("ERG data unavailable")
    before=d['data'].copy()
    bins=d['bins'].copy()
    print("ERG native test distribution",before.shape,"active",int((bins!=0).sum()),"inactive",int((bins==0).sum()))
    y,s=spd_pgs_make_e_spec(d)
    print("ERG native test path changed/all-active",counts(before,d['data'],bins),"spectrum shape",np.shape(s))
    # Isolate mask semantics only, not physical solid-angle weighting.
    expected=np.nanmean(np.where(bins==0,0.,before),axis=1)
    print("ERG actual helper vs exact-element mask mean identical",compare(s,expected))

def mms_real():
    import pyspedas
    from pyspedas.projects.mms.particles.mms_part_getspec import mms_part_getspec
    from pyspedas.tplot_tools import get_data
    common=dict(trange=['2015-10-16/13:06:07','2015-10-16/13:06:10'],data_rate='brst',species='i',
                theta=[0,90],phi=[0,100],energy=[1000,20000])
    first=mms_part_getspec(**common,output='moments',suffix='_ps04_mom')
    names=['density','velocity','avgtemp','ptens']
    def read(suffix):
        result={}
        for n in names:
            x=get_data('mms1_dis_dist_brst_'+n+suffix)
            if x is None:
                raise RuntimeError("missing MMS output "+n+suffix)
            result[n]=np.array(x.y,copy=True)
        return result
    a=read('_ps04_mom')
    second=mms_part_getspec(**common,output='energy theta phi moments',suffix='_ps04_all')
    b=read('_ps04_all')
    print("MMS public driver samples",a['density'].shape)
    print("MMS moments-only vs all products",{n:compare(a[n],b[n]) for n in names})

if __name__=='__main__':
    {'synthetic':synthetic,'erg':erg_real,'mms':mms_real}[sys.argv[1]]()
