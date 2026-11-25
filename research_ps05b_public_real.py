"""PS05B: actual MMS public transform on original and independently clipped time grids."""
import sys
from pathlib import Path
import numpy as np
import pyspedas as ps
from scipy.spatial.transform import Rotation,Slerp
from pyspedas.projects.mms import mec,mms_qcotrans
from pyspedas.projects.mms.cotrans.mms_cotrans_qrotate import mms_cotrans_qrotate

def public_synthetic():
    t=np.array([0.,2.,4.])
    q=Rotation.from_euler('z',np.array([[0.],[90.],[180.]]),degrees=True).as_quat()
    target=np.array([.5,1.5,3.])
    v=np.tile([1.,0.,0.],(3,1))
    ps.store_data('mms1_mec_quat_eci_to_gse',data=dict(x=t,y=q))
    ps.store_data('mms1_ps05_eci',data=dict(x=target,y=v))
    result=mms_qcotrans('mms1_ps05_eci','mms1_ps05_gse',in_coord='eci',out_coord='gse',probe='1')
    assert result==['mms1_ps05_gse'],result
    out=ps.get_data(result[0])
    expected=Slerp(t,Rotation.from_quat(q))(target).apply(v)
    ok=np.allclose(out.y,expected,atol=1e-10)
    print('PUBLIC API synthetic correct:',ok,'max vector error',np.max(np.linalg.norm(out.y-expected,axis=1)))
    return ok

def error_metrics(actual,ref):
    diff=np.linalg.norm(actual-ref,axis=1)
    cross=np.linalg.norm(np.cross(actual,ref),axis=1)
    dot=np.sum(actual*ref,axis=1)
    angles=np.rad2deg(np.arctan2(cross,dot))
    return {'samples':len(diff),'max_vector_error':float(np.max(diff)),'median_vector_error':float(np.median(diff)),
            'max_angle_deg':float(np.max(angles)),'median_angle_deg':float(np.median(angles))}

def real():
    trange=['2015-10-16/13:00:00','2015-10-16/13:10:00']
    mec(trange=trange,probe='1',time_clip=True)
    names=['mms1_mec_quat_eci_to_sm','mms1_mec_quat_eci_to_gse','mms1_mec_r_sm','mms1_mec_r_gse']
    data={n:ps.get_data(n) for n in names}
    if any(d is None for d in data.values()):
        raise RuntimeError('Missing MEC archive variables: '+str([n for n,d in data.items() if d is None]))
    q1,q2,v,truth=[data[n] for n in names]
    print('MEC native counts and equal grids',[(n,len(data[n].times)) for n in names],np.array_equal(v.times,q1.times),np.array_equal(v.times,q2.times))
    assert np.array_equal(v.times,truth.times)
    # Standard public transform, unmodified archive inputs.
    mms_qcotrans(names[2],'mms1_ps05_native',in_coord='sm',out_coord='gse',probe='1')
    native=ps.get_data('mms1_ps05_native').y
    print('MEC native public transform vs published GSE',error_metrics(native,truth.y))
    # Independently clipped windows, ordinary native samples: both same size but offset.
    # Interior vector window, quaternion windows begin one native sample earlier.
    n=min(len(v.times),len(q1.times),len(q2.times),100)-2
    assert n>=4
    tv=v.times[1:n+1].copy()
    vv=v.y[1:n+1].copy()
    ps.store_data('mms1_ps05_clipped_sm',data=dict(x=tv,y=vv))
    def put_q(name,d,stop):
        ps.store_data(name,data=dict(x=d.times[:stop],y=d.y[:stop]))
    # Different count reference, covering all target times.
    for name,d in zip(names[:2],[q1,q2]): put_q(name,d,n+2)
    mms_qcotrans('mms1_ps05_clipped_sm','mms1_ps05_reference',in_coord='sm',out_coord='gse',probe='1')
    ref=ps.get_data('mms1_ps05_reference').y
    # Keep equal count and same first n-1 source quaternions; last target outside this grid
    # is excluded from impact statistics so no endpoint extrapolation confound.
    for name,d in zip(names[:2],[q1,q2]): put_q(name,d,n)
    print('MEC clipped grid counts',len(tv),len(q1.times[:n]),'offset_seconds',float(tv[0]-q1.times[0]),'native_cadence_seconds',float(np.median(np.diff(q1.times))))
    mms_qcotrans('mms1_ps05_clipped_sm','mms1_ps05_shifted',in_coord='sm',out_coord='gse',probe='1')
    actual=ps.get_data('mms1_ps05_shifted').y
    valid=(tv>=max(q1.times[0],q2.times[0])) & (tv<=min(q1.times[n-1],q2.times[n-1]))
    assert valid.any()
    print('MEC shifted equal-count vs different-count reference interior',error_metrics(actual[valid],ref[valid]))
    print('MEC shifted vs published GSE interior',error_metrics(actual[valid],truth.y[1:n+1][valid]))
    print('MEC reference vs published GSE interior',error_metrics(ref[valid],truth.y[1:n+1][valid]))
    print('MEC aligned result same as reference interior',np.allclose(actual[valid],ref[valid],rtol=0,atol=1e-8))
    return np.allclose(actual[valid],ref[valid],rtol=0,atol=1e-8)

if __name__=='__main__':
    mode=sys.argv[1]
    ok=public_synthetic() if mode=='public' else real()
    expected=sys.argv[2]=='fixed'
    assert bool(ok)==expected,(mode,ok,expected)
