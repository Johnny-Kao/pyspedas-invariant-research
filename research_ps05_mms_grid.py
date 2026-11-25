"""MMS quaternion time-grid alignment: deterministic real API reproduction."""
import numpy as np
import pyspedas as ps
from scipy.spatial.transform import Rotation,Slerp
from spacepy.coordinates import quaternionRotateVector
from pyspedas.projects.mms.cotrans.mms_cotrans_qrotate import mms_cotrans_qrotate

def run(times,name,inverse=False):
    ref_times=np.array([0.,2.,4.])
    rotations=Rotation.from_euler('z',[0,90,180],degrees=True)
    q=rotations.as_quat()
    vectors=np.tile([1.,0.,0.],(len(times),1))
    ps.store_data('ps05_mms_q',data={'x':ref_times,'y':q})
    ps.store_data('ps05_mms_v',data={'x':times,'y':vectors})
    mms_cotrans_qrotate('ps05_mms_v','ps05_mms_q',name,'GSE',inverse=inverse)
    out=ps.get_data(name).y
    interpolated=Slerp(ref_times,rotations)(times)
    expected=(interpolated.inv() if inverse else interpolated).apply(vectors)
    err=np.linalg.norm(out-expected,axis=1)
    angles=np.rad2deg(np.arccos(np.clip(np.sum(out*expected,axis=1),-1,1)))
    print(name,'inverse',inverse,'actual',out.tolist(),'expected',expected.tolist(),'vector_error',err.tolist(),'angle_error_deg',angles.tolist(),'PASS',bool(np.allclose(out,expected,atol=1e-10)))
    return np.allclose(out,expected,atol=1e-10)

if __name__=='__main__':
    aligned=run(np.array([0.,2.,4.]),'ps05_aligned')
    different=run(np.array([.5,1.5,2.5,3.]),'ps05_different_count')
    same=run(np.array([.5,1.5,3.]),'ps05_same_count')
    inv=run(np.array([.5,1.5,3.]),'ps05_inverse',True)
    assert aligned and different, 'control failed; cannot interpret candidate'
    assert same and inv, 'same count different times rotates using incorrect attitude'
