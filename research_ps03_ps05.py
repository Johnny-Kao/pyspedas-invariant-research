"""PS03/PS05 correctness checks against analytic and SciPy references."""
import numpy as np
import pyspedas as ps
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from numpy.testing import assert_allclose
import unittest

class TimeDeepDive(unittest.TestCase):
    def test_irregular_multicomponent(self):
        rng=np.random.default_rng(354)
        for k in range(50):
            x=np.r_[0,np.cumsum(rng.integers(1,100,15))]
            y=rng.normal(size=(len(x),3,3))
            t=rng.integers(-10,x[-1]+10,120)
            expected=interp1d(x,y,axis=0,fill_value='extrapolate')(t)
            out=ps.time_interpolate(dict(x=x,y=y),t)['y']
            assert_allclose(out,expected,rtol=1e-12,atol=1e-12)
    def test_exact_nanosecond_grid(self):
        t0=np.datetime64('2026-11-25','ns')
        x=t0+np.arange(0,200,2)*np.timedelta64(1,'ns')
        t=t0+np.arange(199)*np.timedelta64(1,'ns')
        y=np.arange(0,200,2,dtype=float)
        for method in ['linear','quadratic','spline']:
            assert_allclose(ps.time_interpolate(dict(x=x,y=y),t,method=method)['y'],np.arange(199),atol=1e-10)
    def test_gap_mask(self):
        rng=np.random.default_rng(355)
        for k in range(30):
            x=np.cumsum(rng.integers(1,10,20))
            y=rng.normal(size=(20,3))
            y[rng.random(y.shape)<.3]=np.nan
            t=np.arange(x[0],x[-1]+1)
            actual=ps.time_interpolate(dict(x=x,y=y),t,ignore_nans=True,max_gap_time=12,max_gap_samples=2)['y']
            ref=np.full_like(actual,np.nan)
            for c in range(3):
                good=np.flatnonzero(np.isfinite(y[:,c]))
                if len(good)<2: continue
                ref[:,c]=interp1d(x[good],y[good,c],fill_value='extrapolate')(t)
                for lo,hi in zip(good[:-1],good[1:]):
                    if x[hi]-x[lo]>12 or hi-lo-1>2:
                        ref[(t>x[lo])&(t<x[hi]),c]=np.nan
            assert_allclose(actual,ref,atol=1e-12,equal_nan=True)
    def test_stored_target_order(self):
        x=[0,2,4]; t=[3,0,1,3,4]; y=[1.,5.,9.]
        ps.store_data('ps03_source',data=dict(x=x,y=y))
        expected=ps.time_interpolate(dict(x=x,y=y),t)
        ps.time_interpolate('ps03_source',t,newname='ps03_out')
        actual=ps.get_data('ps03_out',dt=True)
        np.testing.assert_array_equal(actual.times,expected['x'])
        assert_allclose(actual.y,expected['y'])
        ps.time_interpolate(dict(x=x,y=y),t,newname='ps03_mapping')
        actual=ps.get_data('ps03_mapping',dt=True)
        np.testing.assert_array_equal(actual.times,expected['x'])
        assert_allclose(actual.y,expected['y'])
    def test_bin_map_consistency(self):
        source=dict(x=[0,2,4],y=[[10,20],[30,40],[50,60]],v=[[1,2],[3,4],[5,6]])
        t=[3,0,1,3,4]
        expected=ps.time_interpolate(source,t)
        ps.time_interpolate(source,t,newname='ps03_bins')
        actual=ps.get_data('ps03_bins',dt=True)
        np.testing.assert_array_equal(actual.times,expected['x'])
        assert_allclose(actual.y,expected['y'])
        assert_allclose(actual.v,expected['v'])
    def test_repeat_extrapolation_all_nonfinite(self):
        # No finite endpoints; record behavior without assuming an undocumented fallback.
        out=ps.time_interpolate(dict(x=[0,1],y=[np.inf,-np.inf]),[-1,2],repeat_extrapolate=True)
        print('PS03 no finite endpoint exploratory:',out['y'].tolist())

class RotationDeepDive(unittest.TestCase):
    def test_random_scipy_reference(self):
        rng=np.random.default_rng(351)
        max_err=0.
        for k in range(100):
            x=np.cumsum(rng.integers(1,100,12))
            rot=Rotation.random(12,random_state=rng)
            q=rot.as_quat()[:,[3,0,1,2]]
            q*=rng.choice([-1,1],size=(12,1))
            t=rng.integers(x[0],x[-1]+1,80)
            ref=Slerp(x,rot)(t).as_matrix()
            for left in [False,True]:
                m=rot.as_matrix().copy()
                expected=ref.copy()
                if left:
                    m[:,0,:]*=-1; expected[:,0,:]*=-1
                out=ps.interpolate_rotation(dict(x=x,y=m),t)['y']
                max_err=max(max_err,float(np.max(np.abs(out-expected))))
                assert_allclose(out,expected,atol=1e-10)
                assert_allclose(out@out.transpose(0,2,1),np.broadcast_to(np.eye(3),out.shape),atol=1e-12)
            out=ps.interpolate_rotation(dict(x=x,y=q),t,output='matrix')['y']
            assert_allclose(out,ref,atol=1e-10)
        print('PS05 maximum matrix element error',max_err,'100 series x 80 targets x 3 representations')
    def test_close_angle_precision(self):
        # Existing tolerance can collapse very small rotations; quantify instead of claiming severity.
        for delta in [1e-3,1e-5,1e-6,1e-7]:
            r=Rotation.from_rotvec([[0,0,0],[0,0,delta]])
            out=ps.interpolate_rotation(dict(x=[0,2],y=r.as_matrix()),[1])['y']
            expected=Rotation.from_rotvec([[0,0,delta/2]]).as_matrix()
            print('PS05 small rotation rad',delta,'max abs err',float(np.max(np.abs(out-expected))))
    def test_stored_target_order(self):
        x=[0,2,4];t=[3,0,1,3,4]
        y=Rotation.from_euler('z',[0,45,90],degrees=True).as_matrix()
        expected=ps.interpolate_rotation(dict(x=x,y=y),t)
        ps.interpolate_rotation(dict(x=x,y=y),t,newname='ps05_rotation')
        actual=ps.get_data('ps05_rotation',dt=True)
        np.testing.assert_array_equal(actual.times,expected['x'])
        assert_allclose(actual.y,expected['y'])
    def test_gap_nan_whole_sample(self):
        m=Rotation.from_euler('z',[0,30,60,90],degrees=True).as_matrix()
        m[1,0,0]=np.nan;m[2,1,1]=np.nan
        out=ps.interpolate_rotation(dict(x=[0,1,2,3],y=m),[3,1.5,0,1.5],ignore_nans=True,max_gap_samples=1)['y']
        assert_allclose(out[[0,2]],m[[3,0]])
        self.assertTrue(np.isnan(out[[1,3]]).all())

if __name__=='__main__':
    unittest.main(verbosity=2)
