"""Research-only PS02 differential checks for avg_data(width, trange)."""
import sys
import warnings
import unittest
import numpy as np
import pyspedas as ps
from numpy.testing import assert_allclose, assert_array_equal

class WidthRangeTests(unittest.TestCase):
    def tearDown(self):
        ps.del_data('ps02_*')

    def compare_window(self, x, y, start, end, width, v=None):
        mask=(x>=start)&(x<=end)
        n=int(mask.sum())//width
        assert n>=2
        data={'x':x,'y':y}
        selected={'x':x[mask],'y':y[mask]}
        if v is not None:
            data['v']=v
            selected['v']=v[mask] if v.ndim==2 else v
        ps.store_data('ps02_full',data=data)
        ps.store_data('ps02_clip',data=selected)
        ps.avg_data('ps02_full',trange=[start,end],width=width,newname='ps02_out')
        ps.avg_data('ps02_clip',trange=[start,end],width=width,newname='ps02_ref')
        out=ps.get_data('ps02_out')
        ref=ps.get_data('ps02_ref')
        assert out is not None and ref is not None
        expected=np.nanmean(y[mask][:n*width].reshape((n,width)+y.shape[1:]),axis=1)
        assert_allclose(ref.y,expected,equal_nan=True,rtol=1e-12,atol=1e-12)
        assert_allclose(out.times,ref.times,rtol=0,atol=1e-8)
        assert_allclose(out.y,expected,equal_nan=True,rtol=1e-12,atol=1e-12)
        if v is not None:
            ev=np.nanmean(v[mask][:n*width].reshape(n,width,v.shape[1]),axis=1) if v.ndim==2 else v
            assert_allclose(out[2],ev,equal_nan=True,rtol=1e-12,atol=1e-12)
        assert_array_equal(ps.get_data('ps02_full').y,y)

    def test_offset_scalar(self):
        self.compare_window(np.arange(1.,9.),np.array([100.,200.,3.,4.,5.,6.,700.,800.]),3.,6.,2)
    def test_vector_and_dynamic_bins(self):
        x=np.arange(1.,15.)
        y=np.column_stack((x,x*10))
        self.compare_window(x,y,4.,11.,3,y+100)
    def test_static_bins(self):
        x=np.arange(1.,15.)
        self.compare_window(x,np.column_stack((x,x*10)),4.,11.,3,np.array([10.,20.]))
    def test_irregular_and_nan(self):
        x=np.array([0.,1.,2.,5.,6.,8.,13.,20.,21.,24.,30.,40.])
        y=np.column_stack((x,x*10))
        y[3,0]=np.nan; y[4:6,1]=np.nan
        self.compare_window(x,y,2.,24.,2,y+100)
    def test_prefix_window_control(self):
        x=np.arange(1.,15.)
        self.compare_window(x,x,1.,8.,2)
    def test_full_range_control(self):
        x=np.arange(1.,15.)
        self.compare_window(x,x,1.,14.,3)
    def test_outside_source_control(self):
        x=np.arange(1.,15.)
        self.compare_window(x,x,-5.,20.,3)
    def test_res_priority_control(self):
        x=np.arange(1.,15.)
        ps.store_data('ps02_res',data=dict(x=x,y=x))
        ps.avg_data('ps02_res',trange=[3.,11.],res=2,newname='ps02_res_a')
        ps.avg_data('ps02_res',trange=[3.,11.],res=2,width=3,newname='ps02_res_b')
        a,b=[ps.get_data(n) for n in ['ps02_res_a','ps02_res_b']]
        assert_allclose(a.y,b.y)
        assert_allclose(a.times,b.times)
    def test_random_windows(self):
        rng=np.random.default_rng(20261125)
        for k in range(30):
            x=np.cumsum(rng.integers(1,8,36)).astype(float)
            y=rng.normal(size=(36,3))
            lo=int(rng.integers(1,12)); hi=int(rng.integers(25,35))
            self.compare_window(x,y,x[lo],x[hi],int(rng.integers(2,6)))

def real(mode):
    # Use the same mission, date, instrument and variable as upstream's IDL validation.
    ps.projects.themis.esa(trange=['2010-02-13/00:00:00','2010-02-13/11:59:59'],probe='b')
    data=ps.get_data('thb_peir_en_eflux')
    assert data is not None and len(data.times)>100
    x=np.asarray(data.times); y=np.asarray(data.y); v=np.asarray(data[2])
    # Deliberately select an interior window of real observations, not synthetic values.
    # Choose a window with finite comparable values and variable flux.
    width=4; n=64; start=len(x)//3
    stop=start+n*width
    assert stop<len(x)
    start_time,end_time=x[start],x[stop-1]
    mask=(x>=start_time)&(x<=end_time)
    assert int(mask.sum())==n*width
    ps.avg_data('thb_peir_en_eflux',trange=[start_time,end_time],width=width,newname='ps02_real')
    d=ps.get_data('ps02_real')
    exp=np.nanmean(y[mask].reshape(n,width,y.shape[1]),axis=1)
    valid=np.isfinite(exp)&np.isfinite(d.y)
    assert valid.any()
    difference=np.abs(d.y-exp)
    print('THEMIS actual source samples',len(x),'window samples',int(mask.sum()),'output bins',n,'channels',y.shape[1])
    print('THEMIS compared finite values',int(valid.sum()),'maximum absolute flux error',float(np.max(difference[valid])),'median absolute flux error',float(np.median(difference[valid])))
    print('THEMIS mismatched values',int((~np.isclose(d.y,exp,rtol=1e-12,atol=1e-12,equal_nan=True)).sum()))
    ok=np.allclose(d.y,exp,rtol=1e-12,atol=1e-12,equal_nan=True)
    print('THEMIS range average correct',ok)
    assert bool(ok)==(mode=='fixed')
    if mode=='fixed' and v.ndim==2:
        ev=np.nanmean(v[mask].reshape(n,width,v.shape[1]),axis=1)
        assert_allclose(d[2],ev,equal_nan=True,rtol=1e-12,atol=1e-12)

if __name__=='__main__':
    if sys.argv[1]=='real':
        real(sys.argv[2])
    else:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(WidthRangeTests)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        assert not result.errors
        if sys.argv[2]=='baseline':
            expected={'test_offset_scalar','test_vector_and_dynamic_bins','test_static_bins','test_irregular_and_nan','test_random_windows'}
            assert {case._testMethodName for case,_ in result.failures}==expected
        else:
            assert result.wasSuccessful()
