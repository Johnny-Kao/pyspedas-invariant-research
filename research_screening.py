"""Independent research screening of the pinned upstream public APIs."""
import unittest
import numpy as np
import pyspedas as ps
from pyspedas.particles.spd_part_products.spd_pgs_make_e_spec import spd_pgs_make_e_spec

class PS01Coordinates(unittest.TestCase):
    def test_azimuth_range(self):
        a=np.arange(8)*np.pi/4
        x=np.cos(a); y=np.sin(a)
        x[[2,6]]=0; y[[0,4]]=0
        r,theta,phi=ps.cart2spc(x,y,np.zeros(8))
        np.testing.assert_allclose(phi,a,atol=1e-12)
    def test_round_trip(self):
        x=np.array([1,0,-1,0,2,-2,-2,2.])
        y=np.array([0,1,0,-1,2,2,-2,-2.])
        z=np.ones(8)
        r,theta,phi=ps.cart2spc(x,y,z)
        np.testing.assert_allclose(np.array(ps.spc2cart(r,theta,phi)),[x,y,z],atol=1e-12)
    def test_mlt_cardinal(self):
        np.testing.assert_allclose(np.mod(ps.sm2mlt([1.,0,-1,0],[0.,1,0,-1],[0.]*4),24),[12,18,0,6],atol=1e-12)
    def test_goes18_existing_idl_reference(self):
        xyz=np.array([[33199.660,25815.376,2995.6487],[18814.747,37614.243,2990.7136],[-40428.563,-11607.452,2995.3227],[15798.372,-38974.509,2995.0028]])
        np.testing.assert_allclose(ps.sm2mlt(*xyz.T),[14.524527,16.228377,1.0679544,7.4710163],atol=1e-6,rtol=0)

class PS02Averaging(unittest.TestCase):
    def test_subrange_width(self):
        ps.store_data('research_avg',data={'x':np.arange(1,9,dtype=float),'y':[100,200,3,4,5,6,700,800]})
        ps.avg_data('research_avg',trange=[3.,6.],width=2,newname='research_avg_out')
        d=ps.get_data('research_avg_out')
        self.assertIsNotNone(d)
        np.testing.assert_allclose(d.y,[3.5,5.5])
    def test_full_width_control(self):
        ps.store_data('research_avg_control',data={'x':np.arange(1,9,dtype=float),'y':np.arange(1,9,dtype=float)})
        ps.avg_data('research_avg_control',width=2,newname='research_avg_control_out')
        np.testing.assert_allclose(ps.get_data('research_avg_control_out').y,[1.5,3.5,5.5,7.5])

class PS03TimeInterpolation(unittest.TestCase):
    def test_linear_order_and_duplicates(self):
        out=ps.time_interpolate({'x':[0,2,4],'y':[1.,5.,9.]},[3,0,1,3,4])
        np.testing.assert_allclose(out['y'],[7,1,3,7,9])
    def test_descending_equivalence(self):
        x=[0,2,4];y=[1.,5.,9.];t=[3,1,2]
        for m in ['linear','nearest','previous','quadratic','spline']:
            with self.subTest(method=m):
                a=ps.time_interpolate({'x':x,'y':y},t,method=m)
                b=ps.time_interpolate({'x':x[::-1],'y':y[::-1]},t,method=m)
                np.testing.assert_allclose(a['y'],b['y'])
    def test_exact_sample_and_gap(self):
        a=ps.time_interpolate({'x':[0,1,2,3],'y':[0.,np.nan,np.nan,3.]},[0,1.5,3],ignore_nans=True,max_gap_samples=1)
        np.testing.assert_allclose(a['y'],[0,np.nan,3])
    def test_nanosecond_grid(self):
        t=np.datetime64('2020-01-01T00:00:00','ns')
        a=ps.time_interpolate({'x':t+np.array([0,2,4],dtype='timedelta64[ns]'),'y':[0.,2.,4.]},t+np.array([1,3],dtype='timedelta64[ns]'))
        np.testing.assert_allclose(a['y'],[1,3])

class PS04ParticleSpectra(unittest.TestCase):
    def test_input_ownership_exploratory(self):
        data={'data':np.array([[2.,6.],[4.,8.]]),'bins':np.array([[1,0],[1,1]]),'energy':np.array([[10.,10.],[20.,20.]])}
        before=data['data'].copy()
        spd_pgs_make_e_spec(data)
        np.testing.assert_array_equal(data['data'],before)
    def test_all_active_control(self):
        data={'data':np.array([[2.,6.],[4.,8.]]),'bins':np.ones((2,2),int),'energy':np.array([[10.,10.],[20.,20.]])}
        e,y=spd_pgs_make_e_spec(data)
        np.testing.assert_allclose(e,[10,20])
        np.testing.assert_allclose(y,[4,6])

class PS05Rotation(unittest.TestCase):
    def test_sign_equivalence(self):
        q=np.array([[1.,0,0,0],[np.sqrt(.5),0,0,np.sqrt(.5)]])
        a=ps.interpolate_rotation({'x':[0,2],'y':q},[0,1,2],output='matrix')['y']
        b=ps.interpolate_rotation({'x':[0,2],'y':q*np.array([[1],[-1]])},[0,1,2],output='matrix')['y']
        np.testing.assert_allclose(a,b,atol=1e-12)
        np.testing.assert_allclose(a@a.transpose(0,2,1),np.broadcast_to(np.eye(3),a.shape),atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(a),np.ones(3),atol=1e-12)
    def test_exact_and_bounds(self):
        m=np.array([np.eye(3),[[0,-1,0],[1,0,0],[0,0,1]]])
        a=ps.interpolate_rotation({'x':[0,2],'y':m},[-1,0,2,3])['y']
        np.testing.assert_allclose(a,m[[0,0,1,1]],atol=1e-12)

if __name__=='__main__':
    unittest.main(verbosity=2)
