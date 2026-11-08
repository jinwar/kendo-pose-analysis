import unittest
import numpy as np
from average_strikes import weighted_position,axial_mean


class AverageTests(unittest.TestCase):
    def test_identical_observations_have_zero_spread(self):
        result=weighted_position([[2,3]]*5,[.7]*5)
        np.testing.assert_allclose(result['mean_xy'],[2,3])
        np.testing.assert_allclose(result['covariance_px2'],np.zeros((2,2)),atol=1e-12)

    def test_confidence_exclusion_and_minimum_support(self):
        result=weighted_position([[0,0],[1,0],[2,0],[999,999],[np.nan,0]],[1,1,1,.1,1])
        self.assertEqual(result['n'],3)
        np.testing.assert_allclose(result['mean_xy'],[1,0])
        self.assertIsNone(weighted_position([[0,0],[1,0]],[1,1]))

    def test_translation_equivariance(self):
        points=np.array([[0,0],[1,2],[2,3],[2,1],[3,2]])
        a=weighted_position(points,[.8]*5);b=weighted_position(points+[700,-400],[.8]*5)
        np.testing.assert_allclose(np.array(b['mean_xy'])-a['mean_xy'],[700,-400])
        np.testing.assert_allclose(a['covariance_px2'],b['covariance_px2'])

    def test_distant_outlier_downweighted_and_recorded(self):
        r=weighted_position([[0,0],[1,0],[2,0],[3,0],[1000,0]],[1]*5)
        self.assertEqual(r['downweighted_strike_indices'],[4])
        self.assertLess(r['mean_xy'][0],50)

    def test_shaft_orientation_wrap(self):
        theta,concentration=axial_mean(np.deg2rad([179,1]),[1,1])
        self.assertLess(abs(theta),1e-10)
        self.assertGreater(concentration,.99)


if __name__=='__main__':unittest.main()
