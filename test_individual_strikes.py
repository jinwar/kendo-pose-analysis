import unittest
import numpy as np
from compare_individual_strikes import origin


class ContactAnchorTests(unittest.TestCase):
    def test_followthrough_does_not_move_contact_reference(self):
        def row(offset,xy):
            return {'offset_frames':offset,'receiver':{'keypoints_xy':[xy]*17,'keypoint_scores':[.9]*17}}
        sequence=[row(-1,[90,190]),row(0,[100,200]),row(6,[500,600])]
        np.testing.assert_allclose(origin(sequence),[100,200])


if __name__=='__main__':unittest.main()
