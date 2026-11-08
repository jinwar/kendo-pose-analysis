import unittest
import numpy as np
from compare import Models
from video_timing import frame_range


class PipelineTests(unittest.TestCase):
    def test_interval_selection_uses_actual_variable_frame_timestamps(self):
        self.assertEqual(frame_range([0,.03,.06,.20,.23,.27],.05,.19),(2,5))
        self.assertEqual(frame_range([0,.03,.06],.04),(2,3))

    def test_rtmpose_no_detections_does_not_infer_full_frame_person(self):
        models=Models.__new__(Models)
        models.detector=lambda frame: np.empty((0,4))
        def forbidden(*args,**kwargs):
            self.fail('RTMPose must not run on an empty detection set')
        models.pose=forbidden
        boxes,points,scores,confidence=models.infer('rtmpose',np.zeros((64,64,3),dtype=np.uint8))
        self.assertEqual(boxes.shape,(0,4))
        self.assertEqual(points.shape,(0,17,2))
        self.assertEqual(scores.shape,(0,17))
        self.assertEqual(confidence,[])


if __name__=='__main__':
    unittest.main()
