import unittest
import numpy as np
from tracking import Tracker, iou


class TrackingTests(unittest.TestCase):
    def test_detection_order_does_not_change_ids(self):
        tracker = Tracker()
        boxes = np.array([[0,0,100,200],[300,0,400,200]],dtype=float)
        self.assertEqual(tracker.update(boxes),[1,2])
        self.assertEqual(tracker.update(boxes[::-1]+[3,0,3,0]),[2,1])

    def test_short_gap_preserves_id_without_emitting_a_pose(self):
        tracker = Tracker(max_missed=2)
        box = np.array([[0,0,100,200]],dtype=float)
        self.assertEqual(tracker.update(box),[1])
        self.assertEqual(tracker.update(np.empty((0,4))),[])
        self.assertEqual(tracker.update(box),[1])

    def test_expired_track_gets_new_id(self):
        tracker = Tracker(max_missed=1)
        box = np.array([[0,0,100,200]],dtype=float)
        tracker.update(box)
        tracker.update(np.empty((0,4)))
        tracker.update(np.empty((0,4)))
        self.assertEqual(tracker.update(box),[2])

    def test_distant_new_person_is_not_forced_into_existing_track(self):
        tracker = Tracker()
        tracker.update(np.array([[0,0,100,200]],dtype=float))
        self.assertEqual(tracker.update(np.array([[1000,0,1100,200]],dtype=float)),[2])

    def test_degenerate_box_iou(self):
        self.assertEqual(iou(np.zeros(4),np.zeros(4)),0)


if __name__ == '__main__':
    unittest.main()
