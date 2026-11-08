"""Small, shared motion/IoU tracker. IDs are local to each model run.

This baseline does not use appearance/ReID and cannot guarantee identity through
long occlusions. It never emits extrapolated poses during missed detections.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    overlap = np.prod(np.maximum(0, rb - lt))
    area_a = np.prod(np.maximum(0, a[2:] - a[:2]))
    area_b = np.prod(np.maximum(0, b[2:] - b[:2]))
    return float(overlap / max(area_a + area_b - overlap, 1e-6))


class Tracker:
    def __init__(self, max_missed=15):
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 1
        self.frame = 0

    def update(self, boxes):
        self.frame += 1
        self.tracks = {k: v for k, v in self.tracks.items()
                       if self.frame - v['frame'] <= self.max_missed + 1}
        keys = list(self.tracks)
        assignments = {}
        if keys and len(boxes):
            cost = np.full((len(keys), len(boxes)), 1e6)
            for i, key in enumerate(keys):
                state = self.tracks[key]
                predicted = state['box'] + state['velocity'] * (self.frame - state['frame'])
                center = (predicted[:2] + predicted[2:]) / 2
                size = max(np.linalg.norm(predicted[2:] - predicted[:2]), 1)
                for j, box in enumerate(boxes):
                    distance = np.linalg.norm(center - (box[:2] + box[2:]) / 2) / size
                    overlap = iou(predicted, box)
                    if overlap >= .05 or distance <= .35:
                        cost[i, j] = 1 - overlap + .5 * distance
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e5:
                    assignments[j] = keys[i]
        ids = []
        for j, box in enumerate(boxes):
            key = assignments.get(j)
            if key is None:
                key = self.next_id
                self.next_id += 1
                velocity = np.zeros(4)
            else:
                old = self.tracks[key]
                measured = (box - old['box']) / (self.frame - old['frame'])
                velocity = .5 * old['velocity'] + .5 * measured
            self.tracks[key] = {'box': box.copy(), 'velocity': velocity, 'frame': self.frame}
            ids.append(key)
        return ids
