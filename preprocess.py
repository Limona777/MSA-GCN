import json
import numpy as np
import os
from typing import List, Dict

class SkeletonPreprocessor:
    def __init__(self, frame : int = 60, joints : int = 17):
        self.frame = frame
        self.joints = joints

    def load(self, path : str) -> Dict:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def single(self, frame : Dict) -> np.ndarray:
        ske = np.zeros((self.joints, 3))

        if 'body' not in frame or not frame['body']:
            return ske

        body = frame['body']
        for j in body:
            id = int(j[3])
            if id < self.joints:
                ske[id] = [j[0], j[1], j[2]]

        return ske

    def sequence(self, jfiles : List[str]) -> np.ndarray:
        seq = []

        for j in jfiles[:self.frame]:
            data = self.load(j)
            ske = self.single(data)
            seq.append(ske)

        while len(seq) < self.frame:
            seq.append(np.zeros((self.joints, 3)))

        return np.array(seq)

    def normalize(self, seq):
        if np.all(seq == 0):
            return seq

        centre = seq[:, 0:1, :2].copy()
        seq[:, :, :2] -= centre

        max_val = np.max(np.abs(seq[:, :, :2]))
        if max_val > 1e-6:
            seq[:, :, :2] /= max_val

        return seq