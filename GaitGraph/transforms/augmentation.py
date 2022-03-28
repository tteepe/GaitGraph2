import math
import torch
import numpy as np


class NormalizeEmpty:
    def __call__(self, data):
        # Fix empty detections
        frames, joints = np.where(data.x[:, :, 0].numpy() == 0)
        for frame, joint in zip(frames, joints):
            center_of_gravity = torch.mean(data.x[frame], dim=0)
            data.x[frame, joint, 0] = center_of_gravity[0]
            data.x[frame, joint, 1] = center_of_gravity[1]
            data.x[frame, joint, 2] = 0

        return data


class RandomFlipLeftRight:
    def __init__(self, p=0.5, flip_idx=None):
        self.p = p
        self.flip_idx = flip_idx

    def __call__(self, data):
        if torch.rand(1) > self.p:
            return data

        # Pose LR Flip
        data.x = data.x[:, self.flip_idx]
        return data


class RandomFlipSequence:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if torch.rand(1) > self.p:
            return data

        data.x = data.x.flip(0)
        return data


class PadSequence:
    def __init__(self, sequence_length=25):
        self.sequence_length = sequence_length

    def __call__(self, data):
        input_length = data.x.shape[0]
        if input_length > self.sequence_length:
            return data

        diff = self.sequence_length + 1 - input_length
        len_pre = int(math.ceil(diff / 2))
        len_pos = int(diff / 2) + 1

        while len_pre > data.x.shape[0] or len_pos > data.x.shape[0]:
            data.x = torch.cat([data.x.flip(0), data.x, data.x.flip(0)], 0)

        pre = data.x[1:len_pre].flip(0)
        pos = data.x[-1 - len_pos:-1].flip(0)
        data.x = torch.cat([pre, data.x, pos], 0)[:self.sequence_length]

        return data


class RandomCropSequence:
    def __init__(self, min_sequence_length=20, p=0.25):
        self.min_sequence_length = min_sequence_length
        self.p = p

    def __call__(self, data):
        length = data.x.shape[0]
        if length <= self.min_sequence_length or torch.rand(1) > self.p:
            return data

        sequence_length = int(torch.randint(self.min_sequence_length, length, (1,)))
        start = torch.randint(0, length - sequence_length, (1,))
        end = start + sequence_length
        data.x = data.x[start:end]

        return data


class RandomSelectSequence:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        length = data.x.shape[0]
        if length <= self.sequence_length:
            return data

        start = torch.randint(0, length - self.sequence_length, (1,))
        end = start + self.sequence_length
        data.x = data.x[start:end]

        return data


class SelectSequenceCenter:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        start = int((data.x.shape[0] / 2) - (self.sequence_length / 2))
        end = start + self.sequence_length
        data.x = data.x[start:end]

        return data


class ShuffleSequence:
    def __init__(self, enabled=False):
        self.enabled = enabled

    def __call__(self, data):
        if self.enabled:
            rand_idx = torch.randperm(data.x.shape[0])
            data.x = data.x[rand_idx]
        return data


class JointNoise:
    def __init__(self, std=0.5):
        self.std = std

    def __call__(self, data):
        noise = torch.zeros((data.x.shape[1], 3), dtype=torch.float32)
        noise[:, :2].uniform_(-self.std, self.std)

        data.x += noise.repeat((data.x.shape[0], 1, 1))

        return data


class PointNoise:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, data):
        noise = torch.zeros_like(data.x).uniform_(-self.std, self.std)
        data.x += noise

        data.x[:, :, 2] = torch.clamp(data.x[:, :, 2], min=0, max=1)

        return data


class RandomMove:
    def __init__(self, noise=(3, 1)):
        self.noise = noise

    def __call__(self, data):
        noise = torch.zeros(3, dtype=torch.float32)
        noise[0].uniform_(-self.noise[0], self.noise[0])
        noise[1].uniform_(-self.noise[1], self.noise[1])

        data.x += noise.repeat((data.x.shape[0], data.x.shape[1], 1))

        return data
