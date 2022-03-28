import torch


class MultiInput:
    def __init__(self, connect_joint, center=0, enabled=False, concat=False):
        self.connect_joint = connect_joint
        self.center = center
        self.enabled = enabled
        self.concat = concat

    def __call__(self, data):
        if not self.enabled:
            data.x = data.x.unsqueeze(2)
            return data

        # T, V, C -> T, V, I=3, C + 2
        T, V, C = data.x.shape
        x_new = torch.zeros((T, V, 3, C + 2), device=data.x.device)

        # Joints
        x = data.x
        x_new[:, :, 0, :C] = x
        for i in range(V):
            x_new[:, i, 0, C:] = x[:, i, :2] - x[:, self.center, :2]

        # Velocity
        for i in range(T - 2):
            x_new[i, :, 1, :2] = x[i + 1, :, :2] - x[i, :, :2]
            x_new[i, :, 1, 3:] = x[i + 2, :, :2] - x[i, :, :2]
        x_new[:, :, 1, 3] = x[:, :, 2]

        # Bones
        for i in range(V):
            x_new[:, i, 2, :2] = x[:, i, :2] - x[:, self.connect_joint[i], :2]
        bone_length = 0
        for i in range(C - 1):
            bone_length += torch.pow(x_new[:, :, 2, i], 2)
        bone_length = torch.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            x_new[:, :, 2, C+i] = torch.arccos(x_new[:, :, 2, i] / bone_length)
        x_new[:, :, 2, 3] = x[:, :, 2]

        if self.concat:
            x_new = torch.cat([x_new[:, :, i] for i in range(3)], dim=2)

        data.x = x_new
        return data
