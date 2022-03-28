from torch_geometric.data import Data


class ToFlatTensor:
    def __call__(self, data):
        walking_status = data.walking_status if "walking_status" in data else 0
        return data.x, data.y, (data.angle, data.seq_num, walking_status)


class Normalize:
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def __call__(self, data):
        # data.x[:, :, 0] /= self.image_shape[0]
        # data.x[:, :, 1] /= self.image_shape[1]

        return data


class TwoNoiseTransform:
    def __init__(self, transform):
        self.transform = transform
        self.to_tensor = ToFlatTensor()

    def __call__(self, data):
        data1 = self.transform(Data(x=data.x))
        data = self.transform(data)

        out = self.to_tensor(data)

        return [out, (data1.x, ) + (out[1:])]
