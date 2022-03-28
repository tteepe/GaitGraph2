from typing import Optional, Callable
import os.path as osp
import glob
import json
from tqdm import tqdm
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data


class OUMVLPPose(InMemoryDataset):

    split_id_file = "ID_list.csv"

    def __init__(
        self,
        root,
        dataset_path,
        keypoints: str = "openpose",
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        self.dataset_path = dataset_path
        self.keypoints = keypoints
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return f"oumvlp_{self.keypoints}_{self.split}.pt"

    def process(self):
        # Load ids
        ids = pd.read_csv(osp.join(self.dataset_path, self.split_id_file), dtype="Int32")
        id_ix = 0 if self.split == "train" else 1
        id_list = set(ids.iloc[:, id_ix].to_list())

        path = osp.join(self.dataset_path, self.keypoints, "*", "*")
        samples = sorted(glob.glob(path))

        data_list = []
        for sequence in tqdm(samples, f"{self.keypoints} [{self.split}]"):
            p_id, info = sequence.split(osp.sep)[-2:]
            angle, seq_num = info.split("_")

            if int(p_id) not in id_list:
                continue

            keypoints = []
            frame_nums = []
            for file in sorted(glob.glob(osp.join(sequence, "*.json"))):
                with open(file) as f:
                    d = json.load(f)

                if len(d["people"]) == 0:
                    continue

                frame_num = osp.basename(file).split("_")
                frame_num = int(frame_num[0]) if "-" not in frame_num[0] else int(frame_num[-1].split("-")[-1][:-5])

                pose = torch.FloatTensor(d["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)

                frame_nums.append(frame_num)
                keypoints.append(pose)

            if not keypoints:
                # print(f"Invalid sequence: {sequence.split(osp.sep)[-2:]}")
                continue

            data = Data(
                x=torch.stack(keypoints),
                y=int(p_id),
                angle=int(angle),
                seq_num=int(seq_num),
                frame_num=torch.IntTensor(frame_nums),
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    for name, split in [("openpose", "train"), ("openpose", "test"), ("alphapose", "train"), ("alphapose", "test")]:
        dataset = OUMVLPPose("../../data", "../../../datasets/OUMVLP-Pose", name, split)

        idx, item = next(enumerate(dataset))
        print(idx, item)
