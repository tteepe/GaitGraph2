import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_metric_learning import losses, distances
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torchvision.transforms import Compose

from GaitGraph import cli_logo
from GaitGraph.datasets.graph import Graph
from GaitGraph.datasets.oumvlp_pose import OUMVLPPose
from GaitGraph.models import ResGCN, StGCN
from GaitGraph.transforms import ToFlatTensor
from GaitGraph.transforms.augmentation import RandomSelectSequence, PadSequence, SelectSequenceCenter, NormalizeEmpty, \
    RandomFlipLeftRight, PointNoise, RandomFlipSequence, JointNoise, RandomMove, ShuffleSequence
from GaitGraph.transforms.multi_input import MultiInput


class GaitGraphOUMVLP(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float = 0.01,
            lr_div_factor: float = 25.,
            loss_temperature: float = 0.07,
            embedding_layer_size: int = 64,
            multi_input: bool = True,
            backend_name="resgcn-n51-r4",
            tta: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        self.graph = Graph("oumvlp")
        model_args = {
            "A": torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False),
            "num_class": embedding_layer_size,
            "num_input": 3 if multi_input else 1,
            "num_channel": 5 if multi_input else 3,
            "parts": self.graph.parts,
        }
        if backend_name == "st-gcn":
            self.backbone = StGCN(3, self.graph, embedding_layer_size=embedding_layer_size)
        else:
            self.backbone = ResGCN(backend_name, **model_args)

        self.distance = distances.LpDistance()
        self.train_loss = losses.SupConLoss(loss_temperature, distance=self.distance)
        self.val_loss = losses.ContrastiveLoss(distance=self.distance)

    def forward(self, x):
        return self.backbone(x)[0]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        loss = self.train_loss(y_hat, y.squeeze())
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        loss = self.val_loss(y_hat, y.squeeze())
        self.log("val_loss", loss, on_step=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        x, y, (angle, seq_num, _) = batch
        feature = self.backbone(x)[1]

        return feature, x, y, angle, seq_num

    def test_step(self, batch, batch_idx):
        x, y, (angle, seq_num, _) = batch
        bsz = x.shape[0]

        if self.hparams.tta:
            multi_input = MultiInput(self.graph.connect_joint, self.graph.center, self.hparams.multi_input)
            x_flipped = torch.stack([
                multi_input(Data(x=d[:, :, 0, :3].flip(0), device=x.device)).x for d in x
            ])
            x_lr_flipped = torch.stack([
                multi_input(Data(x=d[:, self.graph.flip_idx, 0, :3], device=x.device)).x for d in x
            ])

            x = torch.cat([x, x_flipped, x_lr_flipped], dim=0)

        y_hat = self(x)

        if self.hparams.tta:
            f1, f2, f3 = torch.split(y_hat, [bsz, bsz, bsz], dim=0)
            y_hat = torch.cat((f1, f2, f3), dim=1)

        return y_hat, y, angle, seq_num

    def test_epoch_end(self, outputs, print_output=True):
        embeddings = dict()
        for batch in outputs:
            y_hat, subject_id, angle, seq_num = batch
            embeddings.update({
                (subject_id[i].item(), angle[i].item(), seq_num[i].item()): y_hat[i]
                for i in range(y_hat.shape[0])
            })

        angles = list(range(0, 91, 15)) + list(range(180, 271, 15))
        num_angles = len(angles)
        gallery = {k: v for (k, v) in embeddings.items() if k[2] == 0}

        gallery_per_angle = {}
        for angle in angles:
            gallery_per_angle[angle] = {k: v for (k, v) in gallery.items() if k[1] == angle}

        probe = {k: v for (k, v) in embeddings.items() if k[2] == 1}

        accuracy = torch.zeros((num_angles + 1, num_angles + 1))
        correct = torch.zeros_like(accuracy)
        total = torch.zeros_like(accuracy)

        for gallery_angle in angles:
            gallery_embeddings = torch.stack(list(gallery_per_angle[gallery_angle].values()), 0)
            gallery_targets = list(gallery_per_angle[gallery_angle].keys())
            gallery_pos = angles.index(gallery_angle)

            probe_embeddings = torch.stack(list(probe.values()))
            q_g_dist = self.distance(probe_embeddings, gallery_embeddings)

            for idx, target in enumerate(probe.keys()):
                subject_id, probe_angle, _ = target
                probe_pos = angles.index(probe_angle)

                min_pos = torch.argmin(q_g_dist[idx])
                min_target = gallery_targets[int(min_pos)]

                if min_target[0] == subject_id:
                    correct[gallery_pos, probe_pos] += 1
                total[gallery_pos, probe_pos] += 1

        accuracy[:-1, :-1] = correct[:-1, :-1] / total[:-1, :-1]

        accuracy[:-1, -1] = torch.mean(accuracy[:-1, :-1], dim=1)
        accuracy[-1, :-1] = torch.mean(accuracy[:-1, :-1], dim=0)

        accuracy_avg = torch.mean(accuracy[:-1, :-1])
        accuracy[-1, -1] = accuracy_avg
        self.log("test/accuracy", accuracy_avg)

        for angle, avg in zip(angles, accuracy[:-1, -1].tolist()):
            self.log(f"test/probe_{angle}", avg)
        for angle, avg in zip(angles, accuracy[-1, :-1].tolist()):
            self.log(f"test/gallery_{angle}", avg)

        df = pd.DataFrame(
            accuracy.numpy(),
            angles + ["mean"],
            angles + ["mean"],
        )
        df = (df * 100).round(1)

        if print_output:
            print(f"accuracy: {accuracy_avg * 100:.1f} %")
            print(df.to_markdown())
            print(df.to_latex())

        return df

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader())
        )
        lr_dict = {
            "scheduler": lr_schedule,
            "interval": "step"
        }
        return [optimizer], [lr_dict]


class OUMVLPPoseModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            dataset_path: str,
            keypoints: str = "openpose",
            batch_size: int = 512,
            num_workers: int = 4,
            sequence_length: int = 30,
            multi_input: bool = True,
            flip_sequence_p: float = 0.5,
            flip_lr_p: float = 0.5,
            joint_noise: float = 0.1,
            point_noise: float = 0.05,
            random_move: (float, float) = (3, 1),
            train_shuffle_sequence: bool = False,
            test_shuffle_sequence: bool = False,
            confidence_noise: float = 0.,
    ):
        super().__init__()
        self.graph = Graph("oumvlp")

        transform_train = Compose([
            PadSequence(sequence_length),
            RandomFlipSequence(flip_sequence_p),
            RandomSelectSequence(sequence_length),
            ShuffleSequence(train_shuffle_sequence),
            RandomFlipLeftRight(flip_lr_p, flip_idx=self.graph.flip_idx),
            JointNoise(joint_noise),
            PointNoise(point_noise),
            RandomMove(random_move),
            MultiInput(self.graph.connect_joint, self.graph.center, enabled=multi_input),
            ToFlatTensor()
        ])
        transform_val = Compose([
            NormalizeEmpty(),
            PadSequence(sequence_length),
            SelectSequenceCenter(sequence_length),
            ShuffleSequence(test_shuffle_sequence),
            MultiInput(self.graph.connect_joint, self.graph.center, enabled=multi_input),
            ToFlatTensor()
        ])

        self.dataset_train = OUMVLPPose(data_path, dataset_path, keypoints, "train", transform=transform_train)
        self.dataset_val = OUMVLPPose(data_path, dataset_path, keypoints, "test", transform=transform_val)
        self.dataset_test = OUMVLPPose(data_path, dataset_path, keypoints, "test", transform=transform_val)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)


def cli_main():
    LightningCLI(
        GaitGraphOUMVLP,
        OUMVLPPoseModule,
        seed_everything_default=5318008,
        save_config_overwrite=True,
        run=True
    )


if __name__ == "__main__":
    cli_logo()
    cli_main()
