import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_metric_learning import losses, distances
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torchvision.transforms import Compose

from GaitGraph import cli_logo
from GaitGraph.datasets.casia_b_pose import CASIABPose
from GaitGraph.datasets.graph import Graph
from GaitGraph.models import ResGCN
from GaitGraph.transforms import ToFlatTensor
from GaitGraph.transforms.augmentation import RandomSelectSequence, PadSequence, SelectSequenceCenter, \
    PointNoise, RandomFlipLeftRight, RandomMove, RandomFlipSequence, JointNoise, RandomCropSequence, \
    ShuffleSequence
from GaitGraph.transforms.multi_input import MultiInput


class GaitGraphCASIAB(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float = 0.005,
            weight_decay: float = 1e-5,
            lr_div_factor: float = 25.,
            loss_temperature: float = 0.07,
            embedding_layer_size: int = 128,
            multi_input: bool = True,
            multi_branch: bool = False,
            backend_name="resgcn-n39-r8",
            load_from_checkpoint: str = None,
            tta: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.graph = Graph("coco")
        model_args = {
            "A": torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False),
            "num_class": embedding_layer_size,
            "num_input": 3 if multi_branch else 1,
            "num_channel": 5 if multi_input else 3,
            "parts": self.graph.parts,
        }
        if multi_input and not multi_branch:
            model_args["num_channel"] = 15

        self.backbone = ResGCN(backend_name, **model_args)

        if load_from_checkpoint:
            checkpoint = pl_load(load_from_checkpoint)
            if "model" in checkpoint:
                self.load_state_dict({f"backbone.{k}": v for k, v in checkpoint["model"].items()})
            else:
                self.load_state_dict(checkpoint["state_dict"])

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

        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs, print_output=False)

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        x, y, (angle, seq_num, walking_status) = batch
        feature = self.backbone(x)[1]

        return feature, x, y, angle, seq_num, walking_status

    def test_step(self, batch, batch_idx):
        x, y, (angle, seq_num, walking_status) = batch
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

        return y_hat, y, angle, seq_num, walking_status

    def test_epoch_end(self, outputs, print_output=True):
        embeddings = dict()
        for batch in outputs:
            y_hat, subject_id, angle, seq_num, walking_status = batch
            embeddings.update({
                (subject_id[i].item(), walking_status[i].item(), seq_num[i].item(), angle[i].item()): y_hat[i]
                for i in range(y_hat.shape[0])
            })

        gallery = {k: v for (k, v) in embeddings.items() if k[1] == 0 and k[2] <= 4}
        gallery_per_angle = {}
        for angle in range(0, 181, 18):
            gallery_per_angle[angle] = {k: v for (k, v) in gallery.items() if k[3] == angle}

        probe_nm = {k: v for (k, v) in embeddings.items() if k[1] == 0 and k[2] >= 5}
        probe_bg = {k: v for (k, v) in embeddings.items() if k[1] == 1}
        probe_cl = {k: v for (k, v) in embeddings.items() if k[1] == 2}

        correct = torch.zeros((3, 11, 11))
        total = torch.zeros((3, 11, 11))
        for gallery_angle in range(0, 181, 18):
            gallery_embeddings = torch.stack(list(gallery_per_angle[gallery_angle].values()))
            gallery_targets = list(gallery_per_angle[gallery_angle].keys())
            gallery_pos = int(gallery_angle / 18)

            probe_num = 0
            for probe in [probe_nm, probe_bg, probe_cl]:
                probe_embeddings = torch.stack(list(probe.values()))
                q_g_dist = self.distance(probe_embeddings, gallery_embeddings)

                for idx, target in enumerate(probe.keys()):
                    subject_id, _, _, probe_angle = target
                    probe_pos = int(probe_angle / 18)

                    min_pos = torch.argmin(q_g_dist[idx])
                    min_target = gallery_targets[int(min_pos)]

                    if min_target[0] == subject_id:
                        correct[probe_num, gallery_pos, probe_pos] += 1
                    total[probe_num, gallery_pos, probe_pos] += 1

                probe_num += 1

        accuracy = correct / total

        accuracy_avg = torch.mean(accuracy)
        self.log("test/accuracy", accuracy_avg)

        # Exclude same view
        for i in range(3):
            accuracy[i] -= torch.diag(torch.diag(accuracy[i]))

        accuracy_flat = torch.sum(accuracy, 1) / 10

        header = ["NM#5-6", "BG#1-2", "CL#1-2"]

        sub_accuracies_avg = torch.mean(accuracy_flat, dim=1)
        sub_accuracies = dict(zip(header, list(sub_accuracies_avg)))
        for name, value in sub_accuracies.items():
            self.log(f"test/acc_{name}", value)

        df = pd.DataFrame(
            torch.cat([accuracy_flat, sub_accuracies_avg.unsqueeze(1)], dim=1).numpy(),
            header,
            list(range(0, 181, 18)) + ["mean"],
        )

        df = (df * 100).round(1)

        if print_output:
            print(f"accuracy: {accuracy_avg * 100:.1f} %")
            print(df.to_markdown())
            print(df.to_latex())

        return df

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
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


class CASIABPoseModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 256,
            num_workers: int = 4,
            sequence_length: int = 60,
            multi_input: bool = True,
            multi_branch: bool = False,
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
        self.graph = Graph("coco")

        transform_train = Compose([
            PadSequence(sequence_length),
            RandomFlipSequence(flip_sequence_p),
            RandomSelectSequence(sequence_length),
            ShuffleSequence(train_shuffle_sequence),
            RandomFlipLeftRight(flip_lr_p, flip_idx=self.graph.flip_idx),
            JointNoise(joint_noise),
            PointNoise(point_noise),
            RandomMove(random_move),
            MultiInput(self.graph.connect_joint, self.graph.center, enabled=multi_input, concat=not multi_branch),
            ToFlatTensor()
        ])
        transform_val = Compose([
            PadSequence(sequence_length),
            SelectSequenceCenter(sequence_length),
            ShuffleSequence(test_shuffle_sequence),
            MultiInput(self.graph.connect_joint, self.graph.center, enabled=multi_input),
            ToFlatTensor()
        ])

        self.dataset_train = CASIABPose(data_path, "train", transform=transform_train)
        self.dataset_val = CASIABPose(data_path, "test", transform=transform_val)
        self.dataset_test = CASIABPose(data_path, "test", transform=transform_val)

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
        GaitGraphCASIAB,
        CASIABPoseModule,
        seed_everything_default=5318008,
        save_config_overwrite=True
    )


if __name__ == "__main__":
    cli_logo()
    cli_main()
