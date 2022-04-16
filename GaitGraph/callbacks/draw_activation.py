import glob
from typing import List

import imageio
import matplotlib
from tqdm import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY


matplotlib.use('Qt5Agg')
# matplotlib.use('agg')


def draw_skeleton(result, points, label, graph, pause=.01, render_gif=True, min_conf=0.025, dpi=96):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx
    import matplotlib.colors as colors

    _, T, V = points.shape

    scalar_map = cmx.ScalarMappable(cmap=plt.get_cmap('plasma'), norm=colors.Normalize(vmin=0, vmax=1))

    result = np.maximum(result, 0)
    result = result / np.max(result)

    mean_pos = np.mean(np.mean(points[:2], -1), -1)

    plt.figure(figsize=(1000 / dpi, 1000 / dpi), dpi=dpi)
    plt.colorbar(scalar_map, shrink=0.25, aspect=5)
    plt.ion()
    for t in range(T):
        plt.cla()
        plt.xlim(-450, 450)
        plt.ylim(-450, 450)
        plt.axis('off')
        plt.title(f"subject: {int(label[0])}, angle: {int(label[1])}, sequence: {int(label[2])}, frame: {t}")

        x = points[0, t, :] - mean_pos[0]
        y = mean_pos[1] - points[1, t, :]
        conf = points[2, t, :]

        c = []
        activation = []
        for v in range(V):
            k = graph.connect_joint[v]

            r = np.max(result[:, t // 4, v], axis=0)
            activation.append(r)

            if conf[k] < min_conf or conf[v] < min_conf:
                c.append([0, 0, 0, 0])
                continue

            c.append(scalar_map.to_rgba(r))

            plt.plot([x[v], x[k]], [y[v], y[k]], '-', c=np.array([0.1, 0.1, 0.1]), alpha=0.5, linewidth=3, markersize=0)

        if graph.dataset == "oumvlp":
            plt.plot([x[8], x[11]], [y[8], y[11]], '-', c=np.array([0.1, 0.1, 0.1]), alpha=0.5, linewidth=3,
                     markersize=0)
        elif graph.dataset == "coco":
            plt.plot([x[11], x[12]], [y[11], y[12]], '-', c=np.array([0.1, 0.1, 0.1]), alpha=0.5, linewidth=3,
                     markersize=0)

        c = np.array(c, dtype=np.float)
        s = np.array(activation, dtype=np.float) * 128.
        plt.scatter(x, y, marker='o', c=c, s=s, zorder=2.5)

        # plt.pause(pause)
        plt.savefig(f"../data/output/{int(label[0])}-{int(label[1])}-{int(label[2])}-{t:03}.pdf")
        plt.savefig(f"../data/output/png/{int(label[0])}-{int(label[1])}-{int(label[2])}-{t:03}.png")
        # tikzplotlib.save(f"../output/{int(label[0])}-{int(label[1])}-{int(label[2])}-{t}.tex")

    plt.ioff()
    plt.close()

    if render_gif:
        images = []
        for filename in sorted(glob.glob(f"../data/output/png/{int(label[0])}-{int(label[1])}-{int(label[2])}-*.png")):
            image = imageio.imread(filename)
            images.append(image)
        imageio.mimwrite(f"../data/output/gif/{int(label[0])}-{int(label[1])}-{int(label[2])}.gif", images, duration=0.15)


@CALLBACK_REGISTRY
class DrawActivationCallback(Callback):
    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: List):
        weight = pl_module.backbone.fcn.weight.cpu()

        num_sequences = sum([o[0].shape[0] for o in outputs[0]])
        pbar = tqdm(total=num_sequences, leave=False)

        for output in outputs[0]:
            if len(output) == 6:
                feature, x, y, angle, seq_num, _ = output
            else:
                feature, x, y, angle, seq_num = output

            for i in range(feature.shape[0]):
                result = torch.einsum("kc,ctv->ktv", weight, feature[i])

                draw_skeleton(
                    result.numpy(),
                    x[i, :, :, 0, :3].permute(2, 0, 1).cpu().squeeze().numpy(),
                    (y[i], angle[i], seq_num[i]),
                    pl_module.graph
                )
                pbar.update()
        pbar.close()
