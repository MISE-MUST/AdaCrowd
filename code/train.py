"""Run AdaCrowd on the included LabelMe sample."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import AdaCrowdDataset
from model import CrowdLayerA, DenseNet, edl_digamma_loss


def main():
    parser = argparse.ArgumentParser(description="AdaCrowd minimal example")
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).resolve().parent / "data" / "labelme_sample.npz",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    with np.load(args.data) as sample:
        features = sample["train_features"].astype(np.float32)
        answers = sample["answers"]
        num_classes = int(sample["num_classes"])
        test_features = torch.from_numpy(sample["test_features"].astype(np.float32))
        test_labels = torch.from_numpy(sample["test_labels"].astype(np.int64))

    trainset = AdaCrowdDataset(features, answers, num_classes)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(
        TensorDataset(test_features, test_labels), batch_size=args.batch_size
    )
    backbone = DenseNet(int(np.prod(features.shape[1:])), num_classes)
    model = CrowdLayerA(backbone, trainset.num_workers, num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(
        f"AdaCrowd | train={len(features)} test={len(test_labels)} "
        f"workers={trainset.num_workers} annotations={len(trainset)} device={device}",
        flush=True,
    )
    for step in range(args.steps):
        if not any(trainset.candidates):
            break
        trainset.next_step(model)
        total_loss = 0.0
        total_seen = 0
        for _ in range(args.epochs):
            model.train()
            for chosen, workers, inputs, labels in trainloader:
                if not chosen.any():
                    continue
                workers = workers[chosen].to(device)
                inputs = inputs[chosen].to(device)
                labels = labels[chosen].to(device)
                optimizer.zero_grad()
                _, output = model(workers, inputs)
                loss = edl_digamma_loss(output, labels, step)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                total_seen += len(labels)

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                predictions = backbone(inputs.to(device)).argmax(dim=1).cpu()
                correct += (predictions == labels).sum().item()
        print(
            f"step={step + 1:02d} selected={trainset.selected_count} "
            f"loss={total_loss / max(total_seen, 1):.4f} "
            f"accuracy={correct / len(test_labels):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
