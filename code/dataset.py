"""AdaCrowd annotation acquisition extracted from run/dataset.py:LabelMeA."""

import numpy as np
import torch
from torch.utils.data import Dataset


class AdaCrowdDataset(Dataset):
    def __init__(self, features, answers, num_classes):
        self.features = features
        self.num_classes = num_classes
        self.num_workers = answers.shape[1]
        self.belief = np.zeros(num_classes)
        self.candidates = [set() for _ in range(self.num_workers)]
        self.chosen = [set() for _ in range(self.num_workers)]
        self.annotations = []
        for index, row in enumerate(answers):
            for worker, label in enumerate(row):
                if label != -1:
                    self.annotations.append((worker, index, int(label)))
                    self.candidates[worker].add(index)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        worker, sample, label = self.annotations[index]
        return sample in self.chosen[worker], worker, self.features[sample], label

    @torch.no_grad()
    def next_step(self, model):
        """Acquire one annotation per available worker using the original top-10 rule."""
        device = next(model.parameters()).device
        for worker, candidates in enumerate(self.candidates):
            if not candidates:
                continue
            indices = list(candidates)
            workers = torch.full(
                (len(indices),), worker, dtype=torch.long, device=device
            )
            features = torch.from_numpy(self.features[indices]).to(device)
            alpha1, alpha2 = model(workers, features)
            uncertainty = (self.num_classes / alpha2.sum(dim=1)).cpu().numpy()
            evidence = alpha1.cpu().numpy()
            coverage = (self.belief[None, :] + evidence).var(axis=1)
            shortlist = np.argsort(-uncertainty, kind="mergesort")[:10]
            position = int(shortlist[np.argmin(coverage[shortlist])])
            selected = indices[position]
            self.belief += evidence[position]
            candidates.remove(selected)
            self.chosen[worker].add(selected)

    @property
    def selected_count(self):
        return sum(len(chosen) for chosen in self.chosen)
