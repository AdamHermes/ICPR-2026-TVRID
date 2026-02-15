import argparse
import os
from typing import Dict, List, Tuple

import lightning as L
import torch
from tqdm import tqdm

from utils.data import DataConfig, UnifiedReIDDataset, build_transforms
from utils.models import ReIDLightning


TRACKS = {
    "rgb": {"data_modality": "rgb", "query": "rgb", "gallery": "rgb"},
    "depth": {"data_modality": "depth", "query": "depth", "gallery": "depth"},
    "cross": {"data_modality": "rgbd", "query": "rgb", "gallery": "depth"},
}


def get_args():
    parser = argparse.ArgumentParser(description="Generate rankings on public test set")
    parser.add_argument("--checkpoint", required=True, help="Path to Lightning .ckpt checkpoint")
    parser.add_argument("--track", choices=TRACKS.keys(), default="rgb")
    parser.add_argument("--data-root", default="data/DB_extracted", help="Root dataset directory")
    parser.add_argument("--labels-csv", default="data/DB_extracted/public_test_labels.csv", help="Public test CSV")
    parser.add_argument(
        "--eval-subdir",
        default=None,
        help="Override eval subdirectory (e.g., 'train' for a personal valid split)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="rankings.csv")
    parser.add_argument("--sequence-length", type=int, default=None, help="Override sequence length (optional)")
    return parser.parse_args()
# In eval_generate.py, update the embed_dataset function

@torch.no_grad()
def embed_dataset(
    model: ReIDLightning,
    dataset: UnifiedReIDDataset,
    track: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model.eval()
    model.to(device)

    query_mod = TRACKS[track]["query"]
    gallery_mod = TRACKS[track]["gallery"]

    ids: List[str] = []
    paths: List[str] = []
    query_embeds = []
    gallery_embeds = []

    def _coerce_ids(batch_ids) -> List[str]:
        if isinstance(batch_ids, torch.Tensor):
            if batch_ids.ndim == 0:
                return [str(batch_ids.item())]
            return [str(x) for x in batch_ids.tolist()]
        return [str(x) for x in batch_ids]

    for batch in tqdm(loader, desc="Embedding test set"):
        ids.extend(_coerce_ids(batch["gallery_id"]))
        paths.extend(batch["path"])

        def _to_tensor_list(key):
            x = batch[key]
            if isinstance(x, list):
                return torch.stack(x).to(device)
            return x.to(device)

        if query_mod in batch:
            query_x = _to_tensor_list(query_mod)
            # Use BN features for inference (not raw features)
            _, query_bn, _ = model.forward(query_x, query_mod)
            query_embeds.append(query_bn)
            
        if gallery_mod in batch:
            gallery_x = _to_tensor_list(gallery_mod)
            _, gallery_bn, _ = model.forward(gallery_x, gallery_mod)
            gallery_embeds.append(gallery_bn)

    query_mat = torch.cat(query_embeds, dim=0)
    gallery_mat = torch.cat(gallery_embeds, dim=0)
    return ids, paths, query_mat, gallery_mat


def build_rankings(
    ids: List[str],
    paths: List[str],
    query_mat: torch.Tensor,
    gallery_mat: torch.Tensor,
) -> List[Dict[str, str]]:
    # Compute distances query x gallery
    dists = torch.cdist(query_mat, gallery_mat).cpu()
    results: List[Dict[str, str]] = []
    for i, qid in enumerate(ids):
        row = dists[i]
        sorted_idx = torch.argsort(row)
        rank = 1
        for g_idx in sorted_idx.tolist():
            gid = ids[g_idx]
            if gid == qid:
                continue  # skip self-match
            results.append(
                {
                    "query_gallery_id": qid,
                    "query_path": paths[i],
                    "gallery_id": gid,
                    "gallery_path": paths[g_idx],
                    "rank": rank,
                    "distance": float(row[g_idx].item()),
                }
            )
            rank += 1
    return results


def main():
    args = get_args()
    track_cfg = TRACKS[args.track]

    # Build data config
    data_cfg = DataConfig(
        root=args.data_root,
        eval_csv=args.labels_csv,
        modality=track_cfg["data_modality"],
        val_mode="eval",
    )
    if args.sequence_length is not None:
        data_cfg.sequence.length = args.sequence_length
    if args.eval_subdir is not None:
        data_cfg.eval_subdir = args.eval_subdir

    rgb_t, depth_t = build_transforms(data_cfg.transforms)
    dataset = UnifiedReIDDataset(
        csv_path=data_cfg.eval_csv,
        root=data_cfg.root,
        modality=data_cfg.modality,
        mode="eval",
        sequence=data_cfg.sequence,
        rgb_transform=rgb_t,
        depth_transform=depth_t,
        train_subdir=data_cfg.train_subdir,
        eval_subdir=data_cfg.eval_subdir,
        sampling_strategy=data_cfg.sequence.sampling,
        mask_rgb_with_depth=data_cfg.mask_rgb_with_depth,
        depth_mask_threshold=data_cfg.depth_mask_threshold,
    )

    model = ReIDLightning.load_from_checkpoint(args.checkpoint)

    ids, paths, query_mat, gallery_mat = embed_dataset(
        model=model,
        dataset=dataset,
        track=args.track,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    rankings = build_rankings(ids, paths, query_mat, gallery_mat)

    import csv

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_gallery_id", "query_path", "gallery_id", "gallery_path", "rank", "distance"],
        )
        writer.writeheader()
        writer.writerows(rankings)
    print(f"Wrote {len(rankings)} rows to {args.output}")


if __name__ == "__main__":
    main()
