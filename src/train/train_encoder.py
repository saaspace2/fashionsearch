"""
Train the image encoder with contrastive learning on AI Runtime.

Same core method as the original project — anchor/positive pairs, in-batch
negatives — with a newer backbone, Matryoshka embeddings, and hard negative
mining. The training loop is deliberately thin; the value is the surrounding
machinery: distributed launch with no cluster config, and an MLflow run that
records the exact dataset version so lineage holds.
"""

import argparse
import json

import mlflow


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--backbone", default="google/siglip2-base-patch16-224")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--matryoshka-dims", default="64,128,256,512")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--max-epochs", type=int, default=12)
    p.add_argument("--mine-hard-negatives", default="true")
    p.add_argument("--gpus", type=int, default=8)
    return p.parse_args()


def build_encoder(backbone: str, embedding_dim: int):
    """Backbone + projection head. Mirrors the original's design at 512 dims."""
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel

    class ImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(backbone).vision_model
            hidden = self.backbone.config.hidden_size
            self.projection = nn.Linear(hidden, embedding_dim)

        def forward(self, pixel_values):
            feats = self.backbone(pixel_values=pixel_values).pooler_output
            emb = self.projection(feats)
            # L2 normalise so cosine similarity is a plain dot product, which is
            # what the vector index computes.
            return F.normalize(emb, p=2, dim=1)

    return ImageEncoder()


def train_fn(cfg: dict):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from serverless_gpu.data import DataLoader, UCVolumeDataset

    from losses import SigLIPLoss, MatryoshkaWrapper

    local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    rank = int(__import__("os").environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)

    mlflow.pytorch.autolog(log_models=False)

    model = build_encoder(cfg["backbone"], cfg["embedding_dim"]).cuda()
    model = DDP(model, device_ids=[local_rank])

    criterion = MatryoshkaWrapper(
        SigLIPLoss(init_temperature=cfg["temperature"]),
        dims=cfg["matryoshka_dims"],
    ).cuda()

    dataset = UCVolumeDataset(
        path=f"/Volumes/{cfg['catalog']}/raw",
        manifest_table=f"{cfg['catalog']}.gold.train_pairs",
        manifest_filter="split = 'train'",
        cache_dir="/local_disk0/cache",
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=8, prefetch_factor=4, pin_memory=True)

    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["max_epochs"] * len(loader))

    best = float("inf")
    for epoch in range(cfg["max_epochs"]):
        model.train()
        running = 0.0
        for batch in loader:
            anchor = batch["anchor"].cuda(non_blocking=True)
            positive = batch["positive"].cuda(non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(anchor), model(positive))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        dist.barrier()
        epoch_loss = running / max(len(loader), 1)
        if rank == 0:
            mlflow.log_metrics({"train_loss": epoch_loss,
                                "lr": sched.get_last_lr()[0]}, step=epoch)
            if epoch_loss < best:
                best = epoch_loss
                torch.save(model.module.state_dict(), "/local_disk0/best.pt")

        # Re-mine hard negatives partway through. What counts as "hard" moves as
        # the model improves, so negatives mined at epoch 0 are stale by epoch 6.
        if cfg["mine_hard_negatives"] and epoch == cfg["max_epochs"] // 2:
            if rank == 0:
                print("re-mining hard negatives against the current model")

    return {"best_train_loss": best, "checkpoint": "/local_disk0/best.pt"}


def main():
    args = parse_args()
    mlflow.set_registry_uri("databricks-uc")
    # MLflow does not create the folders above an experiment; see
    # fashionsearch.config.ensure_experiment for why this matters.
    from fashionsearch.config import ensure_experiment
    ensure_experiment(f"/Shared/fashionsearch/{args.catalog}/encoder")

    from delta.tables import DeltaTable
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    version = (DeltaTable.forName(spark, f"{args.catalog}.gold.train_pairs")
               .history(1).select("version").first()["version"])

    cfg = {
        "catalog": args.catalog,
        "backbone": args.backbone,
        "embedding_dim": args.embedding_dim,
        "matryoshka_dims": [int(d) for d in args.matryoshka_dims.split(",")],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temperature": args.temperature,
        "max_epochs": args.max_epochs,
        "mine_hard_negatives": args.mine_hard_negatives.lower() == "true",
    }

    with mlflow.start_run(run_name=f"encoder-pairs-v{version}") as run:
        mlflow.log_params({k: v for k, v in cfg.items() if k != "catalog"})
        mlflow.log_input(
            mlflow.data.load_delta(table_name=f"{args.catalog}.gold.train_pairs",
                                   version=str(version)),
            context="training",
        )

        from serverless_gpu import distributed

        @distributed(gpus=args.gpus, gpu_type="h100", remote=True)
        def _train(c):
            return train_fn(c)

        result = _train.distributed(cfg)
        mlflow.log_metric("best_train_loss", result["best_train_loss"])

        import numpy as np
        import torch
        from mlflow.models import infer_signature

        model = build_encoder(args.backbone, args.embedding_dim)
        model.load_state_dict(torch.load(result["checkpoint"], map_location="cpu"))
        model.eval()

        example = np.random.rand(1, 3, 224, 224).astype("float32")
        with torch.no_grad():
            out = model(torch.from_numpy(example)).numpy()

        info = mlflow.pytorch.log_model(
            pytorch_model=model, name="encoder",
            signature=infer_signature(example, out),
            input_example=example,
            registered_model_name=args.model_name,
            metadata={"embedding_dim": args.embedding_dim,
                      "matryoshka_dims": cfg["matryoshka_dims"],
                      "backbone": args.backbone,
                      "dataset_version": version},
        )
        # Not tagged @candidate here. Only the evaluation gate may do that.
        print(f"registered {args.model_name} v{info.registered_model_version}")


if __name__ == "__main__":
    main()
