"""
FashionSearch — the GPU half, run on Kaggle.

WHY THIS EXISTS
---------------
Embedding 10,000 products on Databricks serverless CPU takes about 45 minutes.
The same work on a Kaggle T4 takes two or three. Nothing about the pipeline
needs that time on CPU — it is simply where the compute happened to be.

So the work is split the way the DR-AutoML project splits it:

    Kaggle      GPU work — fine-tuning and embedding
    Databricks  governance — registry, evaluation, the gate, monitoring

This script does the Kaggle half and hands results back through a Unity Catalog
Volume. Databricks then reads a file, which needs no torch and no GPU.

HOW IT IS RUN
-------------
GitHub Actions pushes this file to Kaggle on every commit but does NOT run it.
That is deliberate: kernels pushed by API lose their secret attachments and
accelerator selection. You open the kernel, tick the Databricks secrets, choose
GPU T4, and Run All.

REQUIRED KAGGLE SECRETS
-----------------------
    DATABRICKS_HOST     https://dbc-xxxx-yyyy.cloud.databricks.com
    DATABRICKS_TOKEN    a personal access token
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# 0 · Configuration
# --------------------------------------------------------------------------

DEFAULTS = {
    "catalog": "fashion_dev",
    "encoder_repo": "yainage90/fashion-image-feature-extractor",
    "detector_repo": "yainage90/fashion-object-detection",
    "pairs_dataset": "yainage90/onthelook-fashion-anchor-positive-images",
    "sample_size": 10000,
    "eval_queries": 1000,
    "batch_size": 128,          # a T4 handles this comfortably; CPU could not
    "finetune": False,
    "finetune_epochs": 3,
    "finetune_lr": 1e-5,
    "temperature": 0.07,
}


def in_notebook() -> bool:
    """True inside a Jupyter/IPython kernel, which is what Kaggle runs."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def parse_args():
    """
    Build the config, from the command line when there is one.

    A Kaggle notebook IS a Jupyter kernel, and sys.argv there is the kernel's
    own launch flags:

        ['.../ipykernel_launcher.py', '-f', '.../kernel-abc123.json']

    Plain parse_args() rejects those, prints usage and calls sys.exit(2) — on
    the first line of main(), before any work, killing the kernel in seconds
    with nothing in the log but ERROR.

    So: no argv at all in a notebook, and parse_known_args elsewhere so an
    unexpected flag is ignored rather than fatal.
    """
    p = argparse.ArgumentParser()
    for key, value in DEFAULTS.items():
        if isinstance(value, bool):
            p.add_argument(f"--{key}", action="store_true", default=value)
        else:
            p.add_argument(f"--{key}", type=type(value), default=value)

    argv = [] if in_notebook() else sys.argv[1:]
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"ignoring unrecognised arguments: {unknown}")
    return args


def on_kaggle() -> bool:
    return os.path.exists("/kaggle/working")


def get_secret(name: str):
    """
    Find a credential, or return None. Never raises.

    Missing credentials are the NORMAL path here, not a failure. Kaggle
    deliberately holds none: dataset writes are refused without phone
    verification, and Kaggle Secrets are stripped on every CLI push. So the
    kernel does pure GPU compute and GitHub Actions does the Databricks work.

    Raising on a missing secret would kill a run that has already done all its
    GPU work and written every artifact — the one outcome worth avoiding.
    """
    import glob as _glob

    for path in _glob.glob("/kaggle/input/*/credentials.json"):
        try:
            with open(path) as fh:
                value = json.load(fh).get(name)
            if value:
                return value
        except Exception:
            continue

    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass

    return os.environ.get(name)


# --------------------------------------------------------------------------
# 1 · Data
# --------------------------------------------------------------------------

def load_pairs(args):
    from datasets import load_dataset
    from datasets import Image as HFImage

    n = args.sample_size
    print(f"downloading {n} pairs from {args.pairs_dataset}")
    ds = load_dataset(args.pairs_dataset, split=f"train[:{n}]")

    features = ds.features
    image_cols = [c for c, f in features.items() if isinstance(f, HFImage)]
    other = [c for c in features if c not in image_cols]
    print(f"  image columns: {image_cols}")

    if len(image_cols) < 2:
        raise SystemExit(f"expected two image columns, found {image_cols}")

    def pick(cols, hints, default):
        for h in hints:
            for c in cols:
                if h in c.lower():
                    return c
        return default

    anchor = pick(image_cols, ("anchor", "query", "crop", "post"), image_cols[0])
    positive = pick([c for c in image_cols if c != anchor],
                    ("positive", "target", "product", "thumb"), image_cols[1])
    category = pick(other, ("category", "class", "label"), None)

    names = getattr(features.get(category), "names", None) if category else None
    print(f"  anchor={anchor}  positive={positive}  category={category}")
    if names:
        print(f"  category names: {names}")

    return ds, anchor, positive, category, names


# --------------------------------------------------------------------------
# 2 · Models
# --------------------------------------------------------------------------

def build_encoder(repo, device):
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import PyTorchModelHubMixin
    from transformers import AutoImageProcessor, SwinConfig, SwinModel

    config = SwinConfig.from_pretrained(repo)
    processor = AutoImageProcessor.from_pretrained(repo)

    class ImageEncoder(nn.Module, PyTorchModelHubMixin):
        def __init__(self):
            super().__init__()
            self.swin = SwinModel(config=config)
            self.embedding_layer = nn.Linear(config.hidden_size, 128)

        def forward(self, pixel_values):
            feats = self.swin(pixel_values).pooler_output
            return F.normalize(self.embedding_layer(feats), p=2, dim=1)

    model = ImageEncoder().from_pretrained(repo).to(device).eval()
    return model, processor, config


def build_detector(repo, device):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    model = AutoModelForObjectDetection.from_pretrained(repo).to(device).eval()
    return model, AutoImageProcessor.from_pretrained(repo)


# --------------------------------------------------------------------------
# 3 · Optional fine-tuning
# --------------------------------------------------------------------------

def finetune(encoder, processor, config, ds, anchor_col, positive_col, args, device):
    """
    Contrastive fine-tuning with in-batch negatives.

    Same method the original author used: for each anchor its own positive is
    pulled close and every other positive in the batch is pushed away. With a
    batch of 128 each anchor learns from 127 negatives for free, which is the
    whole reason this beats triplet loss.

    Only worth doing on a GPU, which is exactly why it lives here and not in
    the Databricks pipeline.
    """
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset

    size = config.image_size
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    class Pairs(Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            item = ds[i]
            return tf(item[anchor_col].convert("RGB")), tf(item[positive_col].convert("RGB"))

    loader = DataLoader(Pairs(), batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    encoder.train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.finetune_lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    t = args.temperature

    for epoch in range(args.finetune_epochs):
        total, started = 0.0, time.time()
        for step, (a, p) in enumerate(loader):
            a, p = a.to(device, non_blocking=True), p.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                ea, ep = encoder(a), encoder(p)
                logits = ea @ ep.T / t
                labels = torch.arange(len(ea), device=device)
                # Symmetric: anchor->positive and positive->anchor.
                loss = 0.5 * (F.cross_entropy(logits, labels)
                              + F.cross_entropy(logits.T, labels))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()

            if step % 20 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(loader)} "
                      f"loss {loss.item():.4f}")

        print(f"  epoch {epoch+1} mean loss {total/max(len(loader),1):.4f} "
              f"({time.time()-started:.0f}s)")

    encoder.eval()
    return encoder


# --------------------------------------------------------------------------
# 4 · Embedding
# --------------------------------------------------------------------------

def embed_all(encoder, processor, config, ds, col, args, device, label):
    import numpy as np
    import torch
    import torchvision.transforms as T

    size = config.image_size
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    vectors, started = [], time.time()
    for start in range(0, len(ds), args.batch_size):
        batch = [tf(ds[i][col].convert("RGB"))
                 for i in range(start, min(start + args.batch_size, len(ds)))]
        x = torch.stack(batch).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            vectors.append(encoder(x).float().cpu().numpy())

        done = min(start + args.batch_size, len(ds))
        if done % (args.batch_size * 10) == 0 or done == len(ds):
            rate = done / (time.time() - started)
            print(f"  {label}: {done}/{len(ds)}  {rate:.0f}/s  "
                  f"~{(len(ds)-done)/rate/60:.1f} min left")

    return np.concatenate(vectors)


def detect_and_crop(detector, det_processor, ds, col, args, device):
    """Returns (crops, categories, area_fractions) for the query side."""
    import torch

    crops, cats, areas = [], [], []
    for i in range(len(ds)):
        img = ds[i][col].convert("RGB")
        W, H = img.size
        inputs = det_processor(images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = detector(**inputs)
        res = det_processor.post_process_object_detection(
            out, threshold=0.35, target_sizes=torch.tensor([[H, W]]).to(device))[0]

        if len(res["boxes"]):
            a = [((b[2]-b[0])*(b[3]-b[1])).item()/(W*H) for b in res["boxes"]]
            best = max(range(len(a)), key=lambda j: a[j] * res["scores"][j].item())
            x1, y1, x2, y2 = [float(v) for v in res["boxes"][best]]
            crops.append(img.crop((x1, y1, x2, y2)))
            cats.append(detector.config.id2label[int(res["labels"][best])])
            areas.append(a[best])
        else:
            # Same fallback the Databricks path uses, so the numbers stay
            # comparable between the two.
            crops.append(img)
            cats.append(None)
            areas.append(1.0)

        if (i + 1) % 200 == 0:
            print(f"  detected {i+1}/{len(ds)}")

    return crops, cats, areas


# --------------------------------------------------------------------------
# 5 · Register the encoder to Unity Catalog — FROM HERE, not from Databricks
# --------------------------------------------------------------------------

def connect_mlflow(catalog: str) -> bool:
    """
    Point MLflow at the Databricks workspace.

    THIS IS THE STEP THAT UNBLOCKS MODEL SERVING, and the reason is not obvious.

    Registering a model to Unity Catalog from inside a Databricks serverless
    notebook uploads artifacts using the cluster's assumed IAM role. On some
    workspace tiers that role is explicitly denied write access to UC managed
    storage, and you get:

        S3UploadFailedError: AccessDenied ... explicit deny in a resource-based policy

    Registering from OUTSIDE, over the REST API with a personal access token,
    goes through a different upload path and is not subject to that deny. So the
    model reaches Unity Catalog, and Model Serving — which can only serve UC
    models — becomes possible.

    Nothing about the model changes. Only who uploads it.
    """
    import mlflow

    host = get_secret("DATABRICKS_HOST")
    token = get_secret("DATABRICKS_TOKEN")
    if not host or not token:
        print("No Databricks credentials — skipping remote registration.")
        print("The artifacts are in the kernel output; GitHub registers them.")
        return False
    host = host.rstrip("/")

    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    experiment = f"/Shared/fashionsearch/{catalog}/kaggle"
    try:
        # MLflow creates experiments but not the folders above them.
        from databricks.sdk import WorkspaceClient
        WorkspaceClient(host=host, token=token).workspace.mkdirs(
            f"/Shared/fashionsearch/{catalog}")
    except Exception as exc:
        print(f"  could not create the experiment folder: {exc}")

    mlflow.set_experiment(experiment)
    print(f"MLflow connected: {host}")
    print(f"  experiment: {experiment}")
    return True


def register_encoder(encoder, processor, config, args, manifest):
    """
    Log the encoder as a pyfunc and register it in Unity Catalog.

    Saved as a state_dict plus config, never as a pickled module. Pickling the
    module records the class by reference as `__main__.ImageEncoder`, which does
    not exist anywhere else — the model then loads fine here and fails on
    Databricks with AttributeError.
    """
    import base64
    import numpy as np
    import pandas as pd
    import mlflow
    import torch
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    from PIL import Image

    model_name = f"{args.catalog}.ml.fashion_encoder"

    art_dir = os.path.join("/kaggle/working" if on_kaggle() else ".", "encoder_artifact")
    os.makedirs(art_dir, exist_ok=True)
    config.save_pretrained(art_dir)
    processor.save_pretrained(art_dir)
    torch.save(encoder.state_dict(), os.path.join(art_dir, "state_dict.pt"))

    class EncoderWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import os as _os
            import torch as _t
            import torch.nn as nn
            import torch.nn.functional as _F
            import torchvision.transforms as _T
            from PIL import Image as _I
            from transformers import AutoImageProcessor, SwinConfig, SwinModel

            self.torch, self.Image = _t, _I
            d = context.artifacts["model"]

            class ImageEncoder(nn.Module):
                def __init__(self, cfg_, dim):
                    super().__init__()
                    self.swin = SwinModel(config=cfg_)
                    self.embedding_layer = nn.Linear(cfg_.hidden_size, dim)

                def forward(self, pixel_values):
                    f = self.swin(pixel_values).pooler_output
                    return _F.normalize(self.embedding_layer(f), p=2, dim=1)

            swin_cfg = SwinConfig.from_pretrained(d)
            proc = AutoImageProcessor.from_pretrained(d)
            state = _t.load(_os.path.join(d, "state_dict.pt"), map_location="cpu")
            dim = state["embedding_layer.weight"].shape[0]

            m = ImageEncoder(swin_cfg, dim)
            m.load_state_dict(state)
            m.eval()
            self.model = m

            size = swin_cfg.image_size
            self.tf = _T.Compose([
                _T.Resize((size, size)), _T.ToTensor(),
                _T.Normalize(mean=proc.image_mean, std=proc.image_std)])

        def predict(self, context, model_input, params=None):
            tensors = []
            for b64 in model_input["image"]:
                img = self.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                tensors.append(self.tf(img))
            with self.torch.no_grad():
                emb = self.model(self.torch.stack(tensors)).cpu().numpy()
            return pd.DataFrame(emb)

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (128, 128, 128)).save(buf, format="JPEG")
    example = pd.DataFrame({"image": [base64.b64encode(buf.getvalue()).decode()]})
    out_example = pd.DataFrame(np.zeros((1, manifest["embedding_dim"]), dtype=np.float32))

    with mlflow.start_run(run_name=f"kaggle-{manifest['created_at']}") as run:
        mlflow.log_params({
            "source": args.encoder_repo,
            "finetuned": manifest["finetuned"],
            "finetune_epochs": manifest["finetune_epochs"],
            "n_products": manifest["n_products"],
            "trained_on": "kaggle",
            "device": manifest["device"],
        })
        mlflow.pyfunc.log_model(
            artifact_path="encoder",
            python_model=EncoderWrapper(),
            artifacts={"model": art_dir},
            signature=infer_signature(example, out_example),
            input_example=example,
            pip_requirements=["torch", "torchvision", "transformers", "pillow"],
        )
        model_uri = f"runs:/{run.info.run_id}/encoder"

    print(f"logged {model_uri}")

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient()
    client.set_model_version_tag(model_name, result.version, "trained_on", "kaggle")
    client.set_model_version_tag(model_name, result.version, "gate_status", "not_evaluated")
    print(f"registered {model_name} version {result.version} in Unity Catalog")

    # Bootstrap the champion only if there is none. After that the Databricks
    # gate decides promotion — a model trained here does not get to promote
    # itself past evaluation.
    try:
        current = client.get_model_version_by_alias(model_name, "production")
        print(f"  champion is already v{current.version} — left alone. "
              f"The Databricks gate decides whether v{result.version} replaces it.")
    except Exception:
        client.set_registered_model_alias(model_name, "production", result.version)
        print(f"  no champion existed — bootstrapped @production = v{result.version}")

    return model_name, result.version


# --------------------------------------------------------------------------
# 5 · Hand results back to Databricks
# --------------------------------------------------------------------------

def push_to_databricks(local_path, catalog, remote_name):
    """Upload one file to a UC Volume. Returns False if not configured."""
    """
    Upload one file into a Unity Catalog Volume.

    Files, not tables, and one file per run with a timestamped name. Two runs
    never collide, and Databricks only has to read a Parquet file — no torch,
    no GPU-shaped library ever gets imported on that side.
    """
    host = get_secret("DATABRICKS_HOST")
    token = get_secret("DATABRICKS_TOKEN")
    if not host or not token:
        return False

    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient(host=host.rstrip("/"), token=token)

    remote = f"/Volumes/{catalog}/silver/kaggle_inbox/{remote_name}"
    print(f"uploading {local_path} -> {remote}")

    with open(local_path, "rb") as fh:
        w.files.upload(remote, fh, overwrite=True)
    print("  uploaded")
    return True


def trigger_github(manifest: dict) -> None:
    """
    Tell GitHub the GPU work is done, so the Databricks half can start.

    This is what makes the two platforms one pipeline rather than two things a
    human joins up. Kaggle finishes, fires a repository_dispatch, and the
    pipeline workflow wakes up and runs evaluation, the gate and the serving
    deploy against embeddings that already exist.

    Needs GH_DISPATCH_TOKEN (a GitHub PAT with `repo` scope) and GH_REPO
    ("owner/name"). Both travel in the same private dataset as the Databricks
    credentials.

    A failure here is reported, not raised: the GPU work is already done and
    uploaded, and you can start the Databricks side by hand.
    """
    import json as _json
    import urllib.error
    import urllib.request

    token = get_secret("GH_DISPATCH_TOKEN")
    repo = get_secret("GH_REPO")
    if not token or not repo:
        print("\nNo GitHub credentials here — expected. The workflow that pushed")
        print("this kernel is waiting for it and will continue on its own.")
        return

    payload = _json.dumps({
        "event_type": "kaggle-complete",
        "client_payload": {
            "created_at": manifest.get("created_at"),
            "n_products": manifest.get("n_products"),
            "n_queries": manifest.get("n_queries"),
            "registered_model": manifest.get("registered_model"),
            "registered_version": manifest.get("registered_version"),
            "finetuned": manifest.get("finetuned"),
        },
    }).encode()

    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/dispatches",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "fashionsearch-kaggle",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 204:
                print(f"\ntriggered the Databricks pipeline in {repo}")
                print("  watch it at https://github.com/" + repo + "/actions")
            else:
                print(f"\nunexpected response from GitHub: {resp.status}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:300]
        print(f"\ncould not trigger GitHub: {exc.code} {body}")
        print("A 404 here usually means the token lacks 'repo' scope, or GH_REPO "
              "is wrong. The embeddings are already uploaded either way.")
    except Exception as exc:
        print(f"\ncould not trigger GitHub: {type(exc).__name__}: {exc}")


def main():
    args = parse_args()
    import numpy as np
    import pandas as pd
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no GPU. Set Accelerator = GPU T4 in the Kaggle sidebar, "
              "or this will be as slow as the Databricks CPU path it replaces.")

    ds, anchor_col, positive_col, category_col, names = load_pairs(args)
    encoder, processor, config = build_encoder(args.encoder_repo, device)
    detector, det_processor = build_detector(args.detector_repo, device)

    if args.finetune:
        print("\nfine-tuning the encoder")
        encoder = finetune(encoder, processor, config, ds,
                           anchor_col, positive_col, args, device)

    print("\nembedding the catalogue (positives)")
    product_vecs = embed_all(encoder, processor, config, ds, positive_col,
                             args, device, "catalogue")

    n_eval = min(args.eval_queries, len(ds))
    print(f"\ndetecting and embedding {n_eval} queries (anchors)")
    eval_ds = ds.select(range(n_eval))
    crops, cats, areas = detect_and_crop(detector, det_processor, eval_ds,
                                         anchor_col, args, device)

    import torchvision.transforms as T
    size = config.image_size
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                    T.Normalize(mean=processor.image_mean, std=processor.image_std)])
    query_vecs = []
    for start in range(0, len(crops), args.batch_size):
        x = torch.stack([tf(c) for c in crops[start:start + args.batch_size]]).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            query_vecs.append(encoder(x).float().cpu().numpy())
    query_vecs = np.concatenate(query_vecs)

    def category_of(i):
        raw = ds[i][category_col] if category_col else None
        if names is not None and isinstance(raw, int):
            return names[raw]
        return str(raw) if raw is not None else "unknown"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = "/kaggle/working" if on_kaggle() else "."

    products = pd.DataFrame({
        "product_id": [f"p{i:07d}" for i in range(len(ds))],
        "category": [category_of(i) for i in range(len(ds))],
        "embedding": [v.tolist() for v in product_vecs],
    })
    queries = pd.DataFrame({
        "post_id": [f"post_p{i:07d}" for i in range(n_eval)],
        "product_id": [f"p{i:07d}" for i in range(n_eval)],
        "category": [category_of(i) for i in range(n_eval)],
        "detected_category": cats,
        "area_frac": areas,
        "size_band": ["small" if a < 0.08 else "medium" if a < 0.30 else "large"
                      for a in areas],
        "occlusion": ["heavy" if a < 0.05 else "none" for a in areas],
        "embedding": [v.tolist() for v in query_vecs],
    })

    manifest = {
        "created_at": stamp,
        "encoder_repo": args.encoder_repo,
        "detector_repo": args.detector_repo,
        "finetuned": bool(args.finetune),
        "finetune_epochs": args.finetune_epochs if args.finetune else 0,
        "n_products": len(products),
        "n_queries": len(queries),
        "embedding_dim": int(product_vecs.shape[1]),
        "device": device,
        "fallback_rate": round(sum(1 for c in cats if c is None) / max(len(cats), 1), 4),
    }
    print("\n" + json.dumps(manifest, indent=2))

    # Save the encoder as weights plus config, for GitHub Actions to register.
    #
    # Kaggle deliberately holds NO Databricks credentials. Publishing them here
    # needs a private Kaggle Dataset, and dataset writes are refused (403) on
    # accounts without phone verification. Kaggle Secrets work but are stripped
    # on every CLI push, so they would need re-ticking by hand after each one —
    # which defeats the point of a pipeline.
    #
    # So Kaggle does pure compute and writes to its kernel output. GitHub
    # downloads that and talks to Databricks itself. A GitHub runner is just as
    # much "outside Databricks" as this kernel is, so Unity Catalog registration
    # still succeeds where it fails from inside a Databricks notebook.
    art_dir = os.path.join(out_dir, "encoder_artifact")
    os.makedirs(art_dir, exist_ok=True)
    config.save_pretrained(art_dir)
    processor.save_pretrained(art_dir)
    torch.save(encoder.state_dict(), os.path.join(art_dir, "state_dict.pt"))
    print(f"encoder artifacts -> {art_dir}: {sorted(os.listdir(art_dir))}")

    # The detector too. It is served alongside the encoder as one model, and
    # registered separately so it keeps its own version history.
    det_dir = os.path.join(out_dir, "detector_artifact")
    os.makedirs(det_dir, exist_ok=True)
    detector.save_pretrained(det_dir)
    det_processor.save_pretrained(det_dir)
    print(f"detector artifacts -> {det_dir}: {sorted(os.listdir(det_dir))}")

    paths = []
    for name, df in [("products", products), ("queries", queries)]:
        local = os.path.join(out_dir, f"{name}_{stamp}.parquet")
        df.to_parquet(local, index=False)
        paths.append((local, f"{name}_{stamp}.parquet"))

    local_manifest = os.path.join(out_dir, f"manifest_{stamp}.json")
    with open(local_manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    paths.append((local_manifest, f"manifest_{stamp}.json"))

    # If credentials happen to be available (Kaggle Secrets, or a run started by
    # hand), upload directly — it saves GitHub a download. Otherwise everything
    # is in the kernel output and GitHub collects it. Either path works.
    uploaded = False
    try:
        uploaded = all(push_to_databricks(local, args.catalog, remote)
                       for local, remote in paths)
    except Exception as exc:
        print(f"\nupload skipped: {type(exc).__name__}: {exc}")

    if uploaded:
        print("\nUploaded directly to Databricks.")
    else:
        print("\nNo Databricks credentials here — that is the expected path.")
        print("Everything is in the kernel output. GitHub Actions collects it,")
        print("uploads the embeddings and registers the encoder.")

    print("\nOutputs written:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
