import logging
import os
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from dlsia.core.networks.baggins import model_baggin
from dlsia.core.train_scripts import Trainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _build_criterion(model_parameters, device, ignore_index=-1):
    """
    This function builds the criterion used for model training based on weights provided from the parameters,
    and pass to the device.
    Input:
        model_parameters: class, pydantic validated model parameters
        device: torch.device object, cpu or gpu
        ignore_index: int, index to ignore in the loss calculation
    Output:
        criterion:
    """
    # Define criterion and optimizer
    criterion = getattr(nn, model_parameters.criterion)
    # Convert the string to a list of floats
    weights = [float(x) for x in model_parameters.weights.strip("[]").split(",")]
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = criterion(weight=weights, ignore_index=ignore_index)
    return criterion


def run_train(
    train_loader,
    val_loader,
    io_parameters,
    networks,
    model_parameters,
    device,
    model_dir,
    use_dvclive=True,
    use_savedvcexp=False,
):
    """
    Run the training process for the given networks.
    Input:
        train_loader: DataLoader object for training data
        val_loader: DataLoader object for validation data
        io_parameters: class, pydantic validated I/O parameters
        networks: list of nn.Module objects to be trained
        model_parameters: class, pydantic validated model parameters
        device: torch.device object, cpu or gpu
        model_dir: str, directory to save model parameters and metrics
        use_dvclive: bool, whether to use dvclive for logging
        use_savedvcexp: bool, whether to save dvclive experiments
    Output:
        net: nn.Module object, trained network
    """
    mlflow.set_experiment(io_parameters.uid_save)
    logging.info(f"Setting MLflow experiment name: {io_parameters.uid_save}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")

        # Log hyperparameters
        mlflow.log_params(
            {
                "network": model_parameters.network,
                "num_classes": model_parameters.num_classes,
                "num_epochs": model_parameters.num_epochs,
                "optimizer": model_parameters.optimizer,
                "criterion": model_parameters.criterion,
                "learning_rate": model_parameters.learning_rate,
                "batch_size_train": model_parameters.batch_size_train,
                "batch_size_val": model_parameters.batch_size_val,
                "val_pct": model_parameters.val_pct,
            }
        )

        torch.cuda.empty_cache()
        criterion = _build_criterion(model_parameters, device)

        network_name = model_parameters.network
        trained_nets = []

        for idx, net in enumerate(networks):
            logger.info(f"{network_name}: {idx+1}/{len(networks)}")
            optimizer = getattr(optim, model_parameters.optimizer)
            optimizer = optimizer(net.parameters(), lr=model_parameters.learning_rate)
            net = net.to(device)

            if use_dvclive:
                from dvclive import Live

                dvclive_savepath = f"{model_dir}/dvc_metrics"
                dvclive = Live(
                    dvclive_savepath, report="html", save_dvc_exp=use_savedvcexp
                )
            else:
                dvclive = None

            trainer = Trainer(
                net,
                train_loader,
                val_loader,
                model_parameters.num_epochs,
                criterion,
                optimizer,
                device,
                dvclive=dvclive,
                savepath=model_dir,
                saveevery=None,
                scheduler=None,
                show=0,
                use_amp=False,
                clip_value=None,
            )
            net, _ = trainer.train_segmentation()  # training happens here

            trained_nets.append(net)

            # Log model to MLflow
            mlflow.pytorch.log_model(
                net, f"model_{idx+1}", registered_model_name=io_parameters.uid_save
            )
            logging.info(f"Model logged to MLflow with name: {io_parameters.uid_save}")

            # Log DVC metrics to MLflow
            if use_dvclive and os.path.exists(dvclive_savepath):
                mlflow.log_artifacts(dvclive_savepath, artifact_path="dvc_metrics")
                logging.info(f"DVC metrics logged to MLflow from {dvclive_savepath}")

            # Clear out unnecessary variables from device memory
            torch.cuda.empty_cache()

        logger.info(f"{network_name} trained successfully.")

        # Create final model (ensemble or single)
        if model_parameters.network == "DLSIA SMSNetEnsemble":
            net = model_baggin(models=trained_nets, model_type="classification")
            logging.info("Ensemble model created from trained networks")
        else:
            net = trained_nets[0]

    return net


def _resolve_lightly_model_name(network_name: str) -> str:
    """
    Map the UI model_name from models.json to the lightly_train model string.
    e.g. "lightly_train DINOv3 ViTS16 EoMT"            -> "dinov3/vits16-eomt"
         "lightly_train DINOv3 ViTS16 EoMT COCO"        -> "dinov3/vits16-eomt-coco"
         "lightly_train DINOv3 ViTS16 EoMT Cityscapes"  -> "dinov3/vits16-eomt-cityscapes"
    """
    mapping = {
        "lightly_train DINOv3 ViTS16 EoMT":            "dinov3/vits16-eomt",
        "lightly_train DINOv3 ViTS16 EoMT COCO":       "dinov3/vits16-eomt-coco",
        "lightly_train DINOv3 ViTS16 EoMT Cityscapes": "dinov3/vits16-eomt-cityscapes",
    }
    if network_name not in mapping:
        raise ValueError(
            f"Unknown lightly_train network: '{network_name}'. "
            f"Known: {list(mapping.keys())}"
        )
    return mapping[network_name]


def _build_dvc_report_from_jsonl(jsonl_path: Path, dvc_dir: Path) -> bool:
    """
    Reads lightly_train's metrics.jsonl output and writes a dvclive HTML report.
    lightly_train writes one JSON object per line, e.g.:
        {"step": 10, "train/loss": 0.45, "val/loss": 0.51, "val/miou": 0.32}

    Returns True if report was written successfully, False otherwise.
    """
    import json

    if not jsonl_path.exists():
        logging.warning(f"metrics.jsonl not found at {jsonl_path}, skipping DVC report")
        return False

    try:
        from dvclive import Live

        dvc_dir.mkdir(parents=True, exist_ok=True)
        dvclive = Live(str(dvc_dir), report="html", save_dvc_exp=False)

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                step = record.pop("step", None)
                for metric_name, value in record.items():
                    dvclive.log_metric(metric_name, value)
                if step is not None:
                    dvclive.next_step()

        dvclive.make_report()
        logging.info(f"DVC HTML report written to {dvc_dir}")
        return True

    except Exception as e:
        logging.warning(f"Could not build DVC report: {e}")
        return False


def run_lightly_train(
    io_parameters,
    model_parameters,
    model_dir: str,
):
    """
    Run lightly_train finetuning for DINOv3 EoMT models.
    Input:
        io_parameters: IOParameters pydantic instance
        model_parameters: LightlyParameters pydantic instance
        model_dir: str, temporary directory for intermediate outputs
    """
    import random as _random
    import tempfile

    import lightly_train
    import numpy as np
    from PIL import Image
    from tiled.client import from_uri

    base_model = _resolve_lightly_model_name(model_parameters.network)
    logging.info(f"lightly_train base model: {base_model}")

    # Point lightly/torch caches into model_dir so nothing escapes the container
    os.environ["LIGHTLY_TRAIN_CACHE_DIR"]       = model_dir
    os.environ["LIGHTLY_TRAIN_MODEL_CACHE_DIR"] = model_dir
    os.environ["TORCH_HOME"]                    = model_dir

    # Connect to Tiled
    data_client = from_uri(
        io_parameters.data_tiled_uri, api_key=io_parameters.data_tiled_api_key
    )
    mask_container = from_uri(
        io_parameters.mask_tiled_uri, api_key=io_parameters.mask_tiled_api_key
    )

    if "mask_idx" not in mask_container.metadata:
        raise KeyError("Mask container is missing required 'mask_idx' metadata.")
    if "mask" not in mask_container.keys():
        raise KeyError("Mask container is missing required 'mask' array.")

    selected_indices = mask_container.metadata["mask_idx"]   # list of data slice ints
    mask_array = mask_container["mask"][:]                   # (N_annotated, H, W) int8

    # Build classes dict from annotation class metadata stored on the mask container.
    # DLSIA stores: {"0": {"label": "Class 1", "color": "#FFA200"}, ...}
    # lightly_train expects: {int: str}
    classes_meta = mask_container.metadata.get("classes", {})
    classes = {int(k): v["label"] for k, v in classes_meta.items()}
    classes[255] = "unknown"   # lightly_train ignore class

    logging.info(f"Classes: {classes}")
    logging.info(f"Annotated slices: {selected_indices}")

    with tempfile.TemporaryDirectory(prefix="lightly_data_") as tmp_dir:
        tmp_root = Path(tmp_dir)

        # Build and shuffle (data_idx, mask_idx) pairs
        all_pairs = list(zip(selected_indices, range(len(selected_indices))))
        _random.shuffle(all_pairs)

        val_n = max(1, int(0.2 * len(all_pairs))) if len(all_pairs) > 1 else 0
        train_pairs = all_pairs[val_n:]
        val_pairs   = all_pairs[:val_n]
        logging.info(f"Split: {len(train_pairs)} train, {len(val_pairs)} val pairs")

        def save_split(pairs, img_dir: Path, msk_dir: Path) -> None:
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)
            for data_idx, mask_pos in pairs:
                fname = f"{data_idx:05d}.png"
                # Image: percentile-normalise to uint8 RGB
                img = np.array(data_client[data_idx]).squeeze().astype(np.float32)
                lo, hi = np.percentile(img, 1), np.percentile(img, 99)
                img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
                img = (img * 255).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)   # (H, W) -> (H, W, 3)
                Image.fromarray(img).save(img_dir / fname)
                # Mask: remap DLSIA's -1 (unlabeled) -> 255 (lightly ignore)
                msk = mask_array[mask_pos].astype(np.int16)
                msk = np.where(msk == -1, 255, msk).astype(np.uint8)
                Image.fromarray(msk).save(msk_dir / fname)

        save_split(train_pairs, tmp_root / "train" / "images", tmp_root / "train" / "masks")
        save_split(val_pairs,   tmp_root / "val"   / "images", tmp_root / "val"   / "masks")

        # Resolve out_dir: keep base_model name filesystem-safe
        safe_base = base_model.replace("/", "_")
        out_dir = str(Path(model_parameters.out_dir) / safe_base)

        logging.info("Starting lightly_train.train_semantic_segmentation ...")
        lightly_train.train_semantic_segmentation(
            out=out_dir,
            model=base_model,
            overwrite=model_parameters.overwrite,
            resume_interrupted=model_parameters.resume_interrupted,
            steps=model_parameters.steps,
            devices=model_parameters.devices,
            num_nodes=model_parameters.num_nodes,
            batch_size=model_parameters.batch_size_train,
            data={
                "train": {
                    "images": str(tmp_root / "train" / "images"),
                    "masks":  str(tmp_root / "train" / "masks"),
                },
                "val": {
                    "images": str(tmp_root / "val" / "images"),
                    "masks":  str(tmp_root / "val" / "masks"),
                },
                "classes":        classes,
                "ignore_classes": [255],
            },
            logger_args={
                # lightly_train's native MLflow logger handles step metrics
                "mlflow": {
                    "experiment_name": io_parameters.uid_save,
                    "run_name":        f"lightly_{io_parameters.uid_save}",
                    "tracking_uri":    io_parameters.mlflow_uri,
                },
                # Disable tensorboard — we're inside a container with no display
                "tensorboard": None,
            },
            save_checkpoint_args={
                "save_every_num_steps": model_parameters.save_every_num_steps,
                "save_last":            model_parameters.save_last,
                "save_best":            model_parameters.save_best,
            },
        )
        logging.info("lightly_train finetuning complete.")

        best_ckpt = Path(out_dir) / "checkpoints" / "best.ckpt"
        if not best_ckpt.exists():
            # Fall back to last checkpoint if best was not saved
            best_ckpt = Path(out_dir) / "checkpoints" / "last.ckpt"
            logging.warning("best.ckpt not found, falling back to last.ckpt")

        # Build dvclive HTML report from lightly's metrics.jsonl so it appears
        # in the same place as DLSIA's DVC reports (UI training stats link)
        metrics_jsonl = Path(out_dir) / "metrics.jsonl"
        dvc_dir = Path(model_dir) / "dvc_metrics"
        dvc_report_ok = _build_dvc_report_from_jsonl(metrics_jsonl, dvc_dir)

        # Register model and upload artifacts to MLflow
        mlflow.set_tracking_uri(io_parameters.mlflow_uri)
        mlflow.set_experiment(io_parameters.uid_save)

        with mlflow.start_run(
            run_name=f"lightly_register_{io_parameters.uid_save}"
        ) as run:
            logging.info(f"MLflow Run ID: {run.info.run_id}")

            mlflow.log_params({
                "network":          model_parameters.network,
                "base_model":       base_model,
                "num_classes":      model_parameters.num_classes,
                "steps":            model_parameters.steps,
                "batch_size_train": model_parameters.batch_size_train,
                "batch_size_val":   model_parameters.batch_size_val,
            })

            # Log DVC metrics to MLflow
            if dvc_report_ok and dvc_dir.exists():
                mlflow.log_artifacts(str(dvc_dir), artifact_path="dvc_metrics")
                logging.info(f"DVC metrics logged to MLflow from {dvc_dir}")

            # Log model to MLflow
            from lightly_mlflow_wrapper import LightlySegWrapper

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=LightlySegWrapper(),
                artifacts={"checkpoint": str(best_ckpt)},
                registered_model_name=io_parameters.uid_save,
                pip_requirements=[
                    "lightly-train==0.13.2",
                    "torch==2.9.1",
                    "torchvision==0.24.1",
                    "pytorch-lightning==2.6.0",
                    "Pillow",
                    "mlflow==2.22.0",
                ],
                code_path=["lightly_mlflow_wrapper.py"],
            )
            logging.info(f"Model logged to MLflow with name: {io_parameters.uid_save}")