import logging

import numpy as np
import torch

from mlex_dlsia.utils.dataloaders import construct_inference_dataloaders

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_inference(dataset, net, seg_client, model_parameters, device):
    torch.cuda.empty_cache()
    logging.info(f"Starting inference on {len(dataset)} frames using device: {device}")

    # Determine if ensemble and whether to apply softmax
    is_ensemble = model_parameters.network == "DLSIA SMSNetEnsemble"
    final_layer = None if is_ensemble else torch.nn.Softmax(dim=1)
    segment_fn = (
        _segment_single_frame_ensemble if is_ensemble else _segment_single_frame
    )

    net.eval().to(device)
    for idx in range(len(dataset)):
        inference_loader = construct_inference_dataloaders(
            dataset[idx], model_parameters
        )
        prediction = segment_fn(
            network=net,
            dataloader=inference_loader,
            final_layer=final_layer,
            device=device,
        )

        stitched_prediction, _ = dataset.qlty_object.stitch(prediction)
        result = torch.argmax(stitched_prediction, dim=1).numpy().astype(np.int8)
        seg_client.write_block(result, block=(idx, 0, 0))
        logging.info(f"Frame {idx+1} result saved to Tiled")

        if device != "cpu" and (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    pass


def _segment_single_frame_ensemble(network, dataloader, device, final_layer=None):
    """
    Segment a single frame using an ensemble network.

    The ensemble already returns softmax probabilities, so no additional
    activation is needed.

    Args:
        network: Ensemble network model
        dataloader: DataLoader providing patches
        final_layer: Not used (kept for API consistency)
        device: Computation device

    Returns:
        Concatenated predictions for all patches
    """
    results = []
    for batch in dataloader:
        with torch.no_grad():
            torch.cuda.empty_cache()
            patches = batch[0].float().to(device)
            mean, _ = network(patches, device=device, return_std=True)
            results.append(mean.cpu())
    results = torch.cat(results)
    return results


def _segment_single_frame(network, dataloader, final_layer, device):
    """
    Segment a single frame using a single network.

    Args:
        network: Single network model
        dataloader: DataLoader providing patches
        final_layer: Activation layer (e.g., Softmax)
        device: Computation device

    Returns:
        Concatenated predictions for all patches
    """
    results = []
    for batch in dataloader:
        with torch.no_grad():
            torch.cuda.empty_cache()
            patches = batch[0].type(torch.FloatTensor)
            tmp = final_layer(network(patches.to(device))).cpu()
            results.append(tmp)
    results = torch.cat(results)
    return results


def run_lightly_inference(dataset, model, seg_client, model_parameters):
    """
    Run inference using a lightly_train pyfunc model loaded from MLflow.

    Unlike DLSIA models, lightly_train handles resizing and patching internally
    via model.predict(), so we bypass TiledDataset.__getitem__ (which applies
    QLTY patching) and read raw frames directly from the underlying data client.

    model.predict() contract (from LightlySegWrapper):
        input : uint8 numpy array, shape (H, W, 3)
        output: int32 numpy array, shape (H, W)

    Input:
        dataset: TiledDataset instance (used for data_client and selected_indices)
        model: mlflow.pyfunc loaded LightlySegWrapper model
        seg_client: Tiled array client for writing results
        model_parameters: LightlyParameters pydantic instance
    """
    n_frames = len(dataset)
    logging.info(f"Starting lightly inference on {n_frames} frames")

    for idx in range(n_frames):
        # Resolve the actual data slice index.
        # selected_indices is set when inference runs on annotated slices only
        # (partial inference after training). For full-dataset inference it is None.
        if dataset.selected_indices is not None:
            data_idx = dataset.selected_indices[idx]
        else:
            data_idx = idx

        # Read raw frame and normalise to uint8
        raw = np.array(dataset.data_client[data_idx]).squeeze().astype(np.float32)
        lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
        img = np.clip((raw - lo) / (hi - lo + 1e-8), 0, 1)
        img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)   # (H, W) -> (H, W, 3)

        # lightly_train predict returns (H, W) int32
        result = model.predict(img).astype(np.int8)

        # write_block expects (1, H, W) because seg_client was allocated with
        # shape (N, H, W) and chunks of (1, H, W)
        seg_client.write_block(result[np.newaxis], block=(idx, 0, 0))
        logging.info(f"Frame {idx+1}/{n_frames} lightly result saved to Tiled")