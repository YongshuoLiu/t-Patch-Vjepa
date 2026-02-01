import torch
import lib.utils as utils


def compute_classfaction_losses(model, batch_dict):
    """
    Return a dict that keeps a differentiable loss tensor for backward(),
    and also provides a python float for logging.

    train loop should do:
        train_res = compute_classfaction_losses(...)
        train_res["loss"].backward()
        print(train_res["loss_item"])
    """
    out = model.compute_loss(batch_dict)

    # unwrap loss from possible return types
    if isinstance(out, dict):
        loss = out.get("loss", None)
        if loss is None:
            raise ValueError(f"compute_loss returned dict without 'loss' key: {list(out.keys())}")
        extra = {k: v for k, v in out.items() if k != "loss"}

    elif isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise ValueError("compute_loss returned an empty tuple/list.")
        loss = out[0]
        extra = {"extra_outputs": out[1:]} if len(out) > 1 else {}

    else:
        loss = out
        extra = {}

    if not torch.is_tensor(loss):
        raise TypeError(f"loss is not a torch.Tensor, got {type(loss)}")

    # IMPORTANT: keep tensor loss for backward
    results = {"loss": loss}

    # optional float for logging (detached)
    results["loss_item"] = float(loss.detach().item())

    # keep extra (optional)
    results.update(extra)
    return results



@torch.no_grad()
def evaluation(model, dataloader, n_batches):
    """
    Evaluate average loss over n_batches.

    Compatible with model.compute_loss(batch_dict) returning:
      1) torch.Tensor scalar loss
      2) tuple/list: (loss, *others)
      3) dict: {"loss": loss, ...}

    Returns:
      {"BCE": avg_loss, "N": total_samples}
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for _ in range(n_batches):
        batch_dict = utils.get_next_batch(dataloader)
        B = batch_dict["data_sequence"].size(0)

        out = model.compute_loss(batch_dict)

        # unwrap loss
        if isinstance(out, dict):
            loss = out.get("loss", None)
            if loss is None:
                raise ValueError(f"compute_loss returned dict without 'loss' key: {list(out.keys())}")
        elif isinstance(out, (tuple, list)):
            if len(out) == 0:
                raise ValueError("compute_loss returned an empty tuple/list.")
            loss = out[0]
        else:
            loss = out

        if not torch.is_tensor(loss):
            raise TypeError(f"loss is not a torch.Tensor, got {type(loss)}")

        # accumulate sample-weighted loss
        total_loss += float(loss.detach().item()) * B
        total_samples += B

    avg_loss = total_loss / max(total_samples, 1)

    return {
        "BCE": avg_loss,
        "N": total_samples
    }