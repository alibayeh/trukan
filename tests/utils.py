import random
import time

import numpy as np
import torch
from src.trukan.trukan_metadata import TruKanMetadata


def train_model(
    model, dataset, steps=100, loss_fn=None, lr=1.0, batch=-1, metrics=None
):
    if loss_fn is None:
        loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
    else:
        loss_fn = loss_fn_eval = loss_fn

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    results = {}
    results["train_loss"] = []
    results["test_loss"] = []
    results["reg"] = []
    if metrics is not None:
        for i in range(len(metrics)):
            results[metrics[i].__name__] = []

    if batch == -1 or batch > dataset["train_input"].shape[0]:
        batch_size = dataset["train_input"].shape[0]
        batch_size_test = dataset["test_input"].shape[0]
    else:
        batch_size = batch
        batch_size_test = batch

    metadata = TruKanMetadata(model)

    for _ in range(steps):
        train_id = np.random.choice(
            dataset["train_input"].shape[0], batch_size, replace=False
        )
        test_id = np.random.choice(
            dataset["test_input"].shape[0], batch_size_test, replace=False
        )

        pred = model.forward(dataset["train_input"][train_id])
        loss = loss_fn(pred, dataset["train_label"][train_id])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_loss = loss_fn_eval(
            model.forward(dataset["test_input"][test_id]),
            dataset["test_label"][test_id],
        )

        if metrics is not None:
            for i in range(len(metrics)):
                results[metrics[i].__name__].append(metrics[i]().item())

        results["train_loss"].append(torch.sqrt(loss).cpu().detach().numpy())
        results["test_loss"].append(torch.sqrt(test_loss).cpu().detach().numpy())

    return model, metadata, results


def train(
    model: torch.nn.Module,
    dataset: dict,
    n_iteration: int = 100,
) -> tuple[torch.nn.Module, TruKanMetadata, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    seed = 4253
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(
        f"Total_params: {sum(np.fromiter([param.numel() for param in model.parameters()], int))}"
    )
    print(
        f"Trainable_params: {sum(np.fromiter([p.numel() for p in model.parameters() if p.requires_grad], int))}"
    )

    start = time.perf_counter()
    model, model_metadata, result = train_model(
        model, dataset, steps=n_iteration, lr=0.1
    )
    print(f"Total time: {time.perf_counter() - start}")

    with torch.no_grad():
        pred = model.forward(dataset["test_input"])
        loss = torch.nn.MSELoss()(pred, dataset["test_label"])
        print(f"final loss: {loss}")

    return model, model_metadata, result
