import torch
from ssd.data.build import make_data_loader
from ssd.utils.checkpoint import CheckPointer  # Poprawna nazwa klasy


@torch.no_grad()
def do_evaluation(cfg, model, distributed=False, iteration=None):
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    output_folder = cfg.OUTPUT_DIR

    data_loader_val = make_data_loader(
        cfg,
        is_train=False,
        distributed=distributed,
    )

    eval_results = {}
    dataset_name = cfg.DATASETS.TEST[0]  # "val"
    print(f"Evaluating dataset: {dataset_name}")
    eval_result = inference(
        model,
        data_loader_val,
        dataset_name,
        device,
        output_folder,
        iteration=iteration,
    )
    eval_results[dataset_name] = eval_result

    model.train()
    return eval_results


def inference(model, data_loader, dataset_name, device, output_folder, **kwargs):
    import os
    import time
    from tqdm import tqdm
    from ssd.data.datasets.evaluation import evaluate
    import numpy as np

    predictions = []
    timer = time.time()
    num_images = len(data_loader.dataset)
    print(f"Processing {num_images} images from {dataset_name}")

    # Przetwarzaj batche i mapuj predykcje na indeksy obrazów
    img_id_to_pred = {}
    for i, (images, _, img_ids) in enumerate(tqdm(data_loader)):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        for img_id, output in zip(img_ids, outputs):
            img_id_to_pred[img_id.item()] = {
                "boxes": output["boxes"].cpu().numpy(),
                "labels": output["labels"].cpu().numpy(),
                "scores": output["scores"].cpu().numpy(),
            }

    # Upewnij się, że predictions ma dokładnie num_images elementów
    predictions = [img_id_to_pred.get(i, {"boxes": np.array([]), "labels": np.array([]), "scores": np.array([])})
                   for i in range(num_images)]

    print(f"Inference took {time.time() - timer:.3f} seconds")
    result = evaluate(
        dataset=data_loader.dataset,
        predictions=predictions,
        output_dir=output_folder,
        **kwargs,
    )
    return result