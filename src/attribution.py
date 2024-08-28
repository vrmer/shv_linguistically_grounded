import os
import tomli
import torch
import signal
import pickle
import argparse
import evaluate
import numpy as np
from functools import partial
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer, Trainer
)
from transformers.utils import logging
from captum.attr import ShapleyValueSampling
from .utils import (generate_unique_mask, compute_metrics,
                    get_free_gpu, signal_handler, MaskModel, attribute_factory)
from peft import PeftModelForSequenceClassification

logging.set_verbosity_error()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config",
                        type=str, required=True,
                        help="Path to config file")

    parser.add_argument("-u", "--target_uid",
                        type=str, required=False, default=None,
                        help="Target uid, if not provided, it is taken from the config file")

    parser.add_argument("-m", "--mask_id",
                        type=str, required=False, default=None,
                        help="Mask id, if not provided, it is taken from the config file")

    parser.add_argument("-n", "--n_heads",
                        type=int, required=False, default=0,
                        help="Number of heads to mask, if not provided, it is taken from the config file")

    args = parser.parse_args()

    # import parameters
    with open(args.config, "rb") as infile:
        config = tomli.load(infile)

    device = config["device"]
    seed = config["seed"]

    # model opts
    model_name = config["model_name"]

    # paths
    dataset_dir = os.path.join(
        config["paths"]["dataset_dir"], model_name
    )
    tracker_dir = os.path.join(
        config["paths"]["tracker_dir"], model_name
    )
    prune_path = os.path.join(
        config["paths"]["prune_path"], model_name
    )
    finetuned_model = os.path.join(
        config["paths"]["finetuned_model"], model_name
    )

    # attribution opts
    if args.target_uid is None:
        target_uid = config["attribution"]["target_uid"]
    else:
        target_uid = args.target_uid

    inference_type = config["attribution"]["inference_type"]
    n_samples = config["attribution"]["n_samples"]

    # pruning opts
    if args.mask_id is None:
        mask_id = config["pruning"]["mask_id"]
    else:
        mask_id = args.mask_id

    if args.n_heads != 0:
        n_heads = args.n_heads
    else:
        n_heads = config["pruning"]["n_heads"]

    head_mask = os.path.join(
        config["pruning"]["head_mask"], model_name
    )
    filter_type = config["pruning"]["filter_type"]

    if not mask_id:
        if n_heads > 0:
            raise ValueError("If head mask is not 0, mask id cannot be None")


    if device == "cuda:0":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu())
        reserve = torch.tensor(1)
        reserve.to(device)


    # Load model and config
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    cfg = model.config
    layer_count = cfg.num_hidden_layers
    attention_heads = cfg.num_attention_heads
    n_full_mask = layer_count * attention_heads

    model = PeftModelForSequenceClassification.from_pretrained(
            model=model,
            model_id=finetuned_model
            )
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    train_args = TrainingArguments(
            "/tmp/", per_device_eval_batch_size=100,
            report_to="none",
            seed=seed
            )

    if inference_type == "shapley":
        mask = torch.ones((1, n_full_mask)).to(device)
        fake_model = MaskModel(model, mask, target_uid, tracker_dir, layer_count, attention_heads)
        # Assign Handler Function
        signal.signal(signal.SIGINT, partial(signal_handler, fake_model=fake_model))
        target_model = fake_model

    else:

        os.makedirs(tracker_dir, exist_ok=True)
        os.makedirs(prune_path, exist_ok=True)

        target_model = model
        if mask_id:
            if n_heads == 0:
                head_mask = os.path.join(head_mask, f"{filter_type}.npz")
                masks = np.load(head_mask, allow_pickle=True)
                mask = masks["arr_0"].item()[mask_id]
            else:
                mask = generate_unique_mask(tracker_dir,
                                            mask_uid=mask_id, n_heads=n_heads, seed=seed)
                os.makedirs(prune_path, exist_ok=True)
            if mask.dtype != torch.float64:
                mask = torch.from_numpy(mask)
            mask = mask.reshape(layer_count, attention_heads)
            mask = mask.int()

    with open(os.path.join(dataset_dir, "dataset_shuffled.pickle"), "rb") as infile:
        eval_dataset = pickle.load(infile)["test"]

    # select target uid
    eval_dataset = eval_dataset.filter(lambda example: example["UID"] == target_uid)
    label_list = eval_dataset["label"]

    with torch.no_grad():

        metric = evaluate.load("accuracy")

        compute_metrics = partial(compute_metrics, metric=metric, output_dir=None)

        model.eval()

        trainer = Trainer(
                model=target_model,
                args=train_args,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                )

        attribute_factory = partial(attribute_factory, n_full_mask=n_full_mask, trainer=trainer)

        attribute = attribute_factory(target_model)

        if inference_type == "shapley":
            sv = ShapleyValueSampling(attribute)
            attribution = sv.attribute(
                    torch.ones((1, n_full_mask)).to(device), n_samples=n_samples, show_progress=True
                    )
            target_model.finish()

        elif inference_type == "accuracy":

            if mask_id:
                tokens = tokenizer(eval_dataset["sentence_A"],
                                   eval_dataset["sentence_B"],
                                   padding=True, return_tensors="pt").to(device)
                output = model(**tokens, head_mask=mask.to(device))
                preds = torch.nn.functional.softmax(output.logits, dim=1).argmax(-1)
                acc = metric.compute(predictions=preds, references=eval_dataset["label"])

            else:
                acc = trainer.evaluate()["eval_accuracy"]

            print(f"Eval accuracy: {acc}")
            print()

            if not mask_id:
                exp_file = f"{target_uid}.txt"
                output_dir = os.path.join(prune_path, "baseline")
            else:
                if mask_id != target_uid:
                    exp_file = f"{mask_id}-on-{target_uid}.txt"
                else:
                    exp_file = f"{mask_id}.txt"

                if n_heads > 0:
                    output_dir = os.path.join(prune_path, f"{n_heads}_heads")
                else:
                    output_dir = os.path.join(prune_path, f"{filter_type}_filter")

            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, exp_file), "w") as outfile:
                outfile.write(str(acc))
