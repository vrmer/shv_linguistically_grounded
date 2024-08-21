import os
import sys
import tomli
import torch
import signal
import pickle
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
from .utils import (top_n_index, get_all_means, compute_metrics,
                    get_free_gpu, signal_handler, MaskModel, attribute_factory)
from peft import PeftModelForSequenceClassification


# TODO: consider writing it in prune.py
def generate_mask(means: np.ndarray, n_heads: int, seed: int, random: bool):
    """
    Given an array of marginal contribution means,
    returns a binary mask with a number of top heads
    masked that correspond to n_heads.

    :param means: array of marginal contribution means
    :param n_heads: number of top heads to mask
    :param seed: for reproducibility
    :param random: if True, the mask is randomly generated
    :return: binary mask
    """
    mask = np.ones_like(means)
    if random is True:
        random_generator = np.random.default_rng(seed=seed)
        top_indices = random_generator.integers(0, len(means), size=n_heads)
    else:
        top_indices = top_n_index(means, top_n=n_heads)
    mask[top_indices] = 0
    return mask


def generate_unique_mask(input_dir: str,
                         mask_uid: str,
                         n_heads: int,
                         seed: int = 42,
                         random: bool = False):
    all_means = get_all_means(input_dir)
    mask_means = all_means[mask_uid]
    mask = generate_mask(mask_means, n_heads=n_heads, random=random, seed=seed)
    return mask


# import parameters
config_path = sys.argv[1]

with open(config_path, "rb") as infile:
    config = tomli.load(infile)

device = config["device"]

# model opts
model_name = config["model"]["name"]
layer_count = config["model"]["layer_count"]
attention_heads = config["model"]["attention_heads"]

# paths
input_dir = os.path.join(
    config["paths"]["input_dir"], model_name)
tracker_dir = os.path.join(
    config["paths"]["tracker_dir"], model_name)
attribution_path = os.path.join(
    config["paths"]["attribution_path"], model_name)
finetuned_model = config["paths"]["finetuned_model"]
head_mask = config["paths"]["head_mask"]

# attribution opts
if len(sys.argv) > 2:
    target_uid = sys.argv[2]
else:
    target_uid = config["attribution"]["target_uid"]
inference_type = config["attribution"]["inference_type"]
mask_id = config["attribution"]["mask_id"]
n_samples = config["attribution"]["n_samples"]

n_full_mask = layer_count * attention_heads
logging.set_verbosity_error()
os.makedirs(tracker_dir, exist_ok=True)

try:
    head_mask = int(head_mask)
except ValueError or TypeError:
    pass

if not mask_id:
    if isinstance(head_mask, int):
        raise ValueError("If head mask is an integer, mask id cannot be None")


if device == "cuda:0":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu())
    reserve = torch.tensor(1)
    reserve.to(device)


model = AutoModelForSequenceClassification.from_pretrained(model_name)
cfg = model.config

model = PeftModelForSequenceClassification.from_pretrained(
        model=model,
        model_id=finetuned_model
        )
tokenizer = AutoTokenizer.from_pretrained(model_name)


train_args = TrainingArguments(
        "/tmp/", per_device_eval_batch_size=100,
        report_to="none"
        )

if inference_type == "shapley":
    mask = torch.ones((1, n_full_mask)).to(device)
    fake_model = MaskModel(model, mask, target_uid, tracker_dir, layer_count, attention_heads)
    # Assign Handler Function
    signal.signal(signal.SIGINT, partial(signal_handler, fake_model=fake_model))
    target_model = fake_model
else:
    target_model = model
    if mask_id:
        if isinstance(head_mask, str):
            masks = np.load(head_mask, allow_pickle=True)
            mask = masks["arr_0"].item()[mask_id]
        else:
            mask = generate_unique_mask(attribution_path,
                                        mask_uid=mask_id, n_heads=head_mask)
        mask = torch.from_numpy(mask)
        mask = mask.reshape(layer_count, attention_heads)
        mask = mask.int()

with open(os.path.join(input_dir, "dataset_shuffled.pickle"), "rb") as infile:
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

    attribute = attribute_factory(fake_model)

    if inference_type == "shapley":
        sv = ShapleyValueSampling(attribute)
        attribution = sv.attribute(
                torch.ones((1, n_full_mask)).to(device), n_samples=n_samples, show_progress=True
                )
        fake_model.finish()

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

        if mask_id is None:
            exp_file = f"{target_uid}_baseline.txt"
        elif mask_id != target_uid:
            exp_file = f"{mask_id}-{target_uid}.txt"
        else:
            exp_file = f"{mask_id}.txt"
        with open(os.path.join(tracker_dir, exp_file), "w") as outfile:
            outfile.write(str(acc))
