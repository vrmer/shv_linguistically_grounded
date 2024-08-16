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


# def get_free_gpu():
#     cmd = "nvidia-smi -q -d pids | grep -A4 GPU | grep Processes > tmp"
#     p = subprocess.Popen(["/bin/bash", "-c", cmd])
#     p.wait()
#     memory_available = [x.split(":")[-1].strip() for x in open("tmp", "r").readlines()]
#     print(memory_available)
#     id = memory_available.index("None")
#     print("Allocating Model to " + str(id))
#     return id
#
#
# def signal_handler(signal, frame):
#     print("You pressed Ctrl+C!")
#     print(signal)  # Value is 2 for CTRL + C
#     print(frame)  # Where your execution of program is at the moment - the Line Number
#     fake_model.finish()
#     sys.exit(0)
#
#
# def bernstein(sample):
#     """
#     Calculates the confidence interval for the mean of a sample using Bernstein's inequality.
#     """
#     if len(sample) < 2:
#         return -1, 1
#     mean = np.mean(sample)
#     variance = np.std(sample)
#     delta = 0.1
#     R = 1
#     bern_bound = (variance * np.sqrt((2 * np.log(3 / delta))) / len(sample)) + (
#             (3 * R * np.log(3 / delta)) / len(sample)
#             )
#     return mean - bern_bound, mean + bern_bound
#
#
# class MaskModel(torch.nn.Module):
#     """
#     Designed for tracking and controlling the activity of different heads
#     in a neural network model during training. It logs information about the contributions
#     of each head and provides methods for setting masks, determining active heads, and
#     resetting the model.
#     """
#     def __init__(self, real_model, head_mask, exp):
#         super(MaskModel, self).__init__()
#         self.contribs = defaultdict(self.construct_array)  # storing contributions of various heads during training
#         self.counter = 0  # number of iterations/batches processed
#         self.prev = 1.0  # variable storing previous accuracy during training
#         self.exp = exp  # experiment identifier
#         self.real_model = real_model  # underlying transformer model
#         self.head_mask = head_mask  # mask controlling which heads are active
#         self.true_prev = True
#         self.prev_mask = torch.ones_like(head_mask).flatten()  # mask used for tracking changes in the head mask
#         self.u = torch.zeros_like(head_mask).flatten()  # tensor used for tracking activity of heads
#         self.tracker = open(output_dir + "/" + exp + "_tracker.txt", "a")  # file handle for tracking information
#         self.sample_limit = 5000  # limit on number of samples tracking
#         self.layers = layer_count
#         self.attention_heads = attention_heads
#
#     def construct_array(self):
#         return []
#
#     def track(self, head, acc):
#         """
#         Tracks contribution from different heads and updates counters,
#         plus writes this information out to a file.
#         """
#         if head is not None:
#             self.contribs[head].append(self.prev - acc)
#         else:
#             self.baseline = acc
#         self.prev = acc
#         if self.counter % 100 == 0:
#             self.tracker.write(str(self.u.sum()) + "-" + str(self.counter) + "\n")
#             self.tracker.flush()
#         self.counter += 1
#
#     def finish(self):
#         """
#         Writes contribution arrays to a file before closing it.
#         """
#         self.tracker.write("Contribution Arrays")
#         self.tracker.write(json.dumps(self.contribs))
#         self.tracker.close()
#
#     def set_mask(self, mask):
#         """
#         Sets head mask to a specified value
#         """
#         mask = mask.reshape(self.layers, self.attention_heads)
#         self.head_mask = mask
#
#     def get_head(self, mask):
#         """
#         Determines active head by comparing the current head mask with the previous one.
#         Returns index of the active head.
#         """
#         head = (mask.reshape(-1) != self.prev_mask.reshape(-1)).nonzero(as_tuple=True)[0]
#         head = head.detach().cpu().tolist()[0]
#         self.prev_mask = mask
#         return head
#
#     def active(self, head):
#         """
#         Determines if a particular head is active based on a stored value and a function.
#         Uses sample limit for decision-making
#         """
#         def active_memo(head):
#             contribs = np.array(self.contribs[head])
#             lower, upper = bernstein(contribs)
#             if lower > -0.01:
#                 return False
#             elif len(contribs) > self.sample_limit:
#                 return False
#             return True
#
#         stored = self.u[head]
#         if head == None:
#             return True
#         elif stored == 1:
#             return False
#         else:
#             is_active = active_memo(head)
#             if is_active:
#                 return True
#             else:
#                 self.u[head] = 1
#                 return False
#
#     def reset(self):
#         """
#         Resets model state, including masks and counters.
#         """
#         print("RESET")
#         self.true_prev = True
#         self.prev_mask = torch.ones_like(self.prev_mask).flatten()
#         self.head_mask = torch.ones_like(self.head_mask)
#         self.prev = self.baseline
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             # input_embeds=None,
#             labels=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None
#             ):
#         return self.real_model.forward(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 position_ids=position_ids,
#                 # input_embeds=input_embeds,
#                 labels=labels,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 head_mask=self.head_mask
#                 )
#
#
# def attribute_factory(model):
#     def attribute(mask):
#         mask = mask.flatten()
#         if mask.sum() == 1:  # if only 1 head is active, reset the model
#             model.reset()
#         mask = mask == 0  # invert mask order
#         if not mask.sum() == n_full_mask:  # if not every head is active, get the active head
#             head = model.get_head(mask)
#         else:
#             head = None  # if all attention heads active, set the active head to None
#         if not model.active(head) or mask.sum() <= (n_full_mask//2):  # if the model is not active or if the sum is less than or equal half
#             acc = model.prev  # set acc to previous accuracy
#             model.true_prev = False  # set true prev to False
#         else:
#             if not model.true_prev:  # if model head is active or sum is more than half, check if model.true_prev is False
#                 mask_copy = mask.clone()  # create a copy
#                 mask_copy[head] = 1  # set active head to 1
#                 model.set_mask(mask_copy)  # set the model mask to this modified mask
#                 model.prev = trainer.evaluate()["eval_accuracy"]  # update model.prev with this accuracy
#             model.set_mask(mask)  # then resets the mask to the original mask
#             acc = trainer.evaluate()["eval_accuracy"]  # evaluate accuracy
#             model.track(head, acc)  # tracks contribution of the active head
#             model.true_prev = True
#         acc = -1 * acc  # negates accuracy
#         return acc
#
#     return attribute


# import parameters
config_path = sys.argv[1]

with open(config_path, "rb") as infile:
    config = tomli.load(infile)

input_dir = config["input_dir"]
output_dir = config["output_dir"]
target_uid = config["target_uid"]
device = config["device"]
inference_type = config["inference_type"]
finetuned_model = config["finetuned_model"]
head_mask = config["head_mask"]
mask_id = config["mask_id"]
attribution_path = config["attribution_path"]

n_samples = config["n_samples"]

model_name = config["model"]["name"]
layer_count = config["model"]["layer_count"]
attention_heads = config["model"]["attention_heads"]


n_full_mask = layer_count * attention_heads
logging.set_verbosity_error()
os.makedirs(output_dir, exist_ok=True)

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
config = model.config

model = PeftModelForSequenceClassification.from_pretrained(
        model=model,
        model_id=finetuned_model
        )
tokenizer = AutoTokenizer.from_pretrained(model_name)


metric = evaluate.load("accuracy")

compute_metrics = partial(compute_metrics, metric=metric, output_dir=None)


train_args = TrainingArguments(
        "/tmp/", per_device_eval_batch_size=100
        )

if inference_type == "shapley":
    mask = torch.ones((1, n_full_mask)).to(device)
    fake_model = MaskModel(model, mask, target_uid, output_dir, layer_count, attention_heads)
else:
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

# Assign Handler Function
signal.signal(signal.SIGINT, partial(signal_handler, fake_model=fake_model))

if inference_type == "shapley":
    target_model = fake_model
else:
    target_model = model

with open(os.path.join(input_dir, "overall_dataset_all_shuffle.pickle"), "rb") as infile:
    eval_dataset = pickle.load(infile)["test"]

# select target uid
eval_dataset = eval_dataset.filter(lambda example: example["UID"] == target_uid)

label_list = eval_dataset.features["label"].names

with torch.no_grad():

    model.eval()

    trainer = Trainer(
            model=target_model,
            args=train_args,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
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

        if mask_id != "None":
            tokens = tokenizer(eval_dataset["sentence_A"],
                               eval_dataset["sentence_B"],
                               padding=True, return_tensors="pt").to(device)
            output = model(**tokens, head_mask=mask.to(device))
            preds = torch.nn.functional.softmax(output.logits, dim=1).argmax(-1)
            acc = metric.compute(predictions=preds, references=eval_dataset["label"])

        elif mask_id == "None":
            acc = trainer.evaluate()["eval_accuracy"]

        print(f"Eval accuracy: {acc}")
        print()

        if mask_id is None:
            exp_file = f"{target_uid}_baseline.txt"
        elif mask_id != target_uid:
            exp_file = f"{mask_id}-{target_uid}.txt"
        else:
            exp_file = f"{mask_id}.txt"
        with open(os.path.join(output_dir, exp_file), "w") as outfile:
            outfile.write(str(acc))
