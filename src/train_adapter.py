import os
import sys
import tomli
import wandb
import random
import pickle
import evaluate
import warnings
import numpy.random as npr
from functools import partial
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from .utils import compute_metrics, preprocess_whole_dataset


# load parameters
config_path = sys.argv[1]

with open(config_path, "rb") as infile:
    config = tomli.load(infile)

model_name = config["model_name"]
seed = config["seed"]
preserve_dir = config["preserve_dir"]
metric_name = config["metric_name"]
output_dir = config["training"]["output_dir"]

# load device
device = config["device"]

# wandb project
os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
os.environ["WANDB_LOG_MODEL"] = config["wandb"]["log_model"]
wandb.init(project=config["wandb"]["project"])
wandb.config.update(config)


# seeding and randomness
random.seed(seed)
npr.seed(seed)
random_generator = npr.default_rng(seed=seed)

# load metric
metric = evaluate.load(metric_name)

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# process and save dataset
try:
    os.makedirs(preserve_dir, exist_ok=True)
except FileNotFoundError:
    warnings.warn("Preserve dir is not provided, the dataset won't be preserved.")

if config["preprocess"]["shuffle"] is True:
    filename = "dataset_shuffled.pickle"
else:
    filename = "dataset_not_shuffled.pickle"

if not os.path.exists(os.path.join(preserve_dir, filename)):
    dataset = preprocess_whole_dataset(
        **config["preprocess"], tokenizer=tokenizer,
        random_generator=random_generator,
    )
    with open(os.path.join(preserve_dir, filename), "wb") as outfile:
        pickle.dump(dataset, outfile)
else:
    with open(os.path.join(preserve_dir, filename), "rb") as infile:
        dataset = pickle.load(infile)

# initialise PEFT
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, **config["peft"]
)

model = get_peft_model(model, peft_config)

print()
print(model.print_trainable_parameters())
print()

training_args = TrainingArguments(
    **config["training"],
)

if config["test_run"]["size"]:
    dataset["train"] = dataset["train"].select(range(config["test_run"]["size"]))
    dataset["dev"] = dataset["dev"].select(range(config["test_run"]["size"]))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    compute_metrics=partial(compute_metrics, output_dir=output_dir, metric=metric),
)

trainer.train()

model.save_pretrained(output_dir)
