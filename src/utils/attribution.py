import sys
import json
import torch
import subprocess
import numpy as np
from collections import defaultdict


def get_free_gpu():
    cmd = "nvidia-smi -q -d pids | grep -A4 GPU | grep Processes > tmp"
    p = subprocess.Popen(["/bin/bash", "-c", cmd])
    p.wait()
    memory_available = [x.split(":")[-1].strip() for x in open("tmp", "r").readlines()]
    print(memory_available)
    id = memory_available.index("None")
    print("Allocating Model to " + str(id))
    return id


def signal_handler(signal, frame, fake_model):
    print("You pressed Ctrl+C!")
    print(signal)  # Value is 2 for CTRL + C
    print(frame)  # Where your execution of program is at the moment - the Line Number
    fake_model.finish()
    sys.exit(0)


def bernstein(sample):
    """
    Calculates the confidence interval for the mean of a sample using Bernstein's inequality.
    """
    if len(sample) < 2:
        return -1, 1
    mean = np.mean(sample)
    variance = np.std(sample)
    delta = 0.1
    R = 1
    bern_bound = (variance * np.sqrt((2 * np.log(3 / delta))) / len(sample)) + (
            (3 * R * np.log(3 / delta)) / len(sample)
            )
    return mean - bern_bound, mean + bern_bound


class MaskModel(torch.nn.Module):
    """
    Designed for tracking and controlling the activity of different heads
    in a neural network model during training. It logs information about the contributions
    of each head and provides methods for setting masks, determining active heads, and
    resetting the model.
    """
    def __init__(self, real_model, head_mask, exp, output_dir, layer_count, attention_heads):
        super(MaskModel, self).__init__()
        self.contribs = defaultdict(self.construct_array)  # storing contributions of various heads during training
        self.counter = 0  # number of iterations/batches processed
        self.prev = 1.0  # variable storing previous accuracy during training
        self.exp = exp  # experiment identifier
        self.real_model = real_model  # underlying transformer model
        self.head_mask = head_mask  # mask controlling which heads are active
        self.true_prev = True
        self.prev_mask = torch.ones_like(head_mask).flatten()  # mask used for tracking changes in the head mask
        self.u = torch.zeros_like(head_mask).flatten()  # tensor used for tracking activity of heads
        self.tracker = open(output_dir + "/" + exp + "_tracker.txt", "a")  # file handle for tracking information
        self.sample_limit = 5000  # limit on number of samples tracking
        self.layers = layer_count
        self.attention_heads = attention_heads

    def construct_array(self):
        return []

    def track(self, head, acc):
        """
        Tracks contribution from different heads and updates counters,
        plus writes this information out to a file.
        """
        if head is not None:
            self.contribs[head].append(self.prev - acc)
        else:
            self.baseline = acc
        self.prev = acc
        if self.counter % 100 == 0:
            self.tracker.write(str(self.u.sum()) + "-" + str(self.counter) + "\n")
            self.tracker.flush()
        self.counter += 1

    def finish(self):
        """
        Writes contribution arrays to a file before closing it.
        """
        self.tracker.write("Contribution Arrays")
        self.tracker.write(json.dumps(self.contribs))
        self.tracker.close()

    def set_mask(self, mask):
        """
        Sets head mask to a specified value
        """
        mask = mask.reshape(self.layers, self.attention_heads)
        self.head_mask = mask

    def get_head(self, mask):
        """
        Determines active head by comparing the current head mask with the previous one.
        Returns index of the active head.
        """
        head = (mask.reshape(-1) != self.prev_mask.reshape(-1)).nonzero(as_tuple=True)[0]
        head = head.detach().cpu().tolist()[0]
        self.prev_mask = mask
        return head

    def active(self, head):
        """
        Determines if a particular head is active based on a stored value and a function.
        Uses sample limit for decision-making
        """
        def active_memo(head):
            contribs = np.array(self.contribs[head])
            lower, upper = bernstein(contribs)
            if lower > -0.01:
                return False
            elif len(contribs) > self.sample_limit:
                return False
            return True

        stored = self.u[head]
        if head == None:
            return True
        elif stored == 1:
            return False
        else:
            is_active = active_memo(head)
            if is_active:
                return True
            else:
                self.u[head] = 1
                return False

    def reset(self):
        """
        Resets model state, including masks and counters.
        """
        print("RESET")
        self.true_prev = True
        self.prev_mask = torch.ones_like(self.prev_mask).flatten()
        self.head_mask = torch.ones_like(self.head_mask)
        self.prev = self.baseline

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            # input_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            ):
        return self.real_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                # input_embeds=input_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                head_mask=self.head_mask
                )


# TODO: partial n_full_mask, trainer
def attribute_factory(model, n_full_mask, trainer):
    def attribute(mask):
        mask = mask.flatten()
        if mask.sum() == 1:  # if only 1 head is active, reset the model
            model.reset()
        mask = mask == 0  # invert mask order
        if not mask.sum() == n_full_mask:  # if not every head is active, get the active head
            head = model.get_head(mask)
        else:
            head = None  # if all attention heads active, set the active head to None
        if not model.active(head) or mask.sum() <= (n_full_mask//2):  # if the model is not active or if the sum is less than or equal half
            acc = model.prev  # set acc to previous accuracy
            model.true_prev = False  # set true prev to False
        else:
            if not model.true_prev:  # if model head is active or sum is more than half, check if model.true_prev is False
                mask_copy = mask.clone()  # create a copy
                mask_copy[head] = 1  # set active head to 1
                model.set_mask(mask_copy)  # set the model mask to this modified mask
                model.prev = trainer.evaluate()["eval_accuracy"]  # update model.prev with this accuracy
            model.set_mask(mask)  # then resets the mask to the original mask
            acc = trainer.evaluate()["eval_accuracy"]  # evaluate accuracy
            model.track(head, acc)  # tracks contribution of the active head
            model.true_prev = True
        acc = -1 * acc  # negates accuracy
        return acc

    return attribute

