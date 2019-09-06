import os
from typing import Dict, Tuple

import yaml


class Event:
    """
    Hamfisted event handler, so that the controller can send out update events to the GUI.

    NOTE: Using lambdas might have the potential for memory leaks - in a typical control flow you don't keep references to lambdas,
    so you can't really unsubscribe it later.
    """

    def __init__(self):
        self.subscribers = []

    def subscribe(self, subscriber_fn):
        self.subscribers.append(subscriber_fn)

    def unsubscribe(self, subscriber_fn):
        self.subscribers.remove(subscriber_fn)

    def notify(self):
        for subscriber_fn in self.subscribers:
            subscriber_fn()


def load_config(config_filepath: str) -> Dict:
    """
    Loads a YAML config file and expands placeholders.
    """

    with open(config_filepath, 'r') as stream:
        config = yaml.safe_load(stream)

    placeholders = {
        "${subdir_fname_without_ext}": os.path.splitext(config_filepath)[0]
    }

    # Expansion currently only works on top level!
    for attr_key, attr_val in config.items():
        if isinstance(attr_val, str):
            for p_k, p_v in placeholders.items():
                if p_k in attr_val:
                    attr_val = attr_val.replace(p_k, p_v)

            config[attr_key] = attr_val

    return config
