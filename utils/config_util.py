import os
from typing import Dict

import yaml


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
