import json
import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(state: Dict, save_path: str):
    """Robust, atomic checkpoint save (Windows-friendly).

    - Writes to a temporary file in the same directory, then atomically
      replaces the target file to avoid partial writes.
    - Uses legacy zipfile serialization to mitigate occasional write errors
      seen on Windows file systems.
    - Retries a few times if the write fails transiently.
    """
    import time

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_path = save_path + ".tmp"

    def _attempt_write(path: str):
        # Use legacy serialization to avoid inline_container writer issues
        with open(path, "wb") as f:
            torch.save(state, f, _use_new_zipfile_serialization=False)

    # Try write + atomic replace with small retry budget
    last_err = None
    for attempt in range(3):
        try:
            # Ensure any old temp file is cleared
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            _attempt_write(tmp_path)
            # Atomic replace on Windows 10+ and POSIX
            os.replace(tmp_path, save_path)
            return
        except Exception as e:
            last_err = e
            # Small backoff before retrying
            time.sleep(0.25 * (attempt + 1))
        finally:
            # Best-effort cleanup of temp file if something went wrong
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # If all attempts failed, raise the last error
    raise last_err


def load_checkpoint(path: str, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(path, map_location=device)