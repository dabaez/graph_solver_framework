import importlib

import torch
import torch.nn.functional as F
from logzero import logger

from .model import GCN


def _load_model(prob_maps, weight_file=None, cuda_dev=None):
    model = GCN(1, 32, prob_maps, 20, F.relu, 0)
    if cuda_dev is not None:
        model = model.to(cuda_dev)

    if weight_file:
        if cuda_dev:
            model.load_state_dict(torch.load(weight_file, map_location=cuda_dev))
        else:
            model.load_state_dict(
                torch.load(weight_file, map_location=torch.device("cpu"))
            )

    return model


def _locked_log(lock, msg, loglevel):
    if lock:
        if loglevel == "DEBUG":
            with lock:
                logger.debug(msg)
        elif loglevel == "INFO":
            with lock:
                logger.info(msg)
        elif loglevel == "WARN":
            with lock:
                logger.warning(msg)
        elif loglevel == "ERROR":
            with lock:
                logger.error(msg)
        else:
            with lock:
                logger.error(
                    f"The following message was logged with unknown log-level {loglevel}:\n{msg}"
                )
    else:
        if loglevel == "DEBUG":
            logger.debug(msg)
        elif loglevel == "INFO":
            logger.info(msg)
        elif loglevel == "WARN":
            logger.warning(msg)
        elif loglevel == "ERROR":
            logger.error(msg)
        else:
            logger.error(
                f"The following message was logged with unknown log-level {loglevel}:\n{msg}"
            )


def find_module(full_module_name):
    """
    Returns module object if module `full_module_name` can be imported.

    Returns None if module does not exist.

    Exception is raised if (existing) module raises exception during its import.
    """
    try:
        return importlib.import_module(full_module_name)
    except ImportError as exc:
        if not (full_module_name + ".").startswith(str(exc.name) + "."):
            raise
