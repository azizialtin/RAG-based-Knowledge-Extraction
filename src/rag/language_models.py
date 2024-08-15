# pylint:disable = broad-exception-raised

"""
This module defines methods needed for loading and checking the LLM
"""

import ollama
from tqdm import tqdm
from loguru import logger


def is_model_available(model_name: str) -> None:
    """
    Checks if the model is available locally. If not, attempts to pull it.
    :param model_name: Name of the model to check.
    :returns: None
    """
    try:
        logger.debug("Check if the model is available locally")
        available = __is_model_available_locally(model_name)
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        raise Exception("Unable to communicate with the Ollama service, "
                        "make sure to download Ollama "
                        "from https://ollama.com/") from e

    if not available:
        try:
            logger.debug("Model: {model_name} not available locally, "
                         "trying to pull the model.")
            # Attempt to pull the model if not available
            __pull_model(model_name)
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            raise Exception(
                f"Unable to find model '{model_name}', "
                f"please check the name and try again."
            ) from e


def __is_model_available_locally(model_name: str) -> bool:
    """
    Checks if the specified model is available locally.
    :param model_name: Name of the model to check.
    :return: True if the model is available locally, False otherwise.
    """
    try:
        # If ollama.show doesn't raise an error, the model is available
        ollama.show(model_name)
        return True
    except ollama.ResponseError:
        # If ResponseError is raised, the model is not available
        return False


def __pull_model(name: str) -> None:
    """
    Pulls the model with the specified name using Ollama, showing progress.
    :param name: Name of the model to pull.
    """
    current_digest, bars = "", {}
    # Pull the model with streaming progress
    for progress in ollama.pull(name, stream=True):
        digest = progress.get("digest", "")
        # Close the previous progress bar if a new digest is received
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()

        # If no digest is provided, log the status and continue
        if not digest:
            logger.info(progress.get("status"))
            continue

        # Create a new progress bar if one doesn't exist for the current digest
        if digest not in bars and (total := progress.get("total")):
            bars[digest] = tqdm(
                total=total,
                desc=f"pulling {digest[7:19]}",
                unit="B",
                unit_scale=True
            )

        # Update the progress bar with the number of bytes completed#
        if completed := progress.get("completed"):
            bars[digest].update(completed - bars[digest].n)

        # Update the current digest
        current_digest = digest
