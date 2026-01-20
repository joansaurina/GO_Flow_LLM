from functools import wraps
import click
import json
from typing import Callable


def mutually_exclusive_with_config(config_option: str = "config") -> Callable:
    """
    Decorator to ensure CLI parameters are mutually exclusive with config file.

    Args:
        config_option: Name of the config file option
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            ctx = click.get_current_context()
            config_file = kwargs.get(config_option)

            # Remove config option from kwargs for checking other params
            kwargs_without_config = {
                k: v for k, v in kwargs.items() if k != config_option
            }

            # Check if any other parameters are provided
            other_params_provided = any(
                v is not None for v in kwargs_without_config.values()
            )

            if config_file is None and not other_params_provided:
                raise click.UsageError(
                    "Either config file or CLI parameters must be provided."
                )

            # If config file is provided, read it and update kwargs
            if config_file is not None:
                try:
                    with open(config_file, "r") as file_handle:
                        config = json.load(file_handle)
                        # Update kwargs with config values
                        kwargs.update(config)
                except Exception as e:
                    raise click.UsageError(f"Error reading config file: {str(e)}")

            return f(*args, **kwargs)

        return wrapped_f

    return decorator