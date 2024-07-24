import os
import pathlib
from functools import lru_cache

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@lru_cache(maxsize=1)
def get_config() -> dict:
    config_file = os.environ.get("CONFIG_FILE", None)
    if config_file is None:
        raise ValueError("CONFIG_FILE environment variable must be set")
    path = pathlib.Path(config_file)
    with path.open(mode="rb") as fp:
        return tomllib.load(fp)