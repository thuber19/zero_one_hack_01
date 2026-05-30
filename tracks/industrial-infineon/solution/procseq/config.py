"""YAML config loader with dotted access."""
from pathlib import Path
import yaml

class Config(dict):
    __getattr__ = dict.get

def load_config(path) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    return Config(data)
