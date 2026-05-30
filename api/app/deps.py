from functools import lru_cache
from cachetools import LRUCache

_lru_cache: LRUCache = LRUCache(maxsize=256)
_model = None
_shap_explainer = None


def get_model():
    return _model


def get_shap_explainer():
    return _shap_explainer


def get_lru_cache() -> LRUCache:
    return _lru_cache
