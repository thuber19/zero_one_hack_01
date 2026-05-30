from cachetools import LRUCache
import hashlib
import json

_cache: LRUCache = LRUCache(maxsize=256)


def get_cache() -> LRUCache:
    return _cache


def make_sequence_key(sequence: list[dict]) -> str:
    serialized = json.dumps(sequence, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()


def cache_get(key: str):
    return _cache.get(key)


def cache_set(key: str, value) -> None:
    _cache[key] = value


def cache_size() -> int:
    return len(_cache)
