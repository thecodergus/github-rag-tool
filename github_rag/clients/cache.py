import os
import time
import json
import hashlib
import logging
from typing import Optional, Dict, Any

class CacheManager:
    """
    Gerencia cache de requisições HTTP usando arquivos JSON.
    """
    def __init__(
        self,
        cache_dir: str,
        cache_ttl: int = 86400,
        use_cache: bool = True,
    ):
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.use_cache = use_cache
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.logger = logging.getLogger(__name__)

    def get_cache_key(self, url: str, params: Dict = None) -> str:
        params_str = json.dumps(params or {}, sort_keys=True)
        key = f"{url}_{params_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, url: str, params: Dict = None) -> Optional[Dict[str, Any]]:
        if not self.use_cache:
            return None
        cache_key = self.get_cache_key(url, params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if not os.path.exists(cache_file):
            return None
        cache_age = time.time() - os.path.getmtime(cache_file)
        if cache_age < self.cache_ttl:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.logger.debug(f"Cache hit for {cache_key}")
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read cache {cache_key}: {e}")
        else:
            self.logger.debug(f"Cache expired for {cache_key} (age {cache_age:.1f}s)")
        return None

    def set(self, url: str, params: Dict, data: Dict[str, Any]) -> None:
        if not self.use_cache:
            return
        cache_key = self.get_cache_key(url, params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                self.logger.debug(f"Cached data for {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to write cache {cache_key}: {e}")