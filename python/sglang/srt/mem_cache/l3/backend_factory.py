import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict

from sglang.srt.mem_cache.l3.storage import HiCacheStorage, HiCacheStorageConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StorageBackendFactory:

    _registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def load_backend_class(
        module_path: str, class_name: str, backend_name: str
    ) -> type[HiCacheStorage]:
        try:
            module = importlib.import_module(module_path)
            backend_class = getattr(module, class_name)
            if not issubclass(backend_class, HiCacheStorage):
                raise TypeError(
                    f"Backend class {class_name} must inherit from HiCacheStorage"
                )
            return backend_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import backend '{backend_name}' from '{module_path}': {e}"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            ) from e

    @classmethod
    def register_backend(cls, name: str, module_path: str, class_name: str) -> None:
        if name in cls._registry:
            logger.warning(f"Backend '{name}' is already registered, overwriting")

        def loader() -> type[HiCacheStorage]:
            return cls.load_backend_class(module_path, class_name, name)

        cls._registry[name] = {
            "loader": loader,
            "module_path": module_path,
            "class_name": class_name,
        }

    @classmethod
    def create_backend(
        cls,
        backend_name: str,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
        **kwargs,
    ) -> HiCacheStorage:
        if backend_name in cls._registry:
            registry_entry = cls._registry[backend_name]
            backend_class = registry_entry["loader"]()
            logger.info(
                f"Creating storage backend '{backend_name}' "
                f"({registry_entry['module_path']}.{registry_entry['class_name']})"
            )
            return cls.create_builtin_backend(
                backend_name, backend_class, storage_config, mem_pool_host
            )

        if backend_name == "dynamic" and storage_config.extra_config is not None:
            backend_config = storage_config.extra_config
            return cls.create_dynamic_backend(
                backend_config, storage_config, mem_pool_host, **kwargs
            )

        available_backends = list(cls._registry.keys())

        raise ValueError(
            f"Unknown storage backend '{backend_name}'. "
            f"Registered backends: {available_backends}. "
        )

    @classmethod
    def create_dynamic_backend(
        cls,
        backend_config: Dict[str, Any],
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
        **kwargs,
    ) -> HiCacheStorage:
        required_fields = ["backend_name", "module_path", "class_name"]
        for field in required_fields:
            if field not in backend_config:
                raise ValueError(
                    f"Missing required field '{field}' in backend config for 'dynamic' backend"
                )

        backend_name = backend_config["backend_name"]
        module_path = backend_config["module_path"]
        class_name = backend_config["class_name"]

        try:
            backend_class = cls.load_backend_class(
                module_path, class_name, backend_name
            )

            logger.info(
                f"Creating dynamic storage backend '{backend_name}' "
                f"({module_path}.{class_name})"
            )

            return backend_class(storage_config, kwargs)
        except Exception as e:
            logger.error(
                f"Failed to create dynamic storage backend '{backend_name}': {e}"
            )
            raise

    @classmethod
    def create_builtin_backend(
        cls,
        backend_name: str,
        backend_class: type[HiCacheStorage],
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
    ) -> HiCacheStorage:
        if backend_name == "file":
            return backend_class(storage_config)
        elif backend_name == "mooncake":
            backend = backend_class(storage_config)
            return backend
        elif backend_name == "hf3fs":
            return backend_class(storage_config)
        elif backend_name in ("lmcache_local_disk", "lmcache_rust_raw_block"):
            backend = backend_class(storage_config)
            return backend
        else:
            raise ValueError(f"Unknown built-in backend: {backend_name}")


StorageBackendFactory.register_backend(
    "file", "sglang.srt.mem_cache.l3.simple_file", "SimpleFile"
)

StorageBackendFactory.register_backend(
    "mooncake",
    "sglang.srt.mem_cache.l3.mooncake_store.mooncake_store",
    "MooncakeStore",
)

StorageBackendFactory.register_backend(
    "hf3fs",
    "sglang.srt.mem_cache.l3.hf3fs.storage_hf3fs",
    "HiCacheHF3FS",
)

StorageBackendFactory.register_backend(
    "lmcache_local_disk",
    "sglang.srt.mem_cache.l3.lmcache.lmcache_backend",
    "LMCacheLocalDiskBackend",
)

StorageBackendFactory.register_backend(
    "lmcache_rust_raw_block",
    "sglang.srt.mem_cache.l3.lmcache.lmcache_backend",
    "LMCacheRustRawBlockBackend",
)
