import os
import importlib
from sglang.srt.utils import (
    get_colorful_logger,
)
logger = get_colorful_logger(__name__)
# ============ 动态导入 RequestCache ============
def _import_request_cache():
    # 读取环境变量
    module_path = os.getenv("REQUEST_CACHE_MODULE_PATH", "")
    class_name = os.getenv("REQUEST_CACHE_CLASS_NAME", "RequestCache")
    logger.info(f"Loading RequestCache from: module={module_path}, class={class_name}")
    if module_path == "":
        return None
    try:
        module = importlib.import_module(module_path)
        RequestCache = getattr(module, class_name)
        logger.info(f"Successfully loaded RequestCache from {module_path}")
        return RequestCache
    except (ImportError, AttributeError) as e:
        logger.error(
            f"Failed to import RequestCache from {module_path}: {e}. "
            f"Falling back to default sglang.srt.mem_cache.request_cache"
        )
RequestCache = _import_request_cache()