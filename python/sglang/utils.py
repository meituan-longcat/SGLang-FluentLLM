"""Common utilities"""

import base64
import importlib
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import weakref
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from json import dumps
from typing import Any, Callable, Optional, Union

import numpy as np
import requests
from IPython.display import HTML, display
from tqdm import tqdm

from sglang.srt.utils import kill_process_tree

logger = logging.getLogger(__name__)


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def is_same_type(values: list):
    """Return whether the elements in values are of the same type."""
    if len(values) <= 1:
        return True
    else:
        t = type(values[0])
        return all(isinstance(v, t) for v in values[1:])


def read_jsonl(filename: str):
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)


class HttpResponse:
    def __init__(self, resp):
        self.resp = resp

    def json(self):
        return json.loads(self.resp.read())

    @property
    def status_code(self):
        return self.resp.status


def http_request(
    url,
    json=None,
    stream=False,
    api_key=None,
    verify=None,
    method: Optional[str] = None,
):
    """A faster version of requests.post with low-level urllib API."""
    headers = {"Content-Type": "application/json; charset=utf-8"}

    # add the Authorization header if an api key is provided
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    if stream:
        return requests.post(url, json=json, stream=True, headers=headers, timeout=3600)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)  # noqa: S310
        data = None if json is None else bytes(dumps(json), encoding="utf-8")

        try:
            resp = urllib.request.urlopen(req, data=data, cafile=verify)  # noqa: S310
            return HttpResponse(resp)
        except urllib.error.HTTPError as e:
            return HttpResponse(e)


def encode_image_base64(image_path: Union[str, bytes]):
    """Encode an image in base64."""
    if isinstance(image_path, str):
        with open(image_path, "rb") as image_file:
            data = image_file.read()
            return base64.b64encode(data).decode("utf-8")
    elif isinstance(image_path, bytes):
        return base64.b64encode(image_path).decode("utf-8")
    else:
        # image_path is PIL.WebPImagePlugin.WebPImageFile
        image = image_path
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_frame(frame):
    import cv2  # pip install opencv-python-headless
    from PIL import Image

    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image to easily convert to bytes
    im_pil = Image.fromarray(frame)

    # Convert to bytes
    buffered = BytesIO()

    # frame_format = str(os.getenv('FRAME_FORMAT', "JPEG"))

    im_pil.save(buffered, format="PNG")

    frame_bytes = buffered.getvalue()

    # Return the bytes of the frame
    return frame_bytes


def encode_video_base64(video_path: str, num_frames: int = 16):
    import cv2  # pip install opencv-python-headless

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video file:{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"target_frames: {num_frames}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            # Handle the case where the frame could not be read
            pass

    cap.release()

    # Safely select frames based on frame_indices, avoiding IndexError
    frames = [frames[i] for i in frame_indices if i < len(frames)]

    # If there are not enough frames, duplicate the last frame until we reach the target
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Use ThreadPoolExecutor to process and encode frames in parallel
    with ThreadPoolExecutor() as executor:
        encoded_frames = list(executor.map(encode_frame, frames))

    # encoded_frames = list(map(encode_frame, frames))

    # Concatenate all frames bytes
    video_bytes = b"".join(encoded_frames)

    # Encode the concatenated bytes to base64
    video_base64 = "video:" + base64.b64encode(video_bytes).decode("utf-8")

    return video_base64


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    return (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    )


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n") or len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


def graceful_registry(sub_module_name: str):
    def graceful_shutdown(signum, frame):
        logger.info(f"{sub_module_name} Received signal to shutdown. Performing graceful shutdown...")
        if signum == signal.SIGTERM:
            logger.info(f"{sub_module_name} recive sigterm")

    signal.signal(signal.SIGTERM, graceful_shutdown)


class LazyImport:
    """Lazy import to make `import sglang` run faster."""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    def _load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join(tempfile.gettempdir(), url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True, timeout=3600)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with (
        open(filename, "wb") as f,
        tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def is_in_ci():
    from sglang.test.test_utils import is_in_ci

    return is_in_ci()


def print_highlight(html_content: str):
    if is_in_ci():
        html_content = str(html_content).replace("\n", "<br>")
        display(HTML(f"<strong style='color: #00008B;'>{html_content}</strong>"))
    else:
        print(html_content)


process_socket_map = weakref.WeakKeyDictionary()


def reserve_port(host, start=30000, end=40000):
    """
    Reserve an available port by trying to bind a socket.
    Returns a tuple (port, lock_socket) where `lock_socket` is kept open to hold the lock.
    """
    candidates = list(range(start, end))
    random.shuffle(candidates)

    for port in candidates:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # Attempt to bind to the port on localhost
            sock.bind((host, port))
            return port, sock
        except OSError:
            sock.close()  # Failed to bind, try next port
            continue
    raise RuntimeError("No free port available.")


def release_port(lock_socket):
    """
    Release the reserved port by closing the lock socket.
    """
    try:
        lock_socket.close()
    except Exception as e:
        print(f"Error closing socket: {e}")


def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    """
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT)  # noqa: S603


def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = None):  # noqa: S104
    """
    Launch the server using the given command.
    If no port is specified, a free port is reserved.
    """
    if port is None:
        port, lock_socket = reserve_port(host)
    else:
        lock_socket = None

    full_command = f"{command} --port {port}"
    process = execute_shell_command(full_command)

    if lock_socket is not None:
        process_socket_map[process] = lock_socket

    return process, port


def terminate_process(process):
    """
    Terminate the process and automatically release the reserved port.
    """
    kill_process_tree(process.pid)

    lock_socket = process_socket_map.pop(process, None)
    if lock_socket is not None:
        release_port(lock_socket)


def wait_for_server(base_url: str, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
                timeout=30,
            )
            if response.status_code == 200:
                time.sleep(5)
                print_highlight(
                    """\n
                    NOTE: Typically, the server runs in a separate terminal.
                    In this notebook, we run the server and notebook code together, so their outputs are combined.
                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.
                    We are running those notebooks in a CI parallel environment, so the throughput is not representative of the actual performance.
                    """
                )
                break

            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)


class TypeBasedDispatcher:
    def __init__(self, mapping: list[tuple[type, Callable]]):
        self._mapping = mapping
        self._fallback_fn = None

    def add_fallback_fn(self, fallback_fn: Callable):
        self._fallback_fn = fallback_fn

    def __iadd__(self, other: "TypeBasedDispatcher"):
        self._mapping.extend(other._mapping)
        return self

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)

        if self._fallback_fn is not None:
            return self._fallback_fn(obj)
        raise ValueError(f"Invalid object: {obj}")


def trim_overlap(existing_text, new_chunk):
    """
    Finds the largest suffix of 'existing_text' that is a prefix of 'new_chunk'
    and removes that overlap from the start of 'new_chunk'.
    """
    max_overlap = 0
    max_possible = min(len(existing_text), len(new_chunk))
    for i in range(max_possible, 0, -1):
        if existing_text.endswith(new_chunk[:i]):
            max_overlap = i
            break
    return new_chunk[max_overlap:]


def stream_and_merge(llm, prompt, sampling_params):
    """
    1) Streams the text,
    2) Removes chunk overlaps,
    3) Returns the merged text.
    """
    final_text = ""
    for chunk in llm.generate(prompt, sampling_params, stream=True):
        chunk_text = chunk["text"]
        cleaned_chunk = trim_overlap(final_text, chunk_text)
        final_text += cleaned_chunk
    return final_text


async def async_stream_and_merge(llm, prompt, sampling_params):
    """
    Streams tokens asynchronously, removes chunk overlaps,
    and yields the cleaned chunk in real time for printing.
    """
    final_text = ""
    generator = await llm.async_generate(prompt, sampling_params, stream=True)
    async for chunk in generator:
        chunk_text = chunk["text"]
        cleaned_chunk = trim_overlap(final_text, chunk_text)
        final_text += cleaned_chunk
        yield cleaned_chunk  # yield the non-overlapping portion


def get_cat_reporter(model_name: str, app_key: str):
    class EmptyCat:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    if not model_name:
        return EmptyCat()

    DEFAULT_CAT_TYPE = "Metrics." + model_name
    try:
        from functools import partial

        from pycat import Cat, CatStatusEnum

        if app_key is None:
            app_key = "inference.fluent"
        Cat.init_cat(app_key, disable_falcon=True)
    except Exception as e:
        logger.error(f"Error init Cat: {e}")
        return EmptyCat()
    else:

        def log_transaction(t_type: str, t_name: str, duration: int, suceess: bool = True):
            t = Cat.new_transaction(t_type, t_name)
            if suceess:
                t.duration = duration
            else:
                t.set_status(CatStatusEnum.EXCEPTION)
            t.complete()

        def log_duration(t_name: str, end: float, start: float = 0.0):
            log_transaction(DEFAULT_CAT_TYPE, t_name, duration=int((end - start) * 1000000))

        def log_count(t_name: str, count: float):
            log_transaction(DEFAULT_CAT_TYPE, t_name, duration=int(count * 1000))

        def log_result(t_name: str, success: bool):
            log_transaction(DEFAULT_CAT_TYPE, t_name, duration=0, suceess=success)

        def log_success_rate(t_name: str, success: bool):
            log_transaction(
                DEFAULT_CAT_TYPE,
                t_name,
                duration=0,
                suceess=success,
            )

        Cat.log_count = staticmethod(log_count)
        Cat.log_duration = staticmethod(log_duration)
        Cat.log_result = staticmethod(log_result)
        Cat.log_success_rate = staticmethod(log_success_rate)
        Cat.log_transaction = staticmethod(log_transaction)
        Cat.log_batch_event = staticmethod(partial(Cat.log_batch_event, e_type=DEFAULT_CAT_TYPE))
        return Cat
