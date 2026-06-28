import json
import os
import pickle
import random
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
try:
    from nextqa import NExTQALoader
    NEXTQA_AVAILABLE = True
except ImportError:
    NEXTQA_AVAILABLE = False
    NExTQALoader = None

# from nextqa.video import , VideoPrompt
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.bench_serving import (
    SHAREGPT_URL,
    download_and_cache_file,
    gen_prompt,
    get_gen_prefix_cache_path,
)

# Try to import chat_template functions, but make them optional
try:
    from sglang.lang.chat_template import get_chat_template, get_chat_template_by_model_path
    CHAT_TEMPLATE_AVAILABLE = True
except ImportError:
    # Fallback: these are only needed for video benchmarks
    CHAT_TEMPLATE_AVAILABLE = False
    get_chat_template = None
    get_chat_template_by_model_path = None

from sglang.srt.entrypoints.openai.protocol import ChatCompletionMessageContentPart

# Try to import video encoding utility
try:
    from sglang.utils import encode_video_base64
    VIDEO_ENCODING_AVAILABLE = True
except ImportError:
    VIDEO_ENCODING_AVAILABLE = False
    encode_video_base64 = None

# type of content fields, can be only prompts or with images/videos
MsgContent = Union[str, List[ChatCompletionMessageContentPart]]

# A list of all the conversations. Each conversation is a list of
# tuples. If multiturn is not enabled, the length of list is 1,
# containing only the first Q&A pair.
# For the shared prefix workload (synthetic, loogle, nextqa), it
# is a list of conversations sharing the same prefix (synthetic,
# doc, video)
SampleOutput = List[List[Tuple[MsgContent, int, int]]]


class FixedDatasetManager:
    """Manager for creating and accessing fixed, deterministic ShareGPT datasets."""

    def __init__(self, cache_dir: str = "~/.cache/sglang/fixed_datasets"):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Dataset version for cache invalidation
        self.dataset_version = "v1.0"

        # Scenario configurations
        self.scenario_configs = {
            'serving': {
                'description': 'Short conversations for general serving tests',
                'min_turns': 1,
                'max_turns': 2,
                'min_total_tokens': 100,
                'max_total_tokens': 4000,
                'prefer_short': True
            },
            'multiturn': {
                'description': 'Medium conversations for multi-turn scenarios',
                'min_turns': 2,
                'max_turns': 6,
                'min_total_tokens': 1000,
                'max_total_tokens': 8000,
                'prefer_medium': True
            },
            'mix': {
                'description': 'Mixed workload with diverse conversation types',
                'min_turns': 1,
                'max_turns': 8,
                'min_total_tokens': 100,
                'max_total_tokens': 12000,
                'prefer_mixed': True
            },
            'long_context': {
                'description': 'Long conversations for cache reuse testing',
                'min_turns': 3,
                'max_turns': 12,
                'min_total_tokens': 4000,
                'max_total_tokens': 20000,
                'prefer_long': True
            }
        }

    def _get_cache_path(self, scenario: str, num_requests: int, tokenizer_name: str,
                       fixed_output_len: Optional[int] = None) -> Path:
        """Generate cache file path for a specific configuration."""
        cache_key = f"{scenario}_{num_requests}_{tokenizer_name}_{fixed_output_len}_{self.dataset_version}.pkl"
        return self.cache_dir / cache_key

    def _get_metadata_path(self) -> Path:
        """Get path for metadata file."""
        return self.cache_dir / "metadata.json"

    def _classify_conversation(self, conversation: List[Tuple[str, str]],
                              tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
        """Classify a conversation by its characteristics."""
        num_turns = len(conversation)

        # Calculate total tokens
        total_tokens = 0
        for user_msg, assistant_msg in conversation:
            total_tokens += len(tokenizer.encode(user_msg))
            total_tokens += len(tokenizer.encode(assistant_msg))

        # Classify by length
        if total_tokens < 2000:
            length_category = 'short'
        elif total_tokens < 6000:
            length_category = 'medium'
        else:
            length_category = 'long'

        # Classify by turn count
        if num_turns <= 2:
            turn_category = 'single'
        elif num_turns <= 4:
            turn_category = 'multi'
        else:
            turn_category = 'extended'

        return {
            'num_turns': num_turns,
            'total_tokens': total_tokens,
            'length_category': length_category,
            'turn_category': turn_category,
            'conversation_id': hashlib.md5(str(conversation).encode()).hexdigest()[:8]
        }

    def _select_conversations_for_scenario(self, conversations: List[List[Tuple[str, str]]],
                                         scenario: str, num_requests: int,
                                         tokenizer: PreTrainedTokenizerBase) -> List[List[Tuple[str, str]]]:
        """Select conversations that match the scenario requirements."""
        config = self.scenario_configs[scenario]

        # Classify all conversations
        classified_conversations = []
        for conv in conversations:
            classification = self._classify_conversation(conv, tokenizer)
            if (config['min_turns'] <= classification['num_turns'] <= config['max_turns'] and
                config['min_total_tokens'] <= classification['total_tokens'] <= config['max_total_tokens']):
                classified_conversations.append((conv, classification))

        # Sort by conversation ID for deterministic ordering
        classified_conversations.sort(key=lambda x: x[1]['conversation_id'])

        # Select based on scenario preferences
        selected = []
        if config.get('prefer_short'):
            # Prefer shorter conversations
            classified_conversations.sort(key=lambda x: x[1]['total_tokens'])
        elif config.get('prefer_long'):
            # Prefer longer conversations
            classified_conversations.sort(key=lambda x: -x[1]['total_tokens'])
        elif config.get('prefer_medium'):
            # Prefer medium-length conversations
            classified_conversations.sort(key=lambda x: abs(x[1]['total_tokens'] - 4000))
        elif config.get('prefer_mixed'):
            # Create a balanced mix
            short_convs = [x for x in classified_conversations if x[1]['length_category'] == 'short']
            medium_convs = [x for x in classified_conversations if x[1]['length_category'] == 'medium']
            long_convs = [x for x in classified_conversations if x[1]['length_category'] == 'long']

            # Distribute evenly
            per_category = num_requests // 3
            remainder = num_requests % 3

            selected.extend(short_convs[:per_category + (1 if remainder > 0 else 0)])
            selected.extend(medium_convs[:per_category + (1 if remainder > 1 else 0)])
            selected.extend(long_convs[:per_category])

            # Sort by conversation ID for final deterministic order
            selected.sort(key=lambda x: x[1]['conversation_id'])
            return [conv for conv, _ in selected[:num_requests]]

        # For non-mixed scenarios, just take the first num_requests
        selected = classified_conversations[:num_requests]
        return [conv for conv, _ in selected]

    def create_fixed_dataset_cache(self, dataset_path: str, tokenizer: PreTrainedTokenizerBase,
                                  force_regenerate: bool = False) -> None:
        """Pre-process and cache fixed datasets for different scenarios."""
        print(f"Creating fixed dataset cache from {dataset_path}")

        # Download sharegpt if necessary
        if not os.path.isfile(dataset_path):
            dataset_path = download_and_cache_file(SHAREGPT_URL)

        # Load the dataset
        with open(dataset_path) as f:
            dataset = json.load(f)

        # Filter out conversations with less than 2 turns
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]

        # Convert to our format
        conversations = []
        for data in dataset:
            if len(data["conversations"]) % 2 != 0:
                continue
            if data["conversations"][0]["from"] != "human":
                continue

            chat = []
            for i in range(0, len(data["conversations"]), 2):
                if i + 1 < len(data["conversations"]):
                    chat.append((
                        data["conversations"][i]["value"],
                        data["conversations"][i + 1]["value"]
                    ))
            if chat:
                conversations.append(chat)

        print(f"Loaded {len(conversations)} conversations from ShareGPT")

        # Create metadata
        metadata = {
            'dataset_version': self.dataset_version,
            'source_dataset': dataset_path,
            'total_conversations': len(conversations),
            'tokenizer_name': tokenizer.__class__.__name__,
            'scenarios': self.scenario_configs,
            'created_at': str(Path(dataset_path).stat().st_mtime if os.path.exists(dataset_path) else 'unknown')
        }

        # Save metadata
        with open(self._get_metadata_path(), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {self._get_metadata_path()}")
        print("Fixed dataset cache creation completed")

    def get_fixed_dataset(self, scenario: str, num_requests: int,
                         tokenizer: PreTrainedTokenizerBase,
                         dataset_path: str,
                         fixed_output_len: Optional[int] = None) -> SampleOutput:
        """Return deterministic subset based on scenario."""
        if scenario not in self.scenario_configs:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(self.scenario_configs.keys())}")

        cache_path = self._get_cache_path(scenario, num_requests, tokenizer.__class__.__name__, fixed_output_len)

        # Try to load from cache
        if cache_path.exists():
            print(f"Loading fixed dataset from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"Generating fixed dataset for scenario '{scenario}' with {num_requests} requests")

        # Download sharegpt if necessary
        if not os.path.isfile(dataset_path):
            dataset_path = download_and_cache_file(SHAREGPT_URL)

        # Load the dataset
        with open(dataset_path) as f:
            dataset = json.load(f)

        # Filter and convert conversations
        conversations = []
        for data in dataset:
            if len(data["conversations"]) < 2 or len(data["conversations"]) % 2 != 0:
                continue
            if data["conversations"][0]["from"] != "human":
                continue

            chat = []
            for i in range(0, len(data["conversations"]), 2):
                if i + 1 < len(data["conversations"]):
                    chat.append((
                        data["conversations"][i]["value"],
                        data["conversations"][i + 1]["value"]
                    ))
            if chat:
                conversations.append(chat)

        # Select conversations for this scenario
        selected_conversations = self._select_conversations_for_scenario(
            conversations, scenario, num_requests, tokenizer
        )

        # Convert to SampleOutput format and apply filtering
        new_dataset = selected_conversations

        # Apply the same filtering as the original function
        # Use override_min_output_len=True to allow fixed_output_len < 4 if explicitly set
        filtered_dataset: SampleOutput = common_filter_chat(
            num_requests, new_dataset, tokenizer, 4, 4, None, None, fixed_output_len,
            override_min_output_len=True
        )

        # Cache the result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(filtered_dataset, f)

        print(f"Cached fixed dataset to: {cache_path}")
        return filtered_dataset


def sample_fixed_sharegpt_requests(
    scenario: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    cache_dir: Optional[str] = None,
    fixed_output_len: Optional[int] = None,
) -> SampleOutput:
    """Return deterministic ShareGPT subset based on scenario."""

    # Initialize the fixed dataset manager
    manager = FixedDatasetManager(cache_dir or "~/.cache/sglang/fixed_datasets")

    # Get the fixed dataset for the specified scenario
    return manager.get_fixed_dataset(
        scenario=scenario,
        num_requests=num_requests,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        fixed_output_len=fixed_output_len
    )


def classify_conversations_by_length(conversations: List[List[Tuple[str, str]]],
                                   tokenizer: PreTrainedTokenizerBase) -> Dict[str, List[List[Tuple[str, str]]]]:
    """Classify conversations as short/medium/long based on token count."""
    categories = {'short': [], 'medium': [], 'long': []}

    for conv in conversations:
        total_tokens = 0
        for user_msg, assistant_msg in conv:
            total_tokens += len(tokenizer.encode(user_msg))
            total_tokens += len(tokenizer.encode(assistant_msg))

        if total_tokens < 2000:
            categories['short'].append(conv)
        elif total_tokens < 6000:
            categories['medium'].append(conv)
        else:
            categories['long'].append(conv)

    return categories


def select_deterministic_subset(conversations: List[List[Tuple[str, str]]],
                               num_requests: int,
                               scenario: str) -> List[List[Tuple[str, str]]]:
    """Select fixed subset using conversation ID hash for deterministic ordering."""

    # Create deterministic ordering based on conversation content hash
    conversations_with_hash = []
    for conv in conversations:
        conv_hash = hashlib.md5(str(conv).encode()).hexdigest()
        conversations_with_hash.append((conv, conv_hash))

    # Sort by hash for deterministic ordering
    conversations_with_hash.sort(key=lambda x: x[1])

    # Select the first num_requests conversations
    selected = [conv for conv, _ in conversations_with_hash[:num_requests]]

    return selected


def common_filter_chat(
    num_requests: int,
    new_dataset: List,
    tokenizer: PreTrainedTokenizerBase,
    min_prompt_len: Optional[int],
    min_output_len: Optional[int],
    max_prompt_len: Optional[int],
    max_output_len: Optional[int],
    fixed_output_len: Optional[int],
    override_min_output_len: bool = False,
) -> SampleOutput:
    # Filter out sequences that are too long or too short
    filtered_dataset: SampleOutput = []
    l = 0
    input_tokens = 0
    output_tokens = 0
    while l < num_requests:
        for i in range(len(new_dataset)):
            if l == num_requests:
                break
            processed = []
            for j in new_dataset[i]:
                # Tokenize the prompts and completions.
                prompt = j[0]
                prompt_token_ids = tokenizer.encode(prompt)
                prompt_len = len(prompt_token_ids)

                completion = j[1]
                completion_token_ids = tokenizer.encode(completion)
                output_len = (
                    len(completion_token_ids)
                    if fixed_output_len is None
                    else fixed_output_len
                )
                # If override_min_output_len is True and fixed_output_len is set, skip min_output_len check
                effective_min_output_len = None if (override_min_output_len and fixed_output_len is not None) else min_output_len
                
                if (
                    min_prompt_len is not None
                    and prompt_len < min_prompt_len
                    or effective_min_output_len is not None
                    and output_len < effective_min_output_len
                    or max_prompt_len is not None
                    and prompt_len > max_prompt_len
                    or max_output_len is not None
                    and output_len > max_output_len
                ):
                    # Prune too short sequences.
                    continue
                input_tokens += prompt_len
                output_tokens += output_len
                processed.append((prompt, prompt_len, output_len))
            if len(processed) != 0:
                filtered_dataset.append(processed)
                l += 1

    print(f"#Input tokens: {input_tokens}")
    print(f"#Output tokens: {output_tokens}")
    return filtered_dataset


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    disable_shuffle: bool = False,
    enable_multiturn: bool = True,
    fixed_output_len: Optional[int] = None,
    use_fixed_data: bool = True,
    scenario: str = 'serving',
    cache_dir: Optional[str] = None,
) -> SampleOutput:
    # Note: Removed the fixed_output_len < 4 check to allow small output lengths
    # The filtering is now handled by override_min_output_len parameter in common_filter_chat

    # Use fixed data if requested (new default behavior)
    if use_fixed_data:
        return sample_fixed_sharegpt_requests(
            scenario=scenario,
            num_requests=num_requests,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            cache_dir=cache_dir,
            fixed_output_len=fixed_output_len,
        )

    # Original random/shuffle-based behavior for backward compatibility
    print("Using legacy random ShareGPT sampling (use_fixed_data=False)")

    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # Keep one conversation in one list
    new_dataset = []
    for data in dataset:
        if len(data["conversations"]) % 2 != 0:
            continue
        if data["conversations"][0]["from"] != "human":
            continue
        chat = []
        total_len = 2
        if enable_multiturn:
            total_len = len(data["conversations"])
        for i in range(0, total_len, 2):
            # One user One Assistant
            chat.append(
                (
                    data["conversations"][i]["value"],
                    data["conversations"][i + 1]["value"],
                )
            )
        new_dataset.append(chat)

    if not disable_shuffle:
        # Shuffle the dataset.
        random.shuffle(new_dataset)

    # Filter out sequences that are too long or too short
    # Use override_min_output_len=True to allow fixed_output_len < 4 if explicitly set
    filtered_dataset: SampleOutput = common_filter_chat(
        num_requests, new_dataset, tokenizer, 4, 4, None, None, fixed_output_len,
        override_min_output_len=True
    )
    return filtered_dataset


def sample_ultrachat_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    disable_shuffle: bool = False,
    enable_multiturn: bool = True,
    fixed_output_len: Optional[int] = None,
) -> SampleOutput:
    # Note: Removed the fixed_output_len < 4 check to allow small output lengths
    # The filtering is now handled by override_min_output_len parameter in common_filter_chat

    # Load the dataset
    dataset = []
    with open(dataset_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            dataset.append(json.loads(line))

    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["data"]) >= 2]

    # Keep one conversation in one list
    new_dataset = []
    for data in dataset:
        if len(data["data"]) % 2 != 0:
            continue
        chat = []
        total_len = 2
        if enable_multiturn:
            total_len = len(data["data"])
        for i in range(0, total_len, 2):
            # One user One Assistant
            chat.append((data["data"][i], data["data"][i + 1]))
        new_dataset.append(chat)

    # Shuffle the dataset.
    if not disable_shuffle:
        random.shuffle(new_dataset)

    # Filter out sequences that are too long or too short
    # Use override_min_output_len=True to allow fixed_output_len < 4 if explicitly set
    filtered_dataset: SampleOutput = common_filter_chat(
        num_requests, new_dataset, tokenizer, 4, 4, None, None, fixed_output_len,
        override_min_output_len=True
    )
    return filtered_dataset


def sample_loogle_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    disable_shuffle: bool = False,
    enable_multiturn: bool = True,
    enable_shared_prefix: bool = False,
    fixed_output_len: Optional[int] = None,
) -> SampleOutput:
    # Note: Removed the fixed_output_len < 4 check to allow small output lengths
    # The filtering is now handled by override_min_output_len parameter in common_filter_chat

    # Load the dataset
    dataset = []
    with open(dataset_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            dataset.append(json.loads(line))

    # Keep one conversation in one list
    new_dataset = []
    # TODO: Add shared prefix support for loogle
    # NOTE: Now we preprocess it only for chat
    for data in dataset:
        chat = []
        if (
            "qa_pairs" not in data
            or data["qa_pairs"] == "none"
            or len(data["qa_pairs"]) == 0
        ):
            # If Q is none (for summarization),
            # We add a question for summarization
            # And keep the summary up to 1024 words
            chat.append(
                (
                    "Input: "
                    + data["input"]
                    + " Question: "
                    + "Please summarize the input",
                    data["input"][:1024],
                )
            )
            new_dataset.append(chat)
        else:
            qa_pairs = eval(data["qa_pairs"])
            for i, qa in enumerate(qa_pairs):
                if i == 0 or enable_shared_prefix:
                    # Combine input with the first Q
                    chat.append(
                        ("Input: " + data["input"] + " Question: " + qa["Q"], qa["A"])
                    )
                elif enable_multiturn:
                    chat.append((qa["Q"], qa["A"]))

            new_dataset.append(chat)

    # Shuffle the dataset.
    if not disable_shuffle:
        random.shuffle(new_dataset)

    # Filter out sequences that are too long or too short
    # Note: loogle uses min_output_len=None, so no need for override_min_output_len
    filtered_dataset: SampleOutput = common_filter_chat(
        num_requests, new_dataset, tokenizer, 4, None, None, None, fixed_output_len
    )
    return filtered_dataset


def sample_nextqa_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    max_frames: int,  # Specific for video
    model_path: str,
    disable_shuffle: bool = False,
    enable_multiturn: bool = True,  # No multiturn support for now
    backend: str = "sglang-oai",
    chat_template_name: Optional[str] = None,
    fixed_output_len: Optional[int] = None,
) -> SampleOutput:
    """
    Example of messages:
    message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": base64_data}},
            {"type": "text", "text": video.prompt},
        ],
    }
    """

    if fixed_output_len is None:
        fixed_output_len = 4096

    # TODO: Check for multiturn
    if not NEXTQA_AVAILABLE:
        raise ImportError("NExTQA functionality requires 'av' module. Install with: pip install av")
    dataset = NExTQALoader(video_dir=dataset_path, max_frames=max_frames)
    new_dataset = []
    for v in dataset:
        new_dataset.append(v)

    if not disable_shuffle:
        random.shuffle(new_dataset)

    # TODO: prompt len can get from server side
    filtered_dataset = []
    l = 0
    while l < num_requests:
        for i in range(len(new_dataset)):
            if l == num_requests:
                break

            video = new_dataset[i]

            # text prompt
            prompt = video.prompt

            # NOTE: Chat Template is a must for video benchmark because we have to
            # add special image token for later expansion
            if backend == "sglang" or backend == "sglang-native":
                if not CHAT_TEMPLATE_AVAILABLE:
                    raise ImportError(
                        "Chat template functions are not available. "
                        "This is required for video benchmarks. "
                        "Please ensure sglang.lang.chat_template module is available."
                    )
                if "chat_template" in tokenizer.init_kwargs:
                    chat_template = get_chat_template(tokenizer.get_chat_template())
                elif chat_template_name is not None:
                    chat_template = get_chat_template(chat_template_name)
                else:
                    chat_template = get_chat_template_by_model_path(model_path)
                prompt = chat_template.image_token + prompt

            prompt_token_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_token_ids)
            output_len = fixed_output_len  # max output len, not real output len

            # video input
            if not VIDEO_ENCODING_AVAILABLE:
                raise ImportError(
                    "Video encoding function (encode_video_base64) is not available. "
                    "This is required for video benchmarks. "
                    "Please ensure sglang.utils module has encode_video_base64 function."
                )
            base64_data = encode_video_base64(video.path, video.num_frames)

            # NOTE: This will be replaced by the expanded length from the server
            prompt_len += video.num_frames

            # add to content
            content = [
                {"type": "image_url", "image_url": {"url": base64_data}},
                {"type": "text", "text": prompt},
            ]

            filtered_dataset.append([(content, prompt_len, output_len)])
            l += 1
    return filtered_dataset


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    disable_shuffle: bool = False,
) -> SampleOutput:

    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )

    if True:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

        # Download sharegpt if necessary
        if not os.path.isfile(dataset_path):
            dataset_path = download_and_cache_file(SHAREGPT_URL)

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]

        if not disable_shuffle:
            # Shuffle the dataset.
            random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: SampleOutput = []
        for data in dataset:
            i = len(input_requests)
            if i == num_prompts:
                break

            # Tokenize the prompts and completions.
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            # Skip empty prompt
            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            prompt = tokenizer.decode(input_ids)
            input_requests.append([(prompt, int(input_lens[i]), int(output_lens[i]))])
    else:
        # Sample token ids from random integers. This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            prompt = tokenizer.decode(
                [
                    (offsets[i] + i + j) % tokenizer.vocab_size
                    for j in range(input_lens[i])
                ]
            )
            input_requests.append([(prompt, int(input_lens[i]), int(output_lens[i]))])

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
    args,
    disable_shuffle: bool = False,
) -> SampleOutput:
    """Generate benchmark requests with shared system prompts using random tokens and caching."""
    cache_path = get_gen_prefix_cache_path(args, tokenizer)

    # Try to load from cache first
    if cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("\nGenerating new input data...")

    # Generate system prompts for each group
    system_prompts = []
    for _ in range(num_groups):
        system_prompt = gen_prompt(tokenizer, system_prompt_len)
        system_prompts.append(system_prompt)

    # Generate questions
    questions = []
    for _ in range(num_groups * prompts_per_group):
        question = gen_prompt(tokenizer, question_len)
        questions.append(question)

    # Combine system prompts with questions
    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0

    for group_idx in tqdm(range(num_groups), desc="Generating system prompt"):
        system_prompt = system_prompts[group_idx]
        input_requests.append([])
        for prompt_idx in tqdm(
            range(prompts_per_group), desc="Generating questions", leave=False
        ):
            question = questions[group_idx * prompts_per_group + prompt_idx]
            full_prompt = f"{system_prompt}\n\n{question}"
            prompt_len = len(tokenizer.encode(full_prompt))
            input_requests[-1].append((full_prompt, prompt_len, output_len))
            total_input_tokens += prompt_len
            total_output_tokens += output_len

    if not disable_shuffle:
        # Shuffle questions
        random.shuffle(input_requests)

    # Print statistics
    print(f"\nGenerated shared prefix dataset statistics:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Total prompts: {len(input_requests) * prompts_per_group}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(
        f"Average system prompt length: {sum(len(tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
    )
    print(
        f"Average question length: {sum(len(tokenizer.encode(q)) for q in questions) / len(questions):.1f} tokens\n"
    )

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Caching generated input data to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(input_requests, f)

    return input_requests


def get_dataset(args, tokenizer):
    if args.dataset_name == "sharegpt":
        # Support for fixed data mode
        use_fixed_data = getattr(args, 'use_fixed_data', True)
        scenario = getattr(args, 'data_scenario', 'serving')
        cache_dir = getattr(args, 'fixed_data_cache_dir', None)

        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            disable_shuffle=args.disable_shuffle,
            enable_multiturn=args.enable_multiturn,
            fixed_output_len=args.fixed_output_len,
            use_fixed_data=use_fixed_data,
            scenario=scenario,
            cache_dir=cache_dir,
        )
    elif args.dataset_name == "ultrachat":
        input_requests = sample_ultrachat_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            disable_shuffle=args.disable_shuffle,
            enable_multiturn=args.enable_multiturn,
            fixed_output_len=args.fixed_output_len,
        )
    elif args.dataset_name == "loogle":
        input_requests = sample_loogle_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            disable_shuffle=args.disable_shuffle,
            enable_multiturn=args.enable_multiturn,
            enable_shared_prefix=args.enable_shared_prefix,
            fixed_output_len=args.fixed_output_len,
        )
    elif args.dataset_name == "nextqa":
        input_requests = sample_nextqa_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            max_frames=args.max_frames,
            model_path=args.model,
            disable_shuffle=args.disable_shuffle,
            enable_multiturn=args.enable_multiturn,
            backend=args.backend,
            chat_template_name=args.chat_template,
            fixed_output_len=args.fixed_output_len,
        )
    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
        )
    elif args.dataset_name == "generated-shared-prefix":
        input_requests = sample_generated_shared_prefix_requests(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            args=args,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    return input_requests