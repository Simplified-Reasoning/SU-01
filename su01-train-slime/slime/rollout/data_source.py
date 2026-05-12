import abc
import copy
import json
import logging
import os
import random
from pathlib import Path

import torch

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class CombinedDataset:
    """A dataset that combines multiple datasets into a single view."""

    def __init__(self, datasets, seed=42):
        self.datasets = datasets
        self.seed = seed
        self.epoch_id = -1

        # Build combined samples list
        self.origin_samples = []
        for ds in datasets:
            for sample in ds.origin_samples:
                # Add dataset name to sample metadata for tracking
                if sample.metadata is None:
                    sample.metadata = {}
                sample.metadata["_dataset_name"] = ds.name
                self.origin_samples.append(sample)

        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Length of the data source. May change when samples are added/fetched.
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}
        
        # Replay filtering parameters (for filtering "too easy" queries)
        self.replay_filtering = getattr(args, 'replay_filtering', False)
        self.accuracy_threshold = getattr(args, 'accuracy_threshold', 1.0)
        self.accuracy_window_size = getattr(args, 'accuracy_window_size', 3)
        
        # Tracking accuracy per query_id
        self.query_accuracy = {}  # query_id -> list of recent correctness

        self.tokenizer = None

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            self.tokenizer = tokenizer
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            # Support multiple datasets via args.prompt_datasets
            prompt_datasets = getattr(args, "prompt_datasets", None)
            if prompt_datasets and len(prompt_datasets) > 0:
                # Load multiple datasets and combine them
                self.datasets = []
                for ds_config in prompt_datasets:
                    ds = Dataset(
                        ds_config["path"],
                        tokenizer=tokenizer,
                        processor=processor,
                        max_length=args.rollout_max_prompt_len,
                        prompt_key=args.input_key,
                        multimodal_keys=args.multimodal_keys,
                        label_key=args.label_key,
                        metadata_key=args.metadata_key,
                        tool_key=args.tool_key,
                        apply_chat_template=args.apply_chat_template,
                        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                        seed=args.rollout_seed,
                        name=ds_config["name"],
                    )
                    self.datasets.append(ds)
                    logger.info(f"Loaded prompt dataset '{ds_config['name']}' with {len(ds)} samples from {ds_config['path']}")

                # Create a combined dataset view
                self.dataset = CombinedDataset(self.datasets, seed=args.rollout_seed)
                logger.info(f"Combined {len(prompt_datasets)} prompt datasets with total {len(self.dataset)} samples")
            else:
                self.dataset = None
                self.datasets = []

            if self.dataset is not None and self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None
            self.datasets = []

    def get_samples(self, num_samples):
        # TODO further improve code
        samples = []
        skipped_count = 0
        
        while len(samples) < num_samples:
            # Calculate how many more samples we need
            num_needed = num_samples - len(samples)
            
            # Fetch samples from dataset
            if self.dataset is not None:
                if self.sample_offset + num_needed <= len(self.dataset):
                    prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_needed]
                    self.sample_offset += num_needed
                else:
                    prompt_samples = self.dataset.samples[self.sample_offset :]
                    remaining = num_needed - len(prompt_samples)
                    self.epoch_id += 1
                    if self.args.rollout_shuffle:
                        self.dataset.shuffle(self.epoch_id)
                    prompt_samples += self.dataset.samples[:remaining]
                    self.sample_offset = remaining
            else:
                prompt_samples = [Sample() for _ in range(num_needed)]
            
            # Filter samples if replay filtering is enabled
            for prompt_sample in prompt_samples:
                # Check if this query should be filtered out
                if self.replay_filtering and self._should_filter_query(prompt_sample, self.args.n_samples_per_prompt):
                    skipped_count += 1
                    query_acc = self._get_query_accuracy(prompt_sample)
                    logger.debug(f"Filtering out query {prompt_sample.query_id} with accuracy {query_acc:.2f}")
                    continue
                
                # Create group of samples
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = copy.deepcopy(prompt_sample)
                    sample.group_index = self.sample_group_index
                    sample.index = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                self.sample_group_index += 1
                samples.append(group)
        
        if skipped_count > 0:
            logger.info(f"Replay filtering: skipped {skipped_count} queries that were too easy")
        
        return samples

    def should_bypass_replay_filtering(self, sample: Sample) -> bool:
        """Allow subclasses to keep selected queries eligible for sampling."""
        return False

    def should_track_query_accuracy(self, sample: Sample) -> bool:
        """Allow subclasses to exclude special samples from replay-filter statistics."""
        return True
    
    def _should_filter_query(self, sample: Sample, n_samples_per_prompt: int) -> bool:
        """Check if a query should be filtered out based on its accuracy history."""
        if not self.replay_filtering:
            return False

        if self.should_bypass_replay_filtering(sample):
            return False
        
        # Use query_id to track accuracy across rollouts
        query_id = sample.query_id
        if query_id is None:
            return False
        
        # Get accuracy history from query_accuracy dict (updated after each rollout)
        if query_id in self.query_accuracy:
            accuracy_history = self.query_accuracy[query_id]
        else:
            accuracy_history = sample.accuracy_history
        
        if len(accuracy_history) < n_samples_per_prompt:
            return False
        
        # Filter if all recent rollouts were correct
        if len(accuracy_history) >= self.accuracy_window_size:
            recent_history = accuracy_history[-self.accuracy_window_size:]
            # Filter out if all recent rollouts were correct
            if all(recent_history):
                return True
        
        return False
    
    def _get_query_accuracy(self, sample: Sample) -> float:
        """Get the average accuracy for a query."""
        query_id = sample.query_id
        if query_id is None:
            return 0.0
        
        if query_id in self.query_accuracy:
            accuracy_history = self.query_accuracy[query_id]
        else:
            accuracy_history = sample.accuracy_history
        
        if len(accuracy_history) == 0:
            return 0.0
        
        return sum(accuracy_history) / len(accuracy_history)
    
    def update_query_accuracy(self, query_id: str, is_correct: bool):
        """Update the accuracy history for a query after a rollout."""
        if query_id not in self.query_accuracy:
            self.query_accuracy[query_id] = []
        
        self.query_accuracy[query_id].append(is_correct)
        
        # Keep only recent history (sliding window)
        max_history = getattr(self.args, 'accuracy_max_history', 10)
        if len(self.query_accuracy[query_id]) > max_history:
            self.query_accuracy[query_id] = self.query_accuracy[query_id][-max_history:]

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            "query_accuracy": self.query_accuracy,  # Save accuracy tracking
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        self.query_accuracy = state_dict.get("query_accuracy", {})  # Load accuracy tracking

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def __len__(self) -> int:
        return len(self.dataset)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


class SelfRefineDataSource(RolloutDataSourceWithBuffer):
    """Data source that extends RolloutDataSourceWithBuffer with self-refine capability.

    After each rollout, wrong responses from unsolved query groups are collected
    and transformed into self-refine prompts (original question + previous wrong
    answer + refine instruction). These are stored in a separate buffer and mixed
    into subsequent batches at a configurable ratio. Displaced normal samples are
    pushed back to the normal buffer so no training data is ever lost.

    Enable by setting --data-source-path slime.rollout.data_source.SelfRefineDataSource
    """

    def __init__(self, args):
        super().__init__(args)
        self.self_refine_ratio = getattr(args, 'self_refine_ratio', 0.2)
        self.self_refine_buffer = {}  # query_id -> list[Sample]

        self._refine_template = self._load_refine_template(args)

        logger.info(
            f"SelfRefineDataSource enabled: ratio={self.self_refine_ratio}, "
            f"group_reward_threshold={getattr(args, 'self_refine_group_reward_threshold', 0.5)}"
        )

    @staticmethod
    def _load_refine_template(args) -> str:
        """Load the self-refine prompt template, preferring --self-refine-prompt-file if set."""
        prompt_file = getattr(args, 'self_refine_prompt_file', None)
        if prompt_file is not None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_refine_prompt_module", prompt_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            template = getattr(mod, 'SELF_REFINEMENT_PROMPT', None)
            if template is None:
                raise ValueError(
                    f"--self-refine-prompt-file {prompt_file} does not define SELF_REFINEMENT_PROMPT"
                )
            logger.info(f"Loaded self-refine prompt template from {prompt_file} "
                        f"(len={len(template)}, has {{original_content}}: "
                        f"{'original_content' in template})")
            return template
        return getattr(
            args,
            'self_refine_prompt_template',
            "Your previous attempt at this problem produced the following answer:\n{previous_answer}\n\n"
            "This answer is incorrect. Please carefully re-examine the problem and provide a corrected solution.",
        )

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Fetch normal samples then mix in self-refine samples at the configured ratio."""
        samples = super().get_samples(num_samples)

        if len(self.self_refine_buffer) == 0:
            return samples

        num_refine = min(
            int(num_samples * self.self_refine_ratio),
            len(self.self_refine_buffer),
        )
        if num_refine <= 0:
            return samples

        refine_groups = self._pop_self_refine_samples(num_refine)

        # Displace normal samples back into the normal buffer so they are not lost
        displaced = samples[-num_refine:]
        samples = samples[:-num_refine]
        for group in displaced:
            self.buffer.append(group)

        samples += refine_groups
        logger.info(
            f"Self-refine: mixed {len(refine_groups)} refine groups into batch, "
            f"displaced {len(displaced)} normal groups back to buffer, "
            f"remaining refine buffer size={len(self.self_refine_buffer)}"
        )

        # Print one refine sample for manual inspection
        if refine_groups:
            example = refine_groups[0][0]
            prev_ans = (example.metadata or {}).get('_previous_answer', '')
            payload = {
                "query_id": example.query_id,
                "prompt": example.prompt,
                "previous_wrong_answer": prev_ans,
                "label": str(example.label),
            }
            # Keep as a single line JSON for easier inspection (avoid multiline log blocks).
            logger.info(
                "DEBUG: Self-refine sample preview " +
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )

        return samples

    def _pop_self_refine_samples(self, num_refine: int) -> list[list[Sample]]:
        """Pop self-refine samples from the buffer and wrap them as groups."""
        refine_groups = []
        query_ids = list(self.self_refine_buffer.keys())
        random.shuffle(query_ids)

        for query_id in query_ids[:num_refine]:
            candidates = self.self_refine_buffer[query_id]
            chosen = random.choice(candidates)
            del self.self_refine_buffer[query_id]

            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(chosen)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            refine_groups.append(group)

        return refine_groups

    def add_self_refine_samples(self, failed_samples: list[Sample]):
        """Create self-refine prompts from failed samples and add to the refine buffer.

        For each failed sample, extracts the answer after </think>, constructs a new
        prompt containing the original question + refine instruction + previous wrong
        answer, and stores it in self_refine_buffer.

        The buffer is cleared each rollout so only the most recent rollout's
        wrong answers are available for the next batch.
        """
        # Clear old buffer: only keep failures from the current rollout
        if self.self_refine_buffer:
            logger.info(
                f"Self-refine: clearing previous buffer ({len(self.self_refine_buffer)} queries)"
            )
            self.self_refine_buffer.clear()

        if not failed_samples:
            return

        template = self._refine_template
        template_has_original_content = '{original_content}' in template

        added_count = 0
        for sample in failed_samples:
            original_content = _strip_special_tokens(sample.metadata.get('_original_content', ''))
            if not original_content:
                logger.debug(f"Skipping self-refine for {sample.query_id}: no _original_content in metadata")
                continue

            previous_answer = _strip_special_tokens(_extract_answer_after_think(sample.response))
            if not previous_answer.strip():
                continue

            if template_has_original_content:
                new_content = template.format(
                    original_content=original_content,
                    previous_answer=previous_answer,
                )
            else:
                refine_instruction = template.format(previous_answer=previous_answer)
                new_content = f"{original_content}\n\n{refine_instruction}"

            messages = [{"role": "user", "content": new_content}]
            if self.tokenizer is not None:
                new_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **(getattr(self.args, 'apply_chat_template_kwargs', None) or {}),
                )
            else:
                new_prompt = new_content

            refine_sample = Sample(
                prompt=new_prompt,
                label=sample.label,
                query_id=f"{sample.query_id}:refine",
                metadata={
                    **sample.metadata,
                    '_is_self_refine': True,
                    '_original_query_id': sample.query_id,
                    '_previous_answer': previous_answer,
                    # Pass the original question to the RM so it grades based on the
                    # real problem statement, not the refine prompt with prior answer.
                    'question': original_content,
                },
            )

            original_qid = sample.query_id
            if original_qid not in self.self_refine_buffer:
                self.self_refine_buffer[original_qid] = []
            self.self_refine_buffer[original_qid].append(refine_sample)
            added_count += 1

        if added_count > 0:
            logger.info(
                f"Self-refine: added {added_count} samples to refine buffer, "
                f"total queries in buffer={len(self.self_refine_buffer)}"
            )

    def get_self_refine_buffer_length(self):
        return len(self.self_refine_buffer)

    def save(self, rollout_id):
        super().save(rollout_id)
        if self.args.rollout_global_dataset:
            refine_state = {
                query_id: [s.to_dict() for s in samples]
                for query_id, samples in self.self_refine_buffer.items()
            }
            path = os.path.join(self.args.save, f"rollout/self_refine_buffer_{rollout_id}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(refine_state, path)

    def load(self, rollout_id=None):
        super().load(rollout_id)
        if self.args.rollout_global_dataset and self.args.load is not None:
            path = os.path.join(self.args.load, f"rollout/self_refine_buffer_{rollout_id}.pt")
            if os.path.exists(path):
                logger.info(f"Loading self-refine buffer from {path}")
                refine_state = torch.load(path, weights_only=False)
                self.self_refine_buffer = {
                    query_id: [Sample.from_dict(s) for s in samples]
                    for query_id, samples in refine_state.items()
                }
                logger.info(f"Loaded self-refine buffer with {len(self.self_refine_buffer)} queries")


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


_SPECIAL_TOKENS_TO_STRIP = ('<|im_start|>', '<|im_end|>', '<|endoftext|>')

def _strip_special_tokens(text: str) -> str:
    """Remove specific chat-template special tokens that may leak into content."""
    for tok in _SPECIAL_TOKENS_TO_STRIP:
        text = text.replace(tok, '')
    return text.strip()


def _extract_answer_after_think(response: str) -> str:
    """Extract the answer part after </think> from a model response.

    If no </think> marker is found, returns the entire response.
    """
    marker = "</think>"
    idx = response.find(marker)
    if idx >= 0:
        return response[idx + len(marker):].strip()
    else:
        return "I failed to reason the answer, please try again."
