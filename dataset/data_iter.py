"""Minimal JSONL iterable dataset used by the legacy data path.

The active pipeline lives in :mod:`dataset.dataset` and streams through
`datasets`. This module is kept for shard-level experiments where a plain,
dependency-free reader is easier to reason about.
"""

import json
from glob import glob

from torch.utils.data import IterableDataset


class DataIter(IterableDataset):
    """Iterate over sharded JSONL files, optionally packing documents together.

    Every line of a shard must be a JSON object. A ``dataset`` key holding the
    shard's dataset name is injected into each record before it reaches the
    transform, which lets a single iterator mix corpora with different schemas.

    Shards are named ``part-<dataset_name>-<index>.jsonl`` and are assigned to
    ranks round-robin, so each rank sees a disjoint subset of the data.

    Note:
        Only a single dataloader worker is supported; use ``num_workers=0``.

    Args:
        paths_with_index: ``(index, path)`` pairs, as built by
            :func:`create_shard_kwargs`.
        transform_dict: Per-dataset callables applied to each record. A transform
            may return ``None`` to drop the record, a string, or a list of token
            lists.
        max_length: Sequence length used when ``concat_docs`` is enabled.
        concat_docs: Concatenate consecutive documents and emit fixed-length
            ``max_length`` sequences instead of one example per document.
        process_index: Rank of the current process.
        num_processes: Total number of processes.
    """

    def __init__(
        self,
        paths_with_index,
        transform_dict=None,
        max_length=None,
        concat_docs=False,
        process_index=0,
        num_processes=1,
    ):
        super().__init__()
        self.paths_with_index = paths_with_index
        self.max_length = max_length
        self.transform_dict = transform_dict
        self.concat_docs = concat_docs
        self.process_index = process_index
        self.num_processes = num_processes
        if self.concat_docs:
            self.cache = []

    def __iter__(self):
        past = None
        for i, path in self.paths_with_index:
            dataset_name = path.split("-")[-2]
            if self.num_processes > 1 and i % self.num_processes != self.process_index:
                continue
            # Log once per file so progress is visible without spamming.
            if past != dataset_name:
                print(f"Loading data from {path}")
                past = path
            assert path.endswith(".jsonl"), f"Unsupported shard format: {path}"
            with open(path, encoding="utf-8") as fp:
                for line in fp:
                    # Flush a full-length sequence before reading more documents.
                    if self.concat_docs and len(self.cache) >= self.max_length:
                        seq = self.cache[: self.max_length]
                        self.cache = self.cache[self.max_length :]
                        yield seq
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    line = json.loads(line)
                    line["dataset"] = dataset_name
                    if not self.transform_dict:
                        yield line
                        continue
                    # Transformation, including sampling, tokenization, etc.
                    try:
                        line = self.transform_dict[dataset_name](line)
                    except BaseException as e:
                        print(line)
                        print("Failed key: " + str(e))
                        line = None
                    if line is None:
                        continue
                    elif isinstance(line, str):
                        yield line
                    elif isinstance(line, list) and isinstance(line[0], list):
                        for seq in line:
                            if self.concat_docs:
                                self.cache += seq
                            else:
                                yield seq
                    else:
                        raise Exception(
                            f"Unsupported type in Transformation: {self.transform_dict[dataset_name]}"
                        )


def create_shard_kwargs(patterns, repeat=1):
    """Enumerate shards so that distributed ranks never read the same file.

    Args:
        patterns: Glob patterns matching the shard files.
        repeat: Number of times the shard list is repeated, i.e. epochs.

    Returns:
        A list of ``(index, path)`` pairs consumable by :class:`DataIter`.
    """
    all_path = []
    for p in patterns:
        all_path.extend(glob(p))
    all_path *= repeat
    return [(i, p) for i, p in enumerate(all_path)]


if __name__ == "__main__":
    patterns = ["./data/pretrain_data/*.jsonl"]
    paths = create_shard_kwargs(patterns)
    transform_dict = {"wudao": lambda x: x["title"], "pile": lambda x: [x["text"]]}
    data_iter = DataIter(
        paths, transform_dict=transform_dict, max_length=16, concat_docs=True
    )
    for i, data in enumerate(data_iter):
        print(i, data)
        if i == 20:
            break
