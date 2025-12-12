# legifrance-at-home
# Copyright (c) 2025-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import json, time
from datetime import timedelta, datetime
from pathlib import Path

import lancedb
from lancedb.embeddings import get_registry
from lancedb. pydantic import LanceModel, Vector

from datasets import load_dataset

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status
from rich.progress import Progress, TaskID, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.live import Live

from typing import List, Optional, TextIO, Type
from types import TracebackType

# =============================================================================
# Script parameters
# =============================================================================

LOGDIR='logs'
LOG_FILE_PREFIX = 'import_legifrance'
START_TIME = time.strftime("%Y-%m-%d_%H%M")

DEVICE = "cpu" # or cuda
MAX_BATCH_SIZE=10000

DATASET="AgentPublic/legi"

DB_NAME="legifrance_db"
TABLE_NAME="legi"

# =============================================================================
# 1. UI
# =============================================================================

def ceil_div(num: int, den: int) -> int:
    return (num + den - 1) // den

def timedelta_fmt(duration: timedelta) -> str:
    data = {}
    data['days'], remaining = divmod(duration.total_seconds(), 86_400)
    data['hours'], remaining = divmod(remaining, 3_600)
    data['minutes'], data['seconds'] = divmod(remaining, 60)

    time_parts = [f'{round(value)} {name}' for name, value in data.items() if value > 0]
    return ' '.join(time_parts)

def monotonic_ns_fmt(duration: int) -> str:
    return timedelta_fmt(timedelta(microseconds=duration//1000))

class ProgressTracker:
    completed_batches: int
    completed_rows: int
    display: Live
    table: Table
    progress: Progress
    progress_batches: TaskID
    progress_rows: TaskID
    status: Status
    log_file: TextIO

    def __init__(self, num_rows: int, max_batch_size: int):
        self.completed_batches = 0
        self.completed_rows = 0

        self.progress = self.progress = Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(), expand=True)
        self.progress_rows = self.progress.add_task("[green]Rows ...", total=num_rows)
        self.progress_batches = self.progress.add_task("[green]Batches ...", total=ceil_div(num_rows, max_batch_size))

        self.status = Status('Initializing ...')

        logdir = Path(LOGDIR)
        logdir.mkdir(exist_ok=True, parents=True)
        logfile = logdir / (LOG_FILE_PREFIX + '-' + START_TIME + '.log')
        self.log_file = open(logfile, 'w', buffering=1) # Use line buffering

        self.table = Table.grid()
        self.table.add_row(Panel.fit(self.status))
        self.table.add_row(Panel.fit(self.progress))
        self.display = Live(self.table)

    def batch_completed(self, batch_size: int) -> None:
        self.completed_batches += 1
        self.completed_rows += batch_size
        self.progress.update(self.progress_batches, advance=1)
        self.progress.update(self.progress_rows, advance=batch_size)

    def log(self, line: str) -> None:
        line = f'[{datetime.now()}] {line}'
        self.log_file.write(line + '\n')
        print(line)

    def __enter__(self) -> 'ProgressTracker' :
        self.display.start()
        return self

    def __exit__(self,
                exc_type: Optional[Type[BaseException]],
                exc_val: Optional[BaseException],
                exc_tb: Optional[TracebackType]) -> None:
        self.display.stop()
        self.log_file.close()

# =============================================================================
# 2. Logic
# =============================================================================

# Get the BGE-M3 embedding function from the registry
# BGE-M3 is available via sentence-transformers
bge_m3 = get_registry().get("sentence-transformers").create(
    name="BAAI/bge-m3",
    device=DEVICE
)

# Define the schema with the embedding function
class LegiArticle(LanceModel):
    doc_id: str
    chunk_index:  int
    nature:  str
    category:  str
    ministry:  Optional[str]
    status: str
    title: str
    full_title: str
    start_date: str
    end_date:  str
    links:  str  # Store as JSON string instead of List[dict]
    text:  str = bge_m3.SourceField()  # This field will be used for embedding queries
    vector: Vector(bge_m3.ndims()) = bge_m3.VectorField()  # BGE-M3 is 1024 dims

columns_to_keep = [
    "doc_id", "chunk_index", "nature", "category", "ministry",
    "status", "title", "full_title", "start_date", "end_date",
    "links", "text", "embeddings_bge-m3",
]

def process_batch(items):
    """Process a batch of items, parsing pre-computed embeddings."""
    records = []
    for item in items:
        record = {k: item[k] for k in columns_to_keep if k != "embeddings_bge-m3"}
        # Use pre-computed embeddings (parse from JSON string)
        record["vector"] = json.loads(item["embeddings_bge-m3"])
        record["links"] = json.dumps(item["links"])  # Convert list[dict] to JSON string
        records.append(record)
    return records

def main():
    time_start = time.monotonic_ns()
    dataset = load_dataset(DATASET, split="train")
    time_stop = time.monotonic_ns()
    load_time = time_stop - time_start
    print(f"Dataset load time: {monotonic_ns_fmt(load_time)}\n")

    print(f"Dataset of size {dataset.num_rows} x {dataset.num_columns} with the following features:")
    print(dataset.features)
    print("\n\n")

    print(f"Creating DB/Table: {DB_NAME}/{TABLE_NAME}")
    db = lancedb.connect(uri=DB_NAME)
    table = db.create_table(TABLE_NAME, schema=LegiArticle, mode="overwrite")

    with ProgressTracker(dataset.num_rows, MAX_BATCH_SIZE) as ctx:
        time_start = time.monotonic_ns()
        ctx.log(f'INFO - Starting import of {dataset.num_rows} rows from {DATASET} into {DB_NAME}/{TABLE_NAME}')
        ctx.status.update('Importing ...')
        for i in range(0, dataset.num_rows, MAX_BATCH_SIZE):
            batch = dataset.select(range(i, min(i + MAX_BATCH_SIZE, dataset.num_rows)))
            records = process_batch(batch)
            table.add(records)
            ctx.batch_completed(len(records))
        ctx.log(f'INFO - Finished import of {dataset.num_rows} rows from {DATASET} into {DB_NAME}/{TABLE_NAME}')
        time_stop = time.monotonic_ns()
        import_time = time_stop - time_start
        ctx.log(f"Dataset import time: {monotonic_ns_fmt(import_time)}\n")


        time_start = time.monotonic_ns()
        ctx.log(f'INFO - Indexing for Full-Text Search')
        ctx.status.update('Indexing ...')
        table.create_fts_index(
            "text",
            language="French",
            stem=True,
            ascii_folding=True,
            replace=True,
        )
        table.wait_for_index(["text_idx"])
        time_stop = time.monotonic_ns()
        index_time = time_stop - time_start
        ctx.log(f"Dataset indexing time: {monotonic_ns_fmt(index_time)}\n")

        ctx.status.update('Finished!')
        ctx.log(f'INFO - Finished!')

# =============================================================================
# 3. Calling the code
# =============================================================================

if __name__ == "__main__":
    main()