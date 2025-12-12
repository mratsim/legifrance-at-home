# legifrance-at-home
# Copyright (c) 2025-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

from pathlib import Path
from datetime import timedelta
import argparse, time

import lancedb
from lancedb.rerankers import CrossEncoderReranker

# =============================================================================
# Script parameters
# =============================================================================
DB_NAME="legifrance_db"
TABLE_NAME="legi"
QUERIESDIR="queries"
QUERY_FILE_PREFIX="search_legifrance"
START_TIME = time.strftime("%Y-%m-%d_%H%M")

DEVICE = "cpu" # or cuda. You might want to skip the reranker if you don't have a GPU with "BAAI/bge-reranker-v2-m3"
# RERANKER_MODEL=CrossEncoderReranker(model_name = "BAAI/bge-reranker-v2-m3", device=DEVICE)
RERANKER_MODEL=CrossEncoderReranker(model_name = "jinaai/jina-reranker-v2-base-multilingual", device=DEVICE)

# =============================================================================
# DB Queries + csv output
# =============================================================================

def timedelta_fmt(duration: timedelta) -> str:
    data = {}
    data['days'], remaining = divmod(duration.total_seconds(), 86_400)
    data['hours'], remaining = divmod(remaining, 3_600)
    data['minutes'], data['seconds'] = divmod(remaining, 60)

    time_parts = [f'{round(value)} {name}' for name, value in data.items() if value > 0]
    return ' '.join(time_parts)

def monotonic_ns_fmt(duration: int) -> str:
    return timedelta_fmt(timedelta(microseconds=duration//1000))

def main(num_results: int, query: str):
    db = lancedb.connect(uri=DB_NAME)
    tbl = db.open_table(TABLE_NAME)


    print(f"Query: {query}")
    time_start = time.monotonic_ns()
    results = (
        tbl.search(
            query,
            query_type="hybrid",
            vector_column_name="vector",
            fts_columns="text",
        )
        .rerank(RERANKER_MODEL)
        .limit(num_results)
        .to_pandas()
    )
    time_stop = time.monotonic_ns()
    query_time = time_stop - time_start
    print(f"Query time: {monotonic_ns_fmt(query_time)}")

    print(results)

    queriesdir = Path(QUERIESDIR)
    queriesdir.mkdir(exist_ok=True, parents=True)
    outfile = queriesdir / (QUERY_FILE_PREFIX + '-' + START_TIME + '-' + query + '.csv')
    results.to_csv(outfile, sep=';')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search the Legifrance DB."
    )
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        help="Query string to search for.",
    )
    parser.add_argument(
        "-n",
        "--num-results",
        type=int,
        default=10,
        help="Number of results to return (default: 10).",
    )

    args = parser.parse_args()
    main(args.num_results, args.query)