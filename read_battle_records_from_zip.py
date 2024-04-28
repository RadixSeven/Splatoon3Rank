import argparse
import logging
import pickle
from typing import cast, IO

from read_battle_records import battle_records_from_zip

# Configure logging
# noinspection SpellCheckingInspection
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="Read battle records from a zip file of CSVs and write to a pickle."
    )
    parser.add_argument("zip_file", help="The path to the zip file.")
    parser.add_argument(
        "output",
        help="The path to the output file (will be a "
        "gzipped pickle file e.g., foo.pkl.gz). "
        "Each record will be an entry.",
    )
    args = parser.parse_args()
    import gzip

    with gzip.open(args.output, "wb") as f:
        for record in battle_records_from_zip(args.zip_file):
            pickle.dump(record, cast(IO[bytes], f))


if __name__ == "__main__":
    main()
