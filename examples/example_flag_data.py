"""This example demonstrates how you can programmatically add flags to a datasets
and recordings.

See the `example_data_collection_bigym.py` example for how to create a dataset first.
"""

import argparse

from tqdm import tqdm

import neuracore as nc
from neuracore.core.data.recording import Recording, RecordingStatus


def is_bad_recording_heuristic(recording: Recording) -> bool:
    # Add your custom logic here e.g.
    # recording.start_time > datetime.now() - datetime.timedelta(days=1)
    return True


def main(
    dataset_name: str | None = None,
    dataset_id: str | None = None,
    update_dataset: bool = True,
):
    nc.login()

    # Create a dataset
    dataset = nc.get_dataset(name=dataset_name, id=dataset_id)
    print(f"Processing dataset: {dataset.name} ({dataset.id})")
    if update_dataset:
        print("Updating dataset metadata ...", end="")
        dataset.set_name(f"{dataset.name} (Flagged)")
        dataset.add_tag("Processed")
        dataset.set_description(
            (dataset.description or "") + "\n\n --- \nThis dataset has been flagged."
        )
        print(" [Done]")

    print("Updating dataset Recordings ...", end=None)
    for recording in tqdm(dataset):
        if is_bad_recording_heuristic(recording):
            recording.set_status(RecordingStatus.FLAGGED)
            recording.set_notes("Bad recording")
        else:
            recording.set_status(RecordingStatus.NORMAL)
            recording.set_notes("Good recording")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset_id_group = parser.add_mutually_exclusive_group(required=True)
    dataset_id_group.add_argument(
        "--dataset-name",
        type=str,
        help="The name of the dataset to process",
    )
    dataset_id_group.add_argument(
        "--dataset-id",
        type=str,
        help="The ID of the dataset to process",
    )

    parser.add_argument(
        "--update-dataset",
        action="store_true",
        default=False,
        help="Update the dataset metadata",
    )

    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        dataset_id=args.dataset_id,
        update_dataset=args.update_dataset,
    )
