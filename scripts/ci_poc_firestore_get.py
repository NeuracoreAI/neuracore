#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["google-cloud-firestore>=2.19", "google-auth>=2.34"]
# ///
# cspell:ignore CLOUDSDK firestore

"""Read-only Firestore lookup helper for the CI debug PoC.

Only ever calls .get() / .stream() -- never .set()/.update()/.delete()/.add().
This is the one Firestore entry point the investigating Claude session is
allowed to invoke (see scripts/ci_poc.py's READ_ONLY_ALLOWED_TOOLS) precisely
because it can't do anything but read.

Usage:
    uv run scripts/ci_poc_firestore_get.py <collection> <doc_id> \\
        [--project ID] [--database ID]
    uv run scripts/ci_poc_firestore_get.py <collection> \\
        --where FIELD OP VALUE [--limit N] [--project ID] [--database ID]

Staging is NOT on the `(default)` Firestore database -- omitting --database
silently queries an empty database, not an error. Defaults to
`staging-firestore-us-central1`, the real staging database, so the common
case works without remembering to pass it.

Auth: if $CLOUDSDK_AUTH_ACCESS_TOKEN is set (same token `gcloud` itself uses
when that env var is set -- e.g. from `gcloud auth print-access-token` on
your own machine), it's used directly instead of ADC discovery. Falls back to
normal ADC (`gcloud auth application-default login`) if unset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from google.cloud import firestore
from google.oauth2.credentials import Credentials


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection")
    parser.add_argument("doc_id", nargs="?", help="Fetch this one document by ID")
    parser.add_argument(
        "--where",
        nargs=3,
        metavar=("FIELD", "OP", "VALUE"),
        help=(
            "Query instead of a direct doc_id lookup, "
            "e.g. --where status == registered"
        ),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--project", default="neuracore-staging", help="GCP project (default: staging)"
    )
    parser.add_argument(
        "--database",
        default="staging-firestore-us-central1",
        help=(
            "Firestore database ID -- staging is not on `(default)` "
            "(default: staging)"
        ),
    )
    args = parser.parse_args()

    if not args.doc_id and not args.where:
        parser.error("Provide either doc_id or --where")

    access_token = os.environ.get("CLOUDSDK_AUTH_ACCESS_TOKEN")
    credentials = Credentials(token=access_token) if access_token else None
    client = firestore.Client(
        project=args.project, database=args.database, credentials=credentials
    )
    collection_ref = client.collection(args.collection)

    if args.doc_id:
        doc = collection_ref.document(args.doc_id).get()
        result: Any = {
            "id": doc.id,
            "exists": doc.exists,
            "data": doc.to_dict() if doc.exists else None,
        }
    else:
        field, op, value = args.where
        query = collection_ref.where(field, op, value).limit(args.limit)
        result = [{"id": d.id, "data": d.to_dict()} for d in query.stream()]

    json.dump(result, sys.stdout, default=_json_safe, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
