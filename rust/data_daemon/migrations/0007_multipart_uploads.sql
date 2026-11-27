-- Files upload as S3 multipart uploads instead of GCS resumable sessions. One
-- row per file of a trace. `active_part_url` is the presigned URL of the next
-- part to send, when it is known. `complete_url` is the presigned URL that
-- completes the upload. `completed` is set, and `active_part_url` cleared, once
-- the store has completed the upload and its CRC32C matches the file.
--
-- No foreign key, as for `traces.recording_index`: the store deletes a trace's
-- rows in the same transaction that deletes the trace or restarts its uploads.
CREATE TABLE trace_uploads (
    trace_id        TEXT    NOT NULL,
    filepath        TEXT    NOT NULL,
    upload_id       TEXT    NOT NULL,
    part_size_bytes INTEGER NOT NULL,
    active_part_url TEXT,
    complete_url    TEXT    NOT NULL,
    completed       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (trace_id, filepath)
);

-- One row per part the store accepted, with the ETag it returned. The Complete
-- request lists these parts.
CREATE TABLE trace_upload_parts (
    trace_id    TEXT    NOT NULL,
    filepath    TEXT    NOT NULL,
    part_number INTEGER NOT NULL,
    etag        TEXT    NOT NULL,
    PRIMARY KEY (trace_id, filepath, part_number)
);

-- How many times the uploader restarted this trace's uploads with new upload
-- IDs. The trace fails once the uploader's limit is reached.
ALTER TABLE traces ADD COLUMN upload_restarts INTEGER NOT NULL DEFAULT 0;
