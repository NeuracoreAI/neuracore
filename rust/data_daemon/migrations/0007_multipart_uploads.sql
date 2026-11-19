-- Files upload as S3 multipart uploads instead of GCS resumable sessions. The
-- column holds a JSON object keyed by cloud filepath:
-- {"<filepath>": {"upload_id": "...", "part_size_bytes": 16777216, "crc32c": "..."}}.
-- `crc32c` is set once every part of the file is uploaded.
ALTER TABLE traces ADD COLUMN multipart_uploads TEXT;

-- Traces registered for a resumable session but not yet uploaded register
-- again, so they get multipart uploads. Cancelled and not-found recordings
-- upload nothing more, so their traces are left as they are.
UPDATE traces
   SET registration_status = 'pending',
       upload_status = 'pending',
       bytes_uploaded = 0
 WHERE registration_status = 'registered'
   AND upload_status != 'uploaded'
   AND recording_index NOT IN (
       SELECT recording_index FROM recordings
        WHERE cancelled_at IS NOT NULL OR backend_not_found_at IS NOT NULL);

ALTER TABLE traces DROP COLUMN upload_session_uris;
