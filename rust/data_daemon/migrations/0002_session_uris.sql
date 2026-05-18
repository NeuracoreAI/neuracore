-- Phase 6 — upload pipeline support.
--
-- Adds the JSON-encoded `upload_session_uris` column to the `traces` table so
-- the registration coordinator can persist GCS resumable session URIs alongside
-- the trace row. The uploader reads this column on `ReadyForUpload` and PUTs
-- chunks to the stored URIs. Stored as TEXT (JSON object: `{filepath: uri}`)
-- because sqlite has no native JSON column type and the column rarely needs
-- structured queries.

ALTER TABLE traces ADD COLUMN upload_session_uris TEXT;
