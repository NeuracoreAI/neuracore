-- Track whether `POST /recording/stop` to the backend has been delivered.
--
-- The recording-stop notifier sets `backend_stop_notified_at` once the POST
-- succeeds. On daemon start the notifier sweeps any recordings with
-- `stopped_at NOT NULL` but no `backend_stop_notified_at`, so a recording
-- stopped while the daemon was offline gets its backend notification
-- re-issued once the daemon comes online.
--
-- Migration is additive only — existing recording rows have NULL for the new
-- column, which matches "not yet notified" semantics.

ALTER TABLE recordings ADD COLUMN backend_stop_notified_at DATETIME;
