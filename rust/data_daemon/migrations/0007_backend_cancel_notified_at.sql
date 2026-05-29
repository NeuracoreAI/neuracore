-- Track whether `POST /recording/cancel` to the backend has been delivered.
--
-- The recording-cancel notifier sets `backend_cancel_notified_at` once the
-- POST succeeds. On daemon start the notifier sweeps any recordings that were
-- cancelled while the daemon was offline: rows with `cancelled_at NOT NULL`,
-- a non-NULL cloud `recording_id`, and no `backend_cancel_notified_at`.
--
-- Migration is additive only — existing rows default to NULL.

ALTER TABLE recordings ADD COLUMN backend_cancel_notified_at DATETIME;
