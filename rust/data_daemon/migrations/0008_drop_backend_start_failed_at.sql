-- Drop the now-unused `backend_start_failed_at` column.
--
-- It previously marked a recording whose `/recording/start` POST was
-- permanently skipped (older than a recency bound) so that the registration
-- coordinator would mint a cloud `recording_id` on demand. That local
-- mint-on-demand path has been removed: the cloud `recording_id` now always
-- comes from `/recording/start`, and downstream coordinators simply wait for
-- it. With nothing left to skip, the column has no readers or writers.
--
-- The pending start-notify index covers the column, so drop it first, then the
-- column, then recreate the index over the surviving sweep predicate
-- (`recording_id IS NULL AND backend_start_notified_at IS NULL`).

DROP INDEX IF EXISTS idx_recordings_start_notify;

ALTER TABLE recordings DROP COLUMN backend_start_failed_at;

CREATE INDEX IF NOT EXISTS idx_recordings_start_notify
    ON recordings(recording_id, backend_start_notified_at);
