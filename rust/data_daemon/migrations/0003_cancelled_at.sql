-- Phase 7 — recording cancellation.
--
-- Adds the `cancelled_at` timestamp to the recordings table so the daemon can
-- flag a recording the producer asked us to discard. Cancelled recordings are
-- skipped by the registration / upload / progress coordinators, and the
-- per-trace actors that belong to a cancelled recording delete their on-disk
-- artefacts before exiting.

ALTER TABLE recordings ADD COLUMN cancelled_at DATETIME;

CREATE INDEX IF NOT EXISTS idx_recordings_cancelled_at
    ON recordings(cancelled_at);
