-- Drop columns that are not meaningfully used: each is either written but
-- never read, or never written and never read. None has any production reader,
-- so removing them loses no behaviour (mirrors 0004_drop_trace_count and
-- 0008_drop_backend_start_failed_at).
--
-- recordings:
--   uploaded_trace_count  — never read or written; the upload/progress
--                           coordinators derive completion from the `traces`
--                           table directly.
--   started_at            — written on insert (daemon wall-clock open time) but
--                           never read; the producer capture window uses
--                           `start_timestamp_ns` instead.
--   robot_name            — written on insert but never read; the cloud
--   dataset_name            coordinators key every request on `robot_id` /
--                           `dataset_id`, never the human-readable names.
--
-- traces:
--   num_upload_attempts   — incremented on each retry but never read in
--                           production (only a test asserted it).
--   next_retry_at         — never written or read; backoff is in-memory in the
--                           uploader. Its dedicated index goes first.
--   robot_id              — never written and never read; vestigial
--   robot_instance          denormalisation. A trace carries `recording_index`
--   robot_name              and the daemon joins to `recordings` for identity.
--   dataset_id
--   dataset_name

DROP INDEX IF EXISTS idx_traces_next_retry_at;

ALTER TABLE recordings DROP COLUMN uploaded_trace_count;
ALTER TABLE recordings DROP COLUMN started_at;
ALTER TABLE recordings DROP COLUMN robot_name;
ALTER TABLE recordings DROP COLUMN dataset_name;

ALTER TABLE traces DROP COLUMN num_upload_attempts;
ALTER TABLE traces DROP COLUMN next_retry_at;
ALTER TABLE traces DROP COLUMN robot_id;
ALTER TABLE traces DROP COLUMN robot_instance;
ALTER TABLE traces DROP COLUMN robot_name;
ALTER TABLE traces DROP COLUMN dataset_id;
ALTER TABLE traces DROP COLUMN dataset_name;
