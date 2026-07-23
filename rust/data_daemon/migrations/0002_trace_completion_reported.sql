-- Backend-ack watermark for trace completions. Stamped by the status
-- coordinator once the batch PUT carrying this trace's UPLOAD_COMPLETE is
-- acknowledged; the recording reaper refuses to reclaim a recording while
-- any of its traces is still unstamped, so a queued-but-unflushed completion
-- can no longer be destroyed by the sweep. No dedicated index: lookups are
-- always per-recording via `idx_traces_recording_upload`'s leading column.
ALTER TABLE traces ADD COLUMN completion_reported_at DATETIME;

-- How many times the status coordinator's reconcile pass has re-driven this
-- trace's completion PUT. Reclaim opens once the attempts are exhausted so a
-- permanently-rejected completion cannot pin the recording's disk forever.
ALTER TABLE traces ADD COLUMN completion_report_attempts INTEGER NOT NULL DEFAULT 0;

-- Traces that finished uploading before this upgrade already reported their
-- completion under the old semantics; stamping them keeps recordings that
-- straddle the upgrade reclaimable.
UPDATE traces SET completion_reported_at = last_updated WHERE upload_status = 'uploaded';
