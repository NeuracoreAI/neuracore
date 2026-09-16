-- A recording's start and stop are stored as two values: the caller's tick
-- (one microsecond) on its data clock, posted as the backend
-- `start_timestamp` / `end_timestamp`, and the producer's wall-clock publish
-- time in nanoseconds, posted as the backend `start_time` / `end_time`.
-- `ticks_per_second` is set on every row the daemon creates from now on; a
-- NULL marks a recording whose traces hold float seconds.
ALTER TABLE recordings RENAME COLUMN start_timestamp_ns TO start_timestamp;
ALTER TABLE recordings RENAME COLUMN stop_timestamp_ns TO stop_timestamp;
ALTER TABLE recordings ADD COLUMN start_publish_timestamp_ns INTEGER;
ALTER TABLE recordings ADD COLUMN stop_publish_timestamp_ns INTEGER;
ALTER TABLE recordings ADD COLUMN ticks_per_second INTEGER;

-- Rows written before this upgrade hold nanoseconds, which were the wall clock
-- unless the caller passed its own timestamp. Keep them as the publish time and
-- convert the caller's value to ticks for the daemon's own bookkeeping. Their
-- `ticks_per_second` stays NULL, so the backend is not told the traces are in
-- ticks and converts them from seconds when it saves the recording.
UPDATE recordings
   SET start_publish_timestamp_ns = start_timestamp,
       start_timestamp = start_timestamp / 1000,
       stop_publish_timestamp_ns = stop_timestamp,
       stop_timestamp = stop_timestamp / 1000;
