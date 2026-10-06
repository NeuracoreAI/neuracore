-- A recording's start and stop are stored as two values: the caller's value
-- in microseconds on its data clock, and the wall-clock time in nanoseconds:
-- the producer's publish time, or the daemon's clock when the daemon observed
-- or received it.
ALTER TABLE recordings RENAME COLUMN start_timestamp_ns TO start_timestamp_us;
ALTER TABLE recordings RENAME COLUMN stop_timestamp_ns TO stop_timestamp_us;
ALTER TABLE recordings ADD COLUMN start_publish_timestamp_ns INTEGER;
ALTER TABLE recordings ADD COLUMN stop_publish_timestamp_ns INTEGER;

-- Rows written before this upgrade hold the caller's value in nanoseconds,
-- which is the wall clock unless the caller passed its own timestamp. Keep it
-- as the publish time and convert the caller's value to microseconds.
UPDATE recordings
   SET start_publish_timestamp_ns = start_timestamp_us,
       start_timestamp_us = start_timestamp_us / 1000,
       stop_publish_timestamp_ns = stop_timestamp_us,
       stop_timestamp_us = stop_timestamp_us / 1000;
