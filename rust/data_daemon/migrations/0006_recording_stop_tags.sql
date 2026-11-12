-- User tag names for the recording, as a JSON array of strings. Given with the
-- stop and kept until the backend stop notification is sent, so a restart or an
-- offline period does not lose them. NULL means the stop carried no tags.
ALTER TABLE recordings ADD COLUMN stop_tags TEXT;
