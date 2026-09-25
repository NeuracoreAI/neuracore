-- Stamped when an upload or progress request for the recording gets a plain
-- 404. The backend no longer holds it pending, so nothing more can be sent,
-- but the files are kept: the 404 does not say the recording was cancelled.
ALTER TABLE recordings ADD COLUMN backend_not_found_at DATETIME;
