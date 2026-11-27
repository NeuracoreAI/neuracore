-- When the recording reaper first saw that a recording can never upload: its
-- robot or dataset was refused, the backend no longer holds it, or a trace
-- failed to write, register or upload. The reaper deletes the recording a fixed time
-- later, and the status the user sees says when. NULL until then.
ALTER TABLE recordings ADD COLUMN failed_at DATETIME;
