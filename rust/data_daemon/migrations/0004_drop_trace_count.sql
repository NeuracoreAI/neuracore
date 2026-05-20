-- Drop the denormalised `recordings.trace_count` column.
--
-- It was maintained on every `create_trace` via a `SELECT COUNT(*)` subquery
-- — O(N) per insert, O(N²) over a recording — yet nothing in the daemon ever
-- read it (the upload, registration, progress and status coordinators all
-- derive trace state directly from the `traces` table). Removing the column
-- removes the per-insert recount entirely.

ALTER TABLE recordings DROP COLUMN trace_count;
