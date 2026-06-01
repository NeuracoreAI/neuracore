-- Drop the now-unused `org_id` column from `recordings`.
--
-- The organisation is no longer stamped onto each recording row at creation
-- time (which froze whatever value the daemon happened to read from config at
-- launch). Instead the daemon resolves the *current* org live from
-- `~/.neuracore/config.json` at the moment each cloud POST is made, via a
-- config-file watcher feeding a shared value into every cloud coordinator.
-- This removes the failure mode where a recording created before the org was
-- known stayed `org_id`-less forever and its traces looped back to pending.
--
-- No index references `org_id`, so the column drops cleanly.

ALTER TABLE recordings DROP COLUMN org_id;
