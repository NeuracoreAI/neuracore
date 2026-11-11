-- Recordings started with a robot or dataset made offline, which has only a
-- local id until the daemon finds or creates it on the backend.
--
-- A recording's robot_id and dataset_id keep the local ids for good: robot_id
-- is half of the source key the windows and producers use. The backend's ids
-- are kept apart, and the rest of the pipeline only needs recordings.offline
-- to know whether a recording may start on the backend yet.
--
-- The daemon's tables carry no foreign keys (see 0001): these rows are linked
-- by key values, and delete_recording_cascade removes a recording's
-- recording_offline row with it.

-- One row per instance of a robot connected offline that a recording has
-- used, keyed by the recording's robot_id and robot_instance.
CREATE TABLE local_robots (
    robot_id         TEXT     NOT NULL,  -- local id, as the SDK works it out
    instance         INTEGER  NOT NULL,
    name             TEXT     NOT NULL,
    shared           INTEGER  NOT NULL,
    overwrite_urdf   INTEGER  NOT NULL,  -- connect_robot's overwrite_existing_urdf
    cloud_robot_id   TEXT,               -- set once registered
    status           TEXT     NOT NULL DEFAULT 'pending',  -- pending | done | failed
    detail           TEXT,               -- why it is stuck or failed: robot_archived, credentials, ...
    next_attempt_at  DATETIME,           -- not asked about again before this; NULL = now
    PRIMARY KEY (robot_id, instance)
);

-- One row per dataset selected offline that a recording has used, keyed by
-- the recording's dataset_id.
CREATE TABLE local_datasets (
    dataset_id       TEXT     PRIMARY KEY,  -- local id
    name             TEXT     NOT NULL,
    shared           INTEGER  NOT NULL,
    description      TEXT,
    tags             TEXT     NOT NULL DEFAULT '[]',  -- JSON array
    cloud_dataset_id TEXT,                  -- set once found or created
    status           TEXT     NOT NULL DEFAULT 'pending',  -- pending | done | failed
    detail           TEXT,                  -- name_taken, not_in_org, shared_not_allowed, ...
    next_attempt_at  DATETIME
);

-- The backend's ids for one offline recording once it has them, or why it
-- failed. Whether its robot_id or dataset_id is a local one is whether
-- local_robots or local_datasets has a row for it, which it keeps while the
-- recording is resolving.
CREATE TABLE recording_offline (
    recording_index  INTEGER  PRIMARY KEY,
    cloud_robot_id   TEXT,               -- copied from local_robots when resolved
    cloud_dataset_id TEXT,               -- copied from local_datasets when resolved
    failure          TEXT,               -- why it failed: the refused robot's or dataset's detail
    failure_kind     TEXT,               -- what was refused: 'robot' or 'dataset'
    failure_name     TEXT                -- the refused robot's or dataset's name
);

-- NULL for a recording started online, which never needs resolving.
--   resolving  waiting for its robot and/or dataset
--   resolved   has every backend id; the normal pipeline takes it from here
--   failed     a robot or dataset was refused for good; never uploads
-- Rows written before this upgrade were all started online.
ALTER TABLE recordings ADD COLUMN offline TEXT;
