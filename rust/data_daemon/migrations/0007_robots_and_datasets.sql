-- The robots and datasets this daemon records with, decoupled from recordings.
--
-- Every robot instance a process connects, and every dataset it selects, gets a
-- row here, online or offline. The row's index is what the rest of the daemon
-- uses; the server's id is filled in once the server confirms the row, and only
-- the code that talks to the API reads it. Recordings point at their robot and
-- dataset rows by index.

CREATE TABLE IF NOT EXISTS robots (
    robot_index     INTEGER PRIMARY KEY AUTOINCREMENT,
    -- The organization the robot is connected in. NULL only on rows made for
    -- recordings that predate this table: those use the selected organization.
    org_id          TEXT,
    -- NULL only on rows made for recordings that predate this table.
    name            TEXT,
    shared          INTEGER NOT NULL DEFAULT 0,
    instance        INTEGER NOT NULL,
    overwrite_urdf  INTEGER NOT NULL DEFAULT 0,
    -- The URDF package the SDK saved for the server, deleted once sent.
    package_path    TEXT,
    cloud_robot_id  TEXT,
    -- pending: the server has yet to confirm the row; done: confirmed, with
    -- cloud_robot_id; failed: refused for good.
    status          TEXT    NOT NULL,
    -- Why a pending row is held, or why the row failed.
    reason          TEXT,
    -- When the server may next be asked about a pending row. NULL: now.
    next_attempt_at DATETIME,
    last_used_at    DATETIME NOT NULL,
    created_at      DATETIME NOT NULL,
    UNIQUE (org_id, name, shared, instance)
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_index    INTEGER PRIMARY KEY AUTOINCREMENT,
    -- NULL only on rows made for recordings that predate this table.
    org_id           TEXT,
    -- NULL on rows made for recordings that predate this table, or for a
    -- recording started from the web app.
    name             TEXT,
    shared           INTEGER NOT NULL DEFAULT 0,
    description      TEXT,
    -- Tag names, as a JSON array of strings.
    tags             TEXT    NOT NULL DEFAULT '[]',
    cloud_dataset_id TEXT,
    status           TEXT    NOT NULL,
    reason           TEXT,
    next_attempt_at  DATETIME,
    last_used_at     DATETIME NOT NULL,
    created_at       DATETIME NOT NULL,
    UNIQUE (org_id, name, shared)
);

ALTER TABLE recordings ADD COLUMN robot_index INTEGER REFERENCES robots (robot_index);
ALTER TABLE recordings ADD COLUMN dataset_index INTEGER REFERENCES datasets (dataset_index);
-- A robot's logged data is labelled with its row; the daemon finds a robot's
-- recordings by it.
CREATE INDEX IF NOT EXISTS idx_recordings_robot_row ON recordings (robot_index, robot_instance);
-- When the start notifier first sent the recording's start to the server.
-- Until then no start of it can be in flight or already on the server, so it
-- is safe to send on any re-check.
ALTER TABLE recordings ADD COLUMN start_sent_at DATETIME;

-- Recordings from before this table hold the server's ids. Give them rows that
-- are already confirmed, in no organization: they keep using the selected one.
INSERT INTO robots (org_id, name, shared, instance, cloud_robot_id, status, last_used_at, created_at)
SELECT NULL, NULL, 0, instance, robot_id, 'done', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
  FROM (SELECT DISTINCT robot_id, COALESCE(robot_instance, 0) AS instance
          FROM recordings
         WHERE robot_id IS NOT NULL);

UPDATE recordings
   SET robot_index = (SELECT r.robot_index FROM robots r
                       WHERE r.org_id IS NULL
                         AND r.cloud_robot_id = recordings.robot_id
                         AND r.instance = COALESCE(recordings.robot_instance, 0))
 WHERE robot_id IS NOT NULL;

INSERT INTO datasets (org_id, name, shared, cloud_dataset_id, status, last_used_at, created_at)
SELECT NULL, NULL, 0, dataset_id, 'done', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
  FROM (SELECT DISTINCT dataset_id FROM recordings WHERE dataset_id IS NOT NULL);

UPDATE recordings
   SET dataset_index = (SELECT d.dataset_index FROM datasets d
                         WHERE d.org_id IS NULL
                           AND d.cloud_dataset_id = recordings.dataset_id)
 WHERE dataset_id IS NOT NULL;
