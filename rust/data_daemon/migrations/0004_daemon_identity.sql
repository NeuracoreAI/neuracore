CREATE TABLE daemon_identity (
    id        INTEGER PRIMARY KEY CHECK (id = 1),
    daemon_id TEXT    NOT NULL
);

INSERT INTO daemon_identity (id, daemon_id)
SELECT 1, lower(
    hex(randomblob(4)) || '-' ||
    hex(randomblob(2)) || '-4' ||
    substr(hex(randomblob(2)), 2) || '-' ||
    substr('89ab', 1 + (abs(random()) % 4), 1) ||
    substr(hex(randomblob(2)), 2) || '-' ||
    hex(randomblob(6))
) || '-0';
