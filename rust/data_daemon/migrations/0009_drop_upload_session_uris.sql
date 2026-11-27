-- Files upload as multipart uploads, so the resumable session URIs are unused.
ALTER TABLE traces DROP COLUMN upload_session_uris;
