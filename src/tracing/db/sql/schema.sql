-- name: create_content_table!
CREATE TABLE IF NOT EXISTS content (
    hash  BLOB PRIMARY KEY,
    codec INTEGER NOT NULL,
    data  BLOB NOT NULL
) WITHOUT ROWID
