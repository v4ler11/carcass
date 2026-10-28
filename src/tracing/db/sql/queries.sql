-- name: insert_content*!
INSERT INTO content (hash, codec, data)
VALUES (:hash, :codec, :data)
ON CONFLICT(hash) DO NOTHING

-- name: get_content^
SELECT codec, data FROM content WHERE hash = :hash
