import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Self


class DBManager:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: sqlite3.Connection | None = None

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one transaction: commit on success, roll back on error."""
        assert self._db is not None
        try:
            yield self._db
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def __enter__(self) -> Self:
        self._db = sqlite3.connect(self.db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.commit()
        self._init_schema()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def _init_schema(self) -> None:
        raise NotImplementedError

    @property
    def conn(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("DBManager not entered - use `with`")
        return self._db
