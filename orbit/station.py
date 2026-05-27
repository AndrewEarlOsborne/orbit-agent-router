"""Storage backends for Orbit payloads"""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import json
import logging

logger = logging.getLogger(__name__)


class Station(ABC):
    """Abstract base class for payload storage backends"""

    @abstractmethod
    async def store_payload(self, tool_call_id: str, payload: Any) -> None:
        """
        Store full payload with associated tool_call_id

        Args:
            tool_call_id: Unique identifier for the tool call
            payload: Full payload to store
        """
        pass

    @abstractmethod
    async def get_payload(self, tool_call_id: str) -> Any | None:
        """
        Retrieve payload by tool_call_id

        Args:
            tool_call_id: Unique identifier for the tool call

        Returns:
            Stored payload if found, None otherwise
        """
        pass


class StationCache(Station):
    """In-memory cache-based storage for development and testing"""

    def __init__(self, cache: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize with optional existing cache

        Args:
            cache: Optional dictionary to use as cache backend.
                   If None, creates new empty dict.
        """
        self._cache: Dict[str, Any] = cache if cache else {}
        logger.info("StationCache initialized with %d existing entries", len(self._cache))

    async def store_payload(self, tool_call_id: str, payload: Any) -> None:
        """
        Store payload in in-memory cache

        Args:
            tool_call_id: Unique identifier for the tool call
            payload: Full payload to store
        """
        self._cache[tool_call_id] = payload
        logger.debug("Stored payload for tool_call_id: %s", tool_call_id)

    async def get_payload(self, tool_call_id: str) -> Optional[Any]:
        """
        Retrieve payload from cache

        Args:
            tool_call_id: Unique identifier for the tool call

        Returns:
            Stored payload if found, None otherwise
        """
        payload = self._cache.get(tool_call_id)
        if payload is None:
            logger.debug("No payload found for tool_call_id: %s", tool_call_id)
        else:
            logger.debug("Retrieved payload for tool_call_id: %s", tool_call_id)
        return payload

    def clear(self) -> None:
        """Clear all cached payloads"""
        self._cache.clear()
        logger.info("StationCache cleared")


class StationDB(Station):
    """Database-backed storage for production use"""

    def __init__(
        self, connection_string: str | None = None, table_name: str = "orbit_payloads"
    ) -> None:
        """
        Initialize database connection

        Args:
            connection_string: Database connection string. If None, uses default.
            table_name: Name of table for storing payloads
        """
        self.connection_string = connection_string or "sqlite:///orbit_payloads.db"
        self.table_name = table_name
        self._db: Any | None = None
        logger.info(
            "StationDB initialized with connection: %s, table: %s",
            self.connection_string,
            self.table_name,
        )

    async def _ensure_connection(self) -> None:
        """Ensure database connection is established"""
        if self._db is None:
            import aiosqlite

            db_path = self.connection_string.replace("sqlite:///", "")
            if not db_path:
                raise ValueError(
                    f"Invalid connection string: '{self.connection_string}'. "
                    "Expected format: 'sqlite:///path/to/db.sqlite'"
                )
            try:
                self._db = await aiosqlite.connect(db_path, timeout=30)
                await self._create_table()
                logger.info("Database connection established")
            except aiosqlite.OperationalError as exc:
                raise ConnectionError(
                    f"Failed to open SQLite database at '{db_path}': {exc}"
                ) from exc
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Connection to SQLite database at '{db_path}' timed out"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Unexpected error connecting to SQLite database at '{db_path}': {exc}"
                ) from exc

    async def _create_table(self) -> None:
        """Create payloads table if it doesn't exist"""
        if self._db is None:
            raise RuntimeError("Database connection not established")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            tool_call_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        await self._db.execute(create_table_sql)
        await self._db.commit()
        logger.debug("Table %s created or already exists", self.table_name)

    async def store_payload(self, tool_call_id: str, payload: Any) -> None:
        """
        Store payload in database

        Args:
            tool_call_id: Unique identifier for the tool call
            payload: Full payload to store
        """
        await self._ensure_connection()
        if self._db is None:
            raise RuntimeError("Database connection not established")

        payload_json = json.dumps(payload.to_dict() if hasattr(payload, "to_dict") else payload)
        insert_sql = f"""
        INSERT OR REPLACE INTO {self.table_name} (tool_call_id, payload)
        VALUES (?, ?)
        """
        await self._db.execute(insert_sql, (tool_call_id, payload_json))
        await self._db.commit()
        logger.debug("Stored payload for tool_call_id: %s in database", tool_call_id)

    async def get_payload(self, tool_call_id: str) -> Any:
        """
        Retrieve payload from database

        Args:
            tool_call_id: Unique identifier for the tool call

        Returns:
            Stored payload if found, None otherwise
        """
        await self._ensure_connection()
        if self._db is None:
            raise RuntimeError("Database connection not established")

        select_sql = f"""
        SELECT payload FROM {self.table_name}
        WHERE tool_call_id = ?
        """
        async with self._db.execute(select_sql, (tool_call_id,)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                logger.debug("No payload found for tool_call_id: %s in database", tool_call_id)
                return None
            payload = json.loads(row[0])
            logger.debug("Retrieved payload for tool_call_id: %s from database", tool_call_id)
            return payload

    async def close(self) -> None:
        """Close database connection"""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("Database connection closed")

    # Enable "async with StationDB() as db" usage

    async def __aenter__(self) -> "StationDB":
        """Async context manager entry"""
        await self._ensure_connection()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.close()
