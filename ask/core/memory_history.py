import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HistoryEntry(BaseModel):
    """A single history entry containing a query-response pair."""

    id: int = Field(description="Unique identifier for the history entry")
    query: str = Field(description="The original user query or request")
    content: str = Field(description="The LLM response or content")
    timestamp: int = Field(description="Unix timestamp of when the entry was created")
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")


class History:
    """
    SQLite-based history management for storing query-content pairs.

    Provides methods to add history entries and retrieve paginated results
    with optional timestamp filtering.
    """

    def __init__(self, collection_name: str = "history", db_path: str = "./history.db"):
        """
        Initialize History with SQLite database.

        Args:
            db_path: Path to the SQLite database file
            collection_name: Name of the table to store history entries
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self._init_db()

    def _init_db(self) -> None:
        """Create the history table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            """)
            # Create index on timestamp for efficient filtering
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_timestamp
                ON {self.collection_name}(timestamp)
            """)
            conn.commit()

    def add_history(
        self,
        query: str,
        content: str,
        timestamp: int | None = None,
    ) -> int:
        """
        Add a new history entry to the database.

        Use this tool to:
        - Save conversation exchanges (query and response)
        - Store important interactions for future reference

        The entry is automatically timestamped if no timestamp is provided.

        Example usage:
        - After generating a response, save it with the original query
        - Store user preferences or decisions made during conversations
        - Keep track of all interactions chronologically

        Args:
            query: The user's original query or prompt text
            content: The assistant's response or content that was generated
            timestamp: Unix timestamp for the entry (optional, defaults to current time)

        Returns:
            The ID of the created history entry
        """
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO {self.collection_name} (query, content, timestamp) VALUES (?, ?, ?)",
                (query, content, timestamp),
            )
            conn.commit()
            row_id = cursor.lastrowid
            if row_id is None:
                raise RuntimeError("Failed to get last row ID from database")
            return row_id

    def get_page(
        self,
        page: int = 1,
        page_size: int = 10,
        before: int | None = None,
        order: Literal["asc", "desc"] = "desc",
    ) -> list[HistoryEntry]:
        """
        Retrieve a paginated list of history entries sorted by timestamp.

        Use this tool to:
        - Browse through all stored history entries chronologically
        - Get recent conversation history (use with before=None)

        Filtering by time:
        - Use 'before' parameter to get only entries created before a specific time
        - First use get_timestamp() to generate the timestamp
        - Example: get_timestamp(days=7) returns a timestamp from 7 days ago
        - Use that timestamp as 'before' to get all entries older than 7 days
        - If 'before' is not provided, returns all entries up to now

        Pagination:
        - Results are sorted by timestamp in the specified order
        - Use 'page' to navigate through results
        - Adjust 'page_size' to control how many results per page (1-100)

        Args:
            page: Page number to retrieve (1-based index, default=1)
            page_size: Maximum number of results to return per page (1-100, default=10)
            before: Unix timestamp - only return entries created before this time (optional, defaults to now)
            order: Sort order - "desc" for newest first, "asc" for oldest first (default="desc")

        Returns:
            List of history entries with ID, query, content, and timestamp
        """
        if page < 1:
            raise ValueError("Page must be >= 1")
        if page_size < 1 or page_size > 100:
            raise ValueError("Page size must be between 1 and 100")

        if before is None:
            before = int(datetime.now().timestamp())

        offset = (page - 1) * page_size
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, query, content, timestamp
                FROM {self.collection_name}
                WHERE timestamp <= ?
                ORDER BY timestamp {order.upper()}
                LIMIT ? OFFSET ?
                """,
                (before, page_size, offset),
            )
            rows = cursor.fetchall()
            return [
                HistoryEntry(id=row[0], query=row[1], content=row[2], timestamp=row[3])
                for row in rows
            ]

    def count(self, before: int | None = None) -> int:
        """
        Get total count of history entries.

        Args:
            before: Optional Unix timestamp to count entries created before this time

        Returns:
            Total number of entries

        Example:
            history = History()
            total = history.count()
            print(f"Total entries: {total}")
        """
        if before is None:
            before = int(datetime.now().timestamp())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.collection_name} WHERE timestamp <= ?",
                (before,),
            )
            return cursor.fetchone()[0]

    def clear(self) -> int:
        """
        Delete all entries from the collection.

        Returns:
            Number of rows deleted

        Example:
            history = History()
            deleted_count = history.clear()
            print(f"Deleted {deleted_count} entries")
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.collection_name}")
            conn.commit()
            return cursor.rowcount

    def get_timestamp(
        self,
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        days: int = 0,
        months: int = 0,
    ) -> int:
        """
        Get a Unix timestamp for a future or past time relative to now.

        Use this tool to:
        - Get current Unix timestamp (call with no arguments)
        - Calculate a timestamp for a future or past time.
        - Generate 'before' parameter values for get_page.

        Time parameters can be positive for future time or negative for past time.
        - get_timestamp() returns current timestamp.
        - get_timestamp(seconds=45) returns timestamp for 45 seconds in the future.
        - get_timestamp(seconds=-45) returns timestamp from 45 seconds ago.
        - get_timestamp(minutes=30) returns timestamp for 30 minutes in the future.
        - get_timestamp(minutes=-30) returns timestamp from 30 minutes ago.
        - get_timestamp(days=7) returns timestamp for 7 days in the future.
        - get_timestamp(days=-7) returns timestamp from 7 days ago.

        Args:
            seconds: Number of seconds to go forward (positive) or backward (negative) in time (default=0).
            minutes: Number of minutes to go forward (positive) or backward (negative) in time (default=0).
            hours: Number of hours to go forward (positive) or backward (negative) in time (default=0).
            days: Number of days to go forward (positive) or backward (negative) in time (default=0).
            months: Number of months (30 days each) to go forward (positive) or backward (negative) in time (default=0).

        Returns:
            Unix timestamp as integer (seconds since Jan 1, 1970 UTC).
        """
        now = datetime.now()
        if days:
            now = now.replace(hour=0, minute=0, second=0, microsecond=0)
            # yesterday is -1, but we want the start of the day after midnight
            days += 1  # because we want the start of the day after
        if months:
            # last month is -1, but we want the start of the month after midnight
            now = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            months += 1  # because we want the start of the month after

        offset = timedelta(
            seconds=seconds, minutes=minutes, hours=hours, days=days + (months * 30)
        )
        result_time = now + offset
        return int(result_time.timestamp())

    def get_time_by_timestamp(self, timestamp: int) -> str:
        """
        Convert a Unix timestamp to an ISO 8601 formatted string with timezone.

        Use this tool to:
        - Convert Unix timestamps to human-readable date/time strings
        - Display timestamps from history entries in a readable format
        - Get timezone-aware time representations

        The function automatically detects and uses the local timezone.

        Args:
            timestamp: Unix timestamp (seconds since Jan 1, 1970 UTC)

        Returns:
            ISO 8601 formatted string with timezone (e.g., "2025-10-31T14:30:00-07:00")

        Example:
            history = History()
            iso_time = history.get_time_by_timestamp(1730400000)
            print(iso_time)  # "2024-10-31T12:00:00-07:00"
        """
        dt = datetime.fromtimestamp(timestamp)
        return dt.isoformat()


if __name__ == "__main__":
    """CLI entry point for managing history."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage history entries in SQLite database"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./history.db",
        help="Path to SQLite database file (default: ./history.db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="history",
        help="Collection name (table name) (default: history)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new history entry")
    add_parser.add_argument("--query", type=str, required=True, help="User query")
    add_parser.add_argument(
        "--content", type=str, required=True, help="LLM response content"
    )
    add_parser.add_argument(
        "--timestamp", type=int, help="Unix timestamp (optional, defaults to now)"
    )

    # Get page command
    page_parser = subparsers.add_parser("get", help="Get paginated history entries")
    page_parser.add_argument(
        "--page", type=int, default=1, help="Page number (default: 1)"
    )
    page_parser.add_argument(
        "--page-size",
        type=int,
        default=10,
        help="Number of entries per page (default: 10)",
    )
    page_parser.add_argument(
        "--before", type=int, help="Filter entries before this Unix timestamp"
    )
    page_parser.add_argument(
        "--order",
        type=str,
        choices=["asc", "desc"],
        default="desc",
        help="Sort order: 'desc' for newest first, 'asc' for oldest first (default: desc)",
    )

    # Clear command
    subparsers.add_parser("clear", help="Delete all history entries")

    # Count command
    count_parser = subparsers.add_parser("count", help="Get total count of entries")
    count_parser.add_argument(
        "--before", type=int, help="Count entries before this Unix timestamp"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        exit(1)

    history = History(collection_name=args.collection, db_path=args.db_path)

    if args.command == "add":
        entry_id = history.add_history(
            query=args.query, content=args.content, timestamp=args.timestamp
        )
        print(f"Added entry with ID: {entry_id}")

    elif args.command == "get":
        entries = history.get_page(
            page=args.page, page_size=args.page_size, before=args.before
        )
        total = history.count(before=args.before)

        print(f"Page {args.page} (Total entries: {total})")
        print("-" * 80)

        if not entries:
            print("No entries found.")
        else:
            for entry in entries:
                dt = datetime.fromtimestamp(entry.timestamp)
                print(f"\nID: {entry.id}")
                print(
                    f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')} ({entry.timestamp})"
                )
                print(f"Query: {entry.query}")
                print(
                    f"Content: {entry.content[:100]}{'...' if len(entry.content) > 100 else ''}"
                )
                print("-" * 80)

    elif args.command == "clear":
        deleted = history.clear()
        print(f"Deleted {deleted} entries from collection '{args.collection}'")

    elif args.command == "count":
        total = history.count(before=args.before)
        if args.before:
            dt = datetime.fromtimestamp(args.before)
            print(f"Entries before {dt.strftime('%Y-%m-%d %H:%M:%S')}: {total}")
        else:
            print(f"Total entries: {total}")
