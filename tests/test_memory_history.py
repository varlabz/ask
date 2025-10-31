import pytest
from datetime import datetime, timedelta
from ask.core.memory_history import History
import tempfile
import os

@pytest.fixture
def history():
    """Fixture to create a History instance for testing."""
    # Use a temporary file for the database to ensure it persists across connections in a single test
    db_fd, db_path = tempfile.mkstemp()
    
    h = History(db_path=db_path)
    
    yield h
    
    os.close(db_fd)
    os.unlink(db_path)

def test_get_timestamp_now(history):
    """Test that get_timestamp with no arguments returns the current timestamp."""
    now = int(datetime.now().timestamp())
    timestamp = history.get_timestamp()
    assert timestamp >= now
    assert timestamp - now < 2  # Allow for a small delay

def test_get_timestamp_seconds(history):
    """Test get_timestamp with positive and negative seconds."""
    now = datetime.now()
    
    # Positive seconds
    ts_future = history.get_timestamp(seconds=30)
    expected_future = now + timedelta(seconds=30)
    assert abs(ts_future - int(expected_future.timestamp())) <= 1

    # Negative seconds
    ts_past = history.get_timestamp(seconds=-30)
    expected_past = now + timedelta(seconds=-30)
    assert abs(ts_past - int(expected_past.timestamp())) <= 1

def test_get_timestamp_minutes(history):
    """Test get_timestamp with positive and negative minutes."""
    now = datetime.now()
    
    # Positive minutes
    ts_future = history.get_timestamp(minutes=15)
    expected_future = now + timedelta(minutes=15)
    assert abs(ts_future - int(expected_future.timestamp())) <= 1

    # Negative minutes
    ts_past = history.get_timestamp(minutes=-15)
    expected_past = now + timedelta(minutes=-15)
    assert abs(ts_past - int(expected_past.timestamp())) <= 1

def test_get_timestamp_hours(history):
    """Test get_timestamp with positive and negative hours."""
    now = datetime.now()
    
    # Positive hours
    ts_future = history.get_timestamp(hours=2)
    expected_future = now + timedelta(hours=2)
    assert abs(ts_future - int(expected_future.timestamp())) <= 1

    # Negative hours
    ts_past = history.get_timestamp(hours=-2)
    expected_past = now + timedelta(hours=-2)
    assert abs(ts_past - int(expected_past.timestamp())) <= 1

def test_get_timestamp_days(history):
    """Test get_timestamp with days, which should be relative to midnight."""
    now = datetime.now()
    midnight_today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # 1 day in the future (tomorrow at midnight)
    ts_tomorrow = history.get_timestamp(days=1)
    expected_tomorrow = midnight_today + timedelta(days=1 + 1)
    assert ts_tomorrow == int(expected_tomorrow.timestamp())

    # -1 day in the past (yesterday at midnight)
    ts_yesterday = history.get_timestamp(days=-1)
    expected_yesterday = midnight_today + timedelta(days=-1 + 1)
    assert ts_yesterday == int(expected_yesterday.timestamp())

    # 0 days should be today at the current time, not midnight
    ts_today = history.get_timestamp(days=0)
    assert abs(ts_today - int(now.timestamp())) <= 1

def test_get_timestamp_months(history):
    """Test get_timestamp with months, which should be relative to midnight."""
    now = datetime.now()
    midnight_first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # 1 month in the future
    ts_next_month = history.get_timestamp(months=1)
    expected_next_month = midnight_first_of_month + timedelta(days=30 * (1 + 1))
    assert ts_next_month == int(expected_next_month.timestamp())

    # -1 month in the past
    ts_last_month = history.get_timestamp(months=-1)
    expected_last_month = midnight_first_of_month + timedelta(days=30 * (-1 + 1))
    assert ts_last_month == int(expected_last_month.timestamp())

def test_get_timestamp_combined(history):
    """Test get_timestamp with a combination of parameters."""
    now = datetime.now()
    midnight_today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Test with days and hours
    ts = history.get_timestamp(days=1, hours=5)
    expected = midnight_today + timedelta(days=1 + 1, hours=5)
    assert ts == int(expected.timestamp())

def test_get_page_empty(history):
    """Test get_page with no entries in the history."""
    assert history.get_page() == []

def test_get_page_pagination(history):
    """Test pagination of get_page."""
    timestamps = []
    for i in range(25):
        # Ensure timestamps are unique and ordered
        ts = int((datetime.now() - timedelta(seconds=i * 2)).timestamp())
        history.add_history(f"query {i}", f"content {i}", timestamp=ts)
        timestamps.append(ts)
    timestamps.sort(reverse=True)  # Newest first

    # Test page 1
    page1 = history.get_page(page=1, page_size=10)
    assert len(page1) == 10
    assert page1[0].query == "query 0"
    assert page1[9].query == "query 9"
    assert [e.timestamp for e in page1] == timestamps[0:10]

    # Test page 2
    page2 = history.get_page(page=2, page_size=10)
    assert len(page2) == 10
    assert page2[0].query == "query 10"
    assert page2[9].query == "query 19"
    assert [e.timestamp for e in page2] == timestamps[10:20]

    # Test page 3 (partial)
    page3 = history.get_page(page=3, page_size=10)
    assert len(page3) == 5
    assert page3[0].query == "query 20"
    assert page3[4].query == "query 24"
    assert [e.timestamp for e in page3] == timestamps[20:25]

    # Test page 4 (empty)
    page4 = history.get_page(page=4, page_size=10)
    assert len(page4) == 0

def test_get_page_ordering(history):
    """Test 'asc' and 'desc' ordering of get_page."""
    timestamps = []
    for i in range(5):
        ts = int((datetime.now() - timedelta(seconds=i * 2)).timestamp())
        history.add_history(f"query {i}", f"content {i}", timestamp=ts)
        timestamps.append(ts)

    # Test descending order (default)
    desc_entries = history.get_page(order="desc")
    assert len(desc_entries) == 5
    assert desc_entries[0].query == "query 0"
    assert desc_entries[4].query == "query 4"

    # Test ascending order
    asc_entries = history.get_page(order="asc")
    assert len(asc_entries) == 5
    assert asc_entries[0].query == "query 4"
    assert asc_entries[4].query == "query 0"

def test_get_page_before_filter(history):
    """Test the 'before' timestamp filter of get_page."""
    base_time = datetime.now()
    history.add_history("query 1", "content 1", int((base_time - timedelta(days=2)).timestamp()))
    history.add_history("query 2", "content 2", int((base_time - timedelta(days=1)).timestamp()))
    history.add_history("query 3", "content 3", int(base_time.timestamp()))

    # Filter to get entries before 1.5 days ago
    before_ts = int((base_time - timedelta(days=1, hours=12)).timestamp())
    entries = history.get_page(before=before_ts)
    assert len(entries) == 1
    assert entries[0].query == "query 1"

def test_get_page_invalid_args(history):
    """Test that get_page raises ValueError for invalid arguments."""
    with pytest.raises(ValueError, match="Page must be >= 1"):
        history.get_page(page=0)

    with pytest.raises(ValueError, match="Page size must be between 1 and 100"):
        history.get_page(page_size=0)

    with pytest.raises(ValueError, match="Page size must be between 1 and 100"):
        history.get_page(page_size=101)
