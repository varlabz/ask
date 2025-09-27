import json


class TestNDJSONFormatting:
    """Test NDJSON formatting for chat messages."""

    def test_format_chat_messages_as_ndjson(self):
        """Test formatting chat messages as NDJSON."""
        chat_messages = [
            {"role": "user", "timestamp": "2024-01-01T12:00:00Z", "content": "Hello"},
            {
                "role": "assistant",
                "timestamp": "2024-01-01T12:00:01Z",
                "content": "Hi there!",
            },
        ]

        # Simulate the NDJSON formatting from the API
        payload = b"\n".join(json.dumps(msg).encode("utf-8") for msg in chat_messages)

        # Parse back to verify
        lines = payload.decode().strip().split("\n")
        assert len(lines) == 2

        msg1 = json.loads(lines[0])
        msg2 = json.loads(lines[1])

        assert msg1["role"] == "user"
        assert msg1["content"] == "Hello"
        assert msg2["role"] == "assistant"
        assert msg2["content"] == "Hi there!"

    def test_empty_chat_messages_ndjson(self):
        """Test NDJSON formatting with empty message list."""
        chat_messages = []
        payload = b"\n".join(json.dumps(msg).encode("utf-8") for msg in chat_messages)

        assert payload == b""
