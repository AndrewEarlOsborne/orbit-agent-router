"""Test suite for Station storage backends"""

import pytest
from orbit import StationCache, StationDB


class TestStationCache:
    """Test in-memory cache storage"""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_payload(self) -> None:
        """Test basic store and retrieve operations"""
        station = StationCache()
        tool_call_id = "test-id-123"
        payload = {"content": [{"type": "text", "text": "test"}]}

        await station.store_payload(tool_call_id, payload)
        retrieved = await station.get_payload(tool_call_id)

        assert retrieved == payload

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_payload(self) -> None:
        """Test retrieving non-existent payload returns None"""
        station = StationCache()
        retrieved = await station.get_payload("nonexistent-id")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_store_multiple_payloads(self) -> None:
        """Test storing multiple payloads"""
        station = StationCache()

        payload1 = {"content": [{"type": "text", "text": "first"}]}
        payload2 = {"content": [{"type": "text", "text": "second"}]}

        await station.store_payload("id-1", payload1)
        await station.store_payload("id-2", payload2)

        assert len(station._cache) == 2
        assert await station.get_payload("id-1") == payload1
        assert await station.get_payload("id-2") == payload2

    @pytest.mark.asyncio
    async def test_overwrite_payload(self) -> None:
        """Test that storing with same ID overwrites"""
        station = StationCache()
        tool_call_id = "test-id"

        payload1 = {"content": [{"type": "text", "text": "first"}]}
        payload2 = {"content": [{"type": "text", "text": "second"}]}

        await station.store_payload(tool_call_id, payload1)
        await station.store_payload(tool_call_id, payload2)

        assert len(station._cache.keys()) == 1
        assert await station.get_payload(tool_call_id) == payload2

    @pytest.mark.asyncio
    async def test_clear_cache(self) -> None:
        """Test clearing all cached payloads"""
        station = StationCache()

        await station.store_payload("id-1", {"data": "1"})
        await station.store_payload("id-2", {"data": "2"})

        station.clear()

        assert len(station._cache) == 0
        assert await station.get_payload("id-1") is None

    @pytest.mark.asyncio
    async def test_custom_cache_dict(self) -> None:
        """Test using custom cache dictionary"""
        custom_cache = {"existing-id": {"content": "existing"}}
        station = StationCache(cache=custom_cache)

        assert len(station._cache.keys()) == 1
        retrieved = await station.get_payload("existing-id")
        assert retrieved == {"content": "existing"}

    @pytest.mark.asyncio
    async def test_store_complex_payload(self) -> None:
        """Test storing complex nested payload"""
        station = StationCache()
        complex_payload = {
            "content": [
                {"type": "text", "text": "some text"},
                {
                    "type": "data",
                    "data": {"nested": {"deeply": {"values": [1, 2, 3], "flag": True}}},
                },
            ],
            "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
        }

        await station.store_payload("complex-id", complex_payload)
        retrieved = await station.get_payload("complex-id")

        assert retrieved == complex_payload


class TestStationDB:
    """Test database-backed storage"""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_payload(self) -> None:
        """Test basic store and retrieve with database"""
        # Use in-memory SQLite database for testing
        async with StationDB(connection_string="sqlite:///:memory:") as station:
            tool_call_id = "test-id-123"
            payload = {"content": [{"type": "text", "text": "test"}]}

            await station.store_payload(tool_call_id, payload)
            retrieved = await station.get_payload(tool_call_id)

            assert retrieved == payload

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_payload(self) -> None:
        """Test retrieving non-existent payload returns None"""
        async with StationDB(connection_string="sqlite:///:memory:") as station:
            retrieved = await station.get_payload("nonexistent-id")

            assert retrieved is None

    @pytest.mark.asyncio
    async def test_store_multiple_payloads(self) -> None:
        """Test storing multiple payloads in database"""
        async with StationDB(connection_string="sqlite:///:memory:") as station:
            payload1 = {"content": [{"type": "text", "text": "first"}]}
            payload2 = {"content": [{"type": "text", "text": "second"}]}

            await station.store_payload("id-1", payload1)
            await station.store_payload("id-2", payload2)

            assert await station.get_payload("id-1") == payload1
            assert await station.get_payload("id-2") == payload2

    @pytest.mark.asyncio
    async def test_overwrite_payload(self) -> None:
        """Test that storing with same ID overwrites in database"""
        async with StationDB(connection_string="sqlite:///:memory:") as station:
            tool_call_id = "test-id"

            payload1 = {"content": [{"type": "text", "text": "first"}]}
            payload2 = {"content": [{"type": "text", "text": "second"}]}

            await station.store_payload(tool_call_id, payload1)
            await station.store_payload(tool_call_id, payload2)

            retrieved = await station.get_payload(tool_call_id)
            assert retrieved == payload2

    @pytest.mark.asyncio
    async def test_complex_payload_serialization(self) -> None:
        """Test JSON serialization of complex payloads"""
        async with StationDB(connection_string="sqlite:///:memory:") as station:
            complex_payload = {
                "content": [
                    {"type": "text", "text": "some text"},
                    {
                        "type": "data",
                        "data": {"nested": {"values": [1, 2, 3], "flag": True, "none_value": None}},
                    },
                ]
            }

            await station.store_payload("complex-id", complex_payload)
            retrieved = await station.get_payload("complex-id")

            assert retrieved == complex_payload

    @pytest.mark.asyncio
    async def test_custom_table_name(self) -> None:
        """Test using custom table name"""
        async with StationDB(
            connection_string="sqlite:///:memory:", table_name="custom_payloads"
        ) as station:
            payload = {"content": [{"type": "text", "text": "test"}]}

            await station.store_payload("test-id", payload)
            retrieved = await station.get_payload("test-id")

            assert retrieved == payload
