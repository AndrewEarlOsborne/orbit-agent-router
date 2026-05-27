"""Tests for Orbit Transformations framework"""

import pytest
from orbit.transformations.base import DataType, TransformationRegistry, get_registry
from orbit.transformations.decorators import orbit_transformation_tool_mcp
from orbit.transformations.resources import ResourceManager, TransformContext


class TestDataType:
    """Tests for DataType enum"""

    def test_data_type_values(self) -> None:
        """Test that DataType enum has expected values"""
        assert DataType.CSV.value == "csv"
        assert DataType.JSON.value == "json"
        assert DataType.SQL.value == "sql"
        assert DataType.PARQUET.value == "parquet"

    def test_data_type_enum_members(self) -> None:
        """Test DataType has all expected members"""
        members = [dt.name for dt in DataType]
        assert "CSV" in members
        assert "JSON" in members
        assert "SQL" in members
        assert "PARQUET" in members


class TestTransformationRegistry:
    """Tests for TransformationRegistry"""

    def test_registry_initialization(self) -> None:
        """Test registry initializes empty"""
        registry = TransformationRegistry()
        assert registry is not None
        all_transforms = registry.list_all()
        assert all_transforms == {}

    def test_register_transformation(self) -> None:
        """Test registering a transformation"""
        registry = TransformationRegistry()

        def dummy_transform() -> None:
            pass

        registry.register(
            name="dummy",
            data_type=DataType.CSV,
            description="A dummy transform",
            function=dummy_transform,
            parameters={},
        )

        # Verify registration
        retrieved = registry.get(DataType.CSV, "dummy")
        assert retrieved is not None
        assert retrieved.name == "dummy"
        assert retrieved.description == "A dummy transform"

    def test_register_duplicate_raises_error(self) -> None:
        """Test that registering duplicate raises error"""
        registry = TransformationRegistry()

        def dummy_transform() -> None:
            pass

        registry.register(
            name="dummy",
            data_type=DataType.CSV,
            description="First",
            function=dummy_transform,
            parameters={},
        )

        with pytest.raises(ValueError):
            registry.register(
                name="dummy",
                data_type=DataType.CSV,
                description="Second",
                function=dummy_transform,
                parameters={},
            )

    def test_list_by_type(self) -> None:
        """Test listing transformations by type"""
        registry = TransformationRegistry()

        def dummy_transform() -> None:
            pass

        registry.register(
            name="csv_transform",
            data_type=DataType.CSV,
            description="CSV transform",
            function=dummy_transform,
            parameters={},
        )

        csv_transforms = registry.list_by_type(DataType.CSV)
        assert len(csv_transforms) == 1
        assert csv_transforms[0].name == "csv_transform"

    def test_get_nonexistent_transform(self) -> None:
        """Test getting non-existent transform returns None"""
        registry = TransformationRegistry()
        result = registry.get(DataType.CSV, "nonexistent")
        assert result is None


class TestResourceManager:
    """Tests for ResourceManager"""

    def test_resolve_local_file(self) -> None:
        """Test resolving local file paths"""
        manager = ResourceManager()
        path = "/tmp/test.csv"
        resolved = manager.resolve_uri(path)
        assert resolved == path

    def test_resolve_s3_uri(self) -> None:
        """Test resolving S3 URIs"""
        manager = ResourceManager()
        s3_uri = "s3://bucket/key/data.csv"
        resolved = manager.resolve_uri(s3_uri)
        assert resolved == s3_uri

    def test_resolve_http_uri(self) -> None:
        """Test resolving HTTP(S) URIs"""
        manager = ResourceManager()
        http_uri = "https://example.com/data.csv"
        resolved = manager.resolve_uri(http_uri)
        assert resolved == http_uri

    def test_resolve_invalid_scheme(self) -> None:
        """Test resolving invalid URI scheme raises error"""
        manager = ResourceManager()
        with pytest.raises(ValueError):
            manager.resolve_uri("invalid://path/file.csv")

    def test_generate_output_uri_copy_mode(self) -> None:
        """Test generating output URI in copy mode"""
        manager = ResourceManager()
        original = "/tmp/data.csv"
        output = manager.generate_output_uri(original, "filter", in_place=False)
        assert "filter" in output
        assert output != original

    def test_generate_output_uri_inplace_mode(self) -> None:
        """Test generating output URI in in-place mode"""
        manager = ResourceManager()
        original = "/tmp/data.csv"
        output = manager.generate_output_uri(original, "filter", in_place=True)
        assert output == original


class TestTransformContext:
    """Tests for TransformContext"""

    def test_context_initialization(self) -> None:
        """Test creating transform context"""
        ctx = TransformContext(
            tool_call_id="abc123",
            original_uri="/tmp/data.csv",
            transform_name="filter",
            in_place=False,
        )

        assert ctx.tool_call_id == "abc123"
        assert ctx.original_uri == "/tmp/data.csv"
        assert ctx.transform_name == "filter"
        assert ctx.in_place is False
        assert ctx.execution_id is not None

    def test_context_new_ref_id_copy_mode(self) -> None:
        """Test new ref ID in copy mode"""
        ctx = TransformContext(
            tool_call_id="abc123",
            original_uri="/tmp/data.csv",
            transform_name="filter",
            in_place=False,
        )

        assert ctx.new_tool_call_id != ctx.tool_call_id
        assert "filter" in ctx.new_tool_call_id

    def test_context_same_ref_id_inplace_mode(self) -> None:
        """Test same ref ID in in-place mode"""
        ctx = TransformContext(
            tool_call_id="abc123",
            original_uri="/tmp/data.csv",
            transform_name="filter",
            in_place=True,
        )

        assert ctx.new_tool_call_id == ctx.tool_call_id


class TestDecorator:
    """Tests for @orbit_transformation_tool_mcp decorator"""

    def test_decorator_registers_transformation(self) -> None:
        """Test that decorator registers the transformation"""
        registry = get_registry()

        @orbit_transformation_tool_mcp(data_type=DataType.CSV, description="Test transform")
        async def test_transform(resource_uri: str, in_place: bool = False) -> dict[str, str]:
            return {"type": "text", "text": "Test"}

        # Verify it was registered
        retrieved = registry.get(DataType.CSV, "test_transform")
        assert retrieved is not None
        assert retrieved.name == "test_transform"

    def test_decorator_preserves_function_name(self) -> None:
        """Test that decorator preserves function name"""

        @orbit_transformation_tool_mcp(data_type=DataType.CSV, description="Test transform")
        async def my_custom_transform(resource_uri: str, in_place: bool = False) -> dict[str, str]:
            return {"type": "text", "text": "Test"}

        assert my_custom_transform.__name__ == "my_custom_transform"

    def test_decorator_adds_metadata(self) -> None:
        """Test that decorator adds orbit metadata to function"""

        @orbit_transformation_tool_mcp(
            data_type=DataType.CSV, description="Test transform metadata"
        )
        async def test_transform_metadata(
            resource_uri: str, in_place: bool = False
        ) -> dict[str, str]:
            return {"type": "text", "text": "Test"}

        assert hasattr(test_transform_metadata, "_orbit_transform")
        assert test_transform_metadata._orbit_transform is True
        assert test_transform_metadata._orbit_data_type == DataType.CSV
        assert test_transform_metadata._orbit_description == "Test transform metadata"

    @pytest.mark.asyncio
    async def test_decorator_async_function_execution(self) -> None:
        """Test that async decorated function can be executed"""

        @orbit_transformation_tool_mcp(data_type=DataType.CSV, description="Test async execution")
        async def test_transform_async(
            resource_uri: str, param1: str = "default", in_place: bool = False
        ) -> dict[str, str]:
            return {
                "type": "text",
                "text": f"Processed {resource_uri} with {param1}",
            }

        result = await test_transform_async(
            resource_uri="/tmp/test.csv", param1="test_value", in_place=False
        )

        assert isinstance(result, dict)
        assert result["type"] == "text"
        assert "execution_id" in result
        assert "new_tool_call_id" in result

    def test_decorator_sync_function_execution(self) -> None:
        """Test that sync decorated function can be executed"""

        @orbit_transformation_tool_mcp(data_type=DataType.CSV, description="Test sync execution")
        def test_transform_sync(
            resource_uri: str, param1: str = "default", in_place: bool = False
        ) -> dict[str, str]:
            return {
                "type": "text",
                "text": f"Processed {resource_uri} with {param1}",
            }

        result = test_transform_sync(
            resource_uri="/tmp/test.csv", param1="test_value", in_place=False
        )

        assert isinstance(result, dict)
        assert result["type"] == "text"
        assert "execution_id" in result
        assert "new_tool_call_id" in result


class TestCSVTransforms:
    """Tests for example CSV transforms"""

    @pytest.mark.asyncio
    async def test_count_csv_import(self) -> None:
        """Test that count_csv can be imported"""
        from orbit.transformations.examples.csv_transforms import count_csv

        assert count_csv is not None

    @pytest.mark.asyncio
    async def test_filter_csv_import(self) -> None:
        """Test that filter_csv can be imported"""
        from orbit.transformations.examples.csv_transforms import filter_csv

        assert filter_csv is not None

    @pytest.mark.asyncio
    async def test_select_csv_import(self) -> None:
        """Test that select_csv can be imported"""
        from orbit.transformations.examples.csv_transforms import select_csv

        assert select_csv is not None

    @pytest.mark.asyncio
    async def test_rename_csv_import(self) -> None:
        """Test that rename_csv can be imported"""
        from orbit.transformations.examples.csv_transforms import rename_csv

        assert rename_csv is not None

    @pytest.mark.asyncio
    async def test_group_by_csv_import(self) -> None:
        """Test that group_by_csv can be imported"""
        from orbit.transformations.examples.csv_transforms import group_by_csv

        assert group_by_csv is not None


class TestIntegration:
    """Integration tests for transformations framework"""

    def test_end_to_end_decorator_registration(self) -> None:
        """Test full workflow of decorator and registration"""

        @orbit_transformation_tool_mcp(
            data_type=DataType.JSON, description="JSON transform for testing"
        )
        async def json_transform(
            resource_uri: str, field: str = "data", in_place: bool = False
        ) -> dict[str, str]:
            return {"type": "text", "text": f"Processed field: {field}"}

        registry = get_registry()
        retrieved = registry.get(DataType.JSON, "json_transform")

        assert retrieved is not None
        assert retrieved.description == "JSON transform for testing"
        assert retrieved.data_type == DataType.JSON
