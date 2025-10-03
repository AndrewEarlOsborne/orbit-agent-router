"""Test that orbit package imports work correctly"""



def test_import_core_classes() -> None:
    """Test importing core classes"""
    from orbit import Launchpad, DefaultLaunchpad, Station, StationCache, StationDB

    assert Launchpad is not None
    assert DefaultLaunchpad is not None
    assert Station is not None
    assert StationCache is not None
    assert StationDB is not None


def test_import_protocols() -> None:
    """Test importing protocol types"""
    from orbit import (
        ToolProtocol,
        MCPToolProtocol,
        LangChainToolProtocol,
        MCPToolResult,
        TextContent,
        DataContent,
        ResourceContent,
    )

    assert ToolProtocol is not None
    assert MCPToolProtocol is not None
    assert LangChainToolProtocol is not None
    assert MCPToolResult is not None
    assert TextContent is not None
    assert DataContent is not None
    assert ResourceContent is not None


def test_import_mcp_wrappers() -> None:
    """Test importing MCP wrappers"""
    from orbit import wrap_mcp_tool, MCPClientInterceptor, intercept_mcp_session

    assert wrap_mcp_tool is not None
    assert MCPClientInterceptor is not None
    assert intercept_mcp_session is not None


def test_import_langchain_wrappers() -> None:
    """Test importing LangChain wrappers"""
    from orbit import (
        wrap_langchain_tool,
        LangChainToolNodeInterceptor,
        intercept_tool_node,
    )

    assert wrap_langchain_tool is not None
    assert LangChainToolNodeInterceptor is not None
    assert intercept_tool_node is not None


def test_version() -> None:
    """Test package version is defined"""
    import orbit

    assert hasattr(orbit, "__version__")
    assert orbit.__version__ == "0.1.0"


def test_all_exports() -> None:
    """Test that __all__ includes expected exports"""
    import orbit

    expected_exports = [
        "Launchpad",
        "DefaultLaunchpad",
        "Station",
        "StationCache",
        "StationDB",
        "ToolProtocol",
        "MCPToolProtocol",
        "LangChainToolProtocol",
        "MCPToolResult",
        "TextContent",
        "DataContent",
        "ResourceContent",
        "wrap_mcp_tool",
        "MCPClientInterceptor",
        "intercept_mcp_session",
        "wrap_langchain_tool",
        "LangChainToolNodeInterceptor",
        "intercept_tool_node",
    ]

    for export in expected_exports:
        assert export in orbit.__all__, f"{export} not in __all__"
        assert hasattr(orbit, export), f"{export} not accessible from orbit"


def test_no_types_module_conflict() -> None:
    """Test that orbit.types doesn't conflict with stdlib types module"""
    import types as stdlib_types

    # Verify we can still use stdlib types
    assert hasattr(stdlib_types, "SimpleNamespace")

    # Verify orbit protocols are accessible
    from orbit.protocols import ToolProtocol

    assert ToolProtocol is not None
