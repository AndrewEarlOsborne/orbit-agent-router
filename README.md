# ORBIT MCP LAYER DATA HANDLING AND MASKING

Orbit is a client-side tool data payload handling technology. This enables more observable and deterministic data handling from tool calls, with robust customizability, masking for sensitive information, and data routing. Orbit protects users and developers from exposing sensitive information, adding large data payloads to model context and adding data observability to agentic solutions.

Orbit is a lightweight, python-native solution that can leverage a local cache or a production-level database to store payloads, designed for tool-based agentic systems.

## Purpose
1. Enable AI Engineers to more effectively observe and control large data payloads
2. Choose how to expose, mask, rehydrate, or replace data
3. Stop sensitive/PII information, large payloads, or irrelevant data from being injected into an Agentic LLM's context

## Benefits
1. Reduce Sensitive/PII info exposure
2. Reduce the token cost and cognitive cost of providing data payloads to LLMs
3. Allow customizable deterministic summarization of data payloads to inject relevant context

## Core Concepts

### Terminology
- **Payload/Artifact**: Synonymous terms describing the non-metadata data returned from an LLM agent's tool call
- **Launchpad**: Configurable wrapper that intercepts tool execution and manages payload handling
- **Station**: Storage backend (cache or database) for full payloads
- **Orbit Tool**: A tool wrapped by a Launchpad to enable payload interception and masking

## Local Cache Quickstart
1. Import Orbit
```{python}
from orbit import StationCache, Launchpad
```
2. Make an async orbit cache
```{python}
station = StationCache(my_cache)
```
 
3. Attach an orbit Launchpad to tool calls in your client (Like an MCP client or langgraph toolnode)

### MCP Client
Wrap the tools on discovery
```{python}
async def connect_to_server(self, server_script_path: str):
    """Connect to an MCP server

    Args:
        server_script_path: Path to the server script (.py or .js)
    """
    server_params = StdioServerParameters(
        command=command,
        args=[server_script_path],
        env=None
    )

    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
    self.stdio, self.write = stdio_transport
    self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

    await self.session.initialize()

    # List available tools
    response = await self.session.list_tools()
    tools = response.tools

    # Wrap tools as orbit tools
    default_launchpad = Launchpad()
    orbit_tools = []

    for tool in tools:
        orbit_tools.append(default_launchpad.stage(tool))

# Run the agent and handle tools and messages
async def process_query(self, query: str) -> str:
    """Process a query using Claude and available tools"""
    messages = [
        {
            "role": "user",
            "content": query
        }
    ]

    response = await self.session.list_tools()
    available_tools = [{
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.inputSchema
    } for tool in response.tools]

    # Initial Claude API call
    response = self.anthropic.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=messages,
        tools=available_tools
    )
```

Orbit intercepts all data artifacts from MCP tools for the MCP Client type, '

<Reference: https://modelcontextprotocol.io/docs/develop/build-client>


### For LangChain Toolnode
1. Wrap existing tools using launchpad's stage functionality
```{python}
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from orbit import Launchpad

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Wrap tools as orbit tools
default_launchpad = Launchpad()
orbit_tools = []

for tool in tools:
    orbit_tools.append(default_launchpad.stage(tool))

tool_node = ToolNode(tool_node_tools)
```

2. Declare tools as orbit tools

```{python}
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

<Reference: https://langchain-ai.github.io/langgraph/how-tos/tool-calling/?_gl=1*15yvv83*_ga*MTYwMjY4NzI0NS4xNzU5MDEzNTIw*_ga_47WX3HKKY2*czE3NTkwMTM1MjAkbzEkZzAkdDE3NTkwMTM1MjAkajYwJGwwJGgw>


## Database-enabled Quickstart
This solution leverages a live database connection to handle enterprise and production level requests. 
1. Import StationDB

```{python}
from orbit import StationDB 
```

2. Make an async orbit cache
```{python}
station = StationDB()
```

3. Fetch against this DB with a specific tool call id
```
station.get_payload()
```

## Launchpad Architecture and Behavior

### How Launchpads Work

Launchpads wrap tools to intercept their execution flow. The wrapping process modifies runtime behavior without changing tool semantics or functionality.

**Execution Flow:**
1. Tool is called with arguments
2. Tool executes normally and returns result
3. **Interception Point**: Launchpad intercepts the result
4. Result validation: Check if result is a valid tool message
5. Payload storage: Full result (including result.content) is copied and saved to Station with tool_call_id as key
6. Summary generation: `_generate_summary()` is called to create a masked/summarized version
7. Modified result with summary is returned to agent

### Default Launchpad

The default Launchpad operates on tool response objects and implements automatic masking for large content.

**Default Summary Generation Logic:**
- Iterates over each key-value pair in result.content (which is a list[dict] in MCP)
- For MCP tools, each dict has a "type" key with fields according to that type (e.g., "text", "data", "resource")
- For each content item:
  - If any string value exceeds 2048 characters, it is masked
  - Masked values are replaced with metadata: `{"type": <value_type>, "length": <char_count>, "summary": "Content masked - exceeds 2048 chars"}`
  - Short values (under 2048 chars) pass through unchanged
- Keys are assumed to be unique identifiers and are preserved

**Type-Aware Wrapping:**
- Launchpad.stage() accepts different tool types (MCP tools, LangChain tools, etc.)
- Internal type detection switches to appropriate wrapping logic
- Each tool type requires different interception mechanisms but produces consistent behavior

### Custom Launchpad

Custom Launchpads extend the base Launchpad class to implement domain-specific summarization logic.

**Example: Weather Alert MCP Tool**
```python
from orbit import Launchpad
from typing import Dict, Any, List

class WeatherLaunchpad(Launchpad):
    def _generate_summary(
        self,
        tool_call_id: str,
        tool_name: str,
        content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Custom summary for weather MCP tool that extracts alert counts"""

        summary_content = []

        for item in content:
            if item.get("type") == "data":
                raw_data = item.get("data", {})
                alert_count = len(raw_data.get("Alerts", []))

                if alert_count == 0:
                    summary_content.append({
                        "type": "text",
                        "text": "No weather alerts"
                    })
                else:
                    summary_content.append({
                        "type": "text",
                        "text": f"{alert_count} weather alert(s) detected"
                    })
            else:
                summary_content.append(item)

        return summary_content
```

**Custom Launchpad Usage:**
```python
from orbit import StationCache

station = StationCache()
weather_launchpad = WeatherLaunchpad(station=station)

# Wrap tools
for tool in mcp_tools:
    orbit_tools.append(weather_launchpad.stage(tool))
```

<Reference: https://github.com/hideya/mcp-server-weather-js>

## Debugging and Observability
Orbit enables Agentic systems to retry incorrect tool calls by dynamically returning relevant errors to the agent while providing the full error or failure to the developer, with full configurability.

## Technical Specifications

### Module Structure
```
orbit/
├── __init__.py
├── launchpad.py       # Base Launchpad class and default implementation
├── station.py         # StationCache and StationDB classes
├── wrappers/
│   ├── __init__.py
│   ├── mcp.py        # MCP-specific tool wrapping logic
│   └── langchain.py  # LangChain-specific tool wrapping logic
└── types.py          # Type definitions and protocols
```

### Type Safety
All public APIs use type hints for parameters and return values. Custom Launchpad implementations should maintain type safety in overridden methods.

## Future Work
1. Investigate pydantic framework handling
   1. Investigate adding pythonic agent framework in addtion to langchain, MCP
   2. Investigate pydantic baseclass handling as configs for client-side interaction, ie with classes
      1. Ensure LLM format constraining is maintained when using the module
2. Add summary creation functions for common use-case tools (primarily MCP) to be levered correctly by default.
   1. Database MCPs (Common and have standardized MCPs)
3. Investigate buffering logic when working with Larger Than Memory data results
   1. Lever Blockchain scraper