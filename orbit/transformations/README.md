# Orbit Transformations Framework

Transform data in-place without loading it into your LLM's context window. The transformations framework enables AI agents to apply operations (filter, select, rename, aggregate, etc.) to stored data while keeping the full dataset server-side. Agents work with summaries and data references only.

## Overview

Orbit Transformations is a **user-extensible framework** for defining and executing data transformations. Instead of bloating your LLM context with large payloads, agents reference stored data by ID and apply transformations. Each transformation can produce a new data snapshot (copy) or update the original in-place, giving agents full control over their data pipeline.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ AI Agent                                                         │
│  - Receives data summary: "CSV: 1M rows, [id, name, price]"    │
│  - Calls: filter_csv(ref="abc123", col="price", op=">", val=100)
│  - Receives: "Filtered: 450K rows"                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Orbit Framework                                                  │
│  - Retrieves resource "abc123" from Station                    │
│  - Passes to your transformation function                      │
│  - Stores result (new ref or in-place update)                 │
│  - Returns summary to agent                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Station (Payload Storage)                                        │
│  - "abc123": original CSV (1M rows)                             │
│  - "filtered_abc123": transformed CSV (450K rows)              │
│  - Full data never enters LLM context                          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Define a Transformation

Create a file (e.g., `my_transforms.py`) in your MCP server:

```python
from orbit.transformations.decorators import orbit_transformation_tool_mcp
from orbit.transformations.base import DataType
from pydantic import BaseModel
from typing import List

class SelectConfig(BaseModel):
    columns: List[str]

@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,
    description="Select specific columns from a CSV file"
)
async def select_csv(
    resource_uri: str,
    columns: List[str],
    in_place: bool = False
) -> dict:
    """
    Select columns from CSV and return summarized result.
    
    Args:
        resource_uri: File path or S3 URI to the CSV file
        columns: List of column names to keep
        in_place: If True, update original. If False, create new entry.
    
    Returns:
        Summary dict with row count, schema, sample rows
    """
    import csv
    from io import StringIO
    
    # Load the CSV
    with open(resource_uri, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter to selected columns
    selected = []
    for row in rows:
        selected.append({col: row.get(col) for col in columns})
    
    # Write result (simplified)
    output_uri = resource_uri + ".selected" if not in_place else resource_uri
    with open(output_uri, 'w', newline='') as f:
        if selected:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(selected)
    
    # Return summary
    return {
        "type": "text",
        "text": f"Selected {len(selected)} rows with columns: {columns}"
    }
```

### 2. The Decorator Handles:

- ✅ Extracting `tool_call_id` from MCP parameters
- ✅ Retrieving the resource from Orbit's Station
- ✅ Decoding the resource URI (file path, S3, etc.)
- ✅ Passing resource location to your function
- ✅ Storing the result (new entry or in-place update)
- ✅ Generating fastmcp tool definition
- ✅ Returning summarized result to the agent

### 3. Your Function Receives:

- `resource_uri` — where to load the data (file path, S3 URI, etc.)
- Transform parameters — from Pydantic config or function args
- `in_place` — controls mutation behavior (default: `False`)

Your function should:
1. Load data from `resource_uri`
2. Apply your transformation
3. Return a summary (row count, schema, sample) — **never return full data**

## In-Place Flag Behavior

### `in_place=False` (Default)

Creates a new Station entry for the transformed data:

```
Original: "abc123" → 1M rows
↓
Filter applied
↓
New: "filtered_abc123" → 450K rows
```

**Use case**: Branching pipelines where you want to preserve history or explore multiple paths.

```python
# Agent does: filter → select → group_by
result1 = filter_csv(ref="abc123", ...)  # Returns "filtered_abc123"
result2 = select_csv(ref="filtered_abc123", ...)  # Returns "selected_filtered_abc123"
```

### `in_place=True`

Updates the original Station entry:

```
Original: "abc123" → 1M rows
↓
Filter applied (in_place=True)
↓
"abc123" → 450K rows (mutated)
```

**Use case**: Linear pipelines where you don't need intermediate snapshots.

```python
# Agent does sequential mutations
filter_csv(ref="abc123", ..., in_place=True)  # "abc123" is now filtered
select_csv(ref="abc123", ..., in_place=True)  # "abc123" is now filtered + selected
```

## Supported Data Types

The framework ships with example transformations for:

- **CSV** — `orbit/transformations/examples/csv_transforms.py`
  - `filter_csv()` — filter rows by column condition
  - `select_csv()` — select specific columns
  - `rename_csv()` — rename columns
  - `group_by_csv()` — group and aggregate
  - `count_csv()` — return row count

Future support planned:
- **JSON** — nested object filtering and selection
- **SQL** — query result transformations
- **Parquet** — columnar data operations
- **A2A (Agent-to-Agent)** — pass data references between agents without serialization

## Building Custom Transformations

### Step 1: Define Your Function

```python
from orbit.transformations.decorators import orbit_transformation_tool_mcp
from orbit.transformations.base import DataType

@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,  # or .JSON, .SQL, etc.
    description="Your transformation description"
)
async def my_transform(
    resource_uri: str,
    param1: str,
    param2: int,
    in_place: bool = False
) -> dict:
    """Your transform logic here."""
    # Load from resource_uri
    # Apply transformation
    # Return summary
```

### Step 2: Load, Transform, Return Summary

```python
async def my_transform(resource_uri: str, in_place: bool = False):
    # 1. LOAD
    with open(resource_uri, 'r') as f:
        data = load_your_format(f)
    
    # 2. TRANSFORM
    transformed = apply_logic(data)
    
    # 3. SAVE (optional, if not in_place)
    output_path = resource_uri if in_place else resource_uri + ".transformed"
    save_your_format(transformed, output_path)
    
    # 4. RETURN SUMMARY (NEVER FULL DATA)
    return {
        "type": "text",
        "text": f"Transformed: {len(transformed)} items, schema: {schema}"
    }
```

### Step 3: Register in Your MCP Server

In your fastmcp server, just import your decorated function:

```python
from fastmcp import Server
from my_transforms import my_transform

server = Server("my-transforms")

# The @orbit_transformation_tool_mcp decorator automatically
# registers my_transform as an MCP tool
server.add_resource(my_transform)
```

That's it. Orbit handles the rest.

## Integration with Launchpad

Transformations work seamlessly with Orbit's existing Launchpad interception:

```python
from orbit import Launchpad, DefaultLaunchpad, StationCache
from orbit.wrappers.mcp_wrapper import intercept_mcp_session

# Set up Launchpad (masks large payloads, stores in Station)
launchpad = DefaultLaunchpad(station=StationCache())

# Intercept your data source tools
interceptor = intercept_mcp_session(client.session, launchpad)

# Agent reads CSV: intercepted → stored with ref "abc123"
result = await client.session.call_tool("read_csv", {"path": "data.csv"})
# Agent receives summary only

# Agent calls transformation: retrieves "abc123", applies logic
result = await client.session.call_tool("filter_csv", {
    "tool_call_id": "abc123",
    "filter_col": "price",
    "operator": ">",
    "value": 100,
    "in_place": False
})
# Returns new ref: "filtered_abc123"
```

## API Reference

### Decorator: `@orbit_transformation_tool_mcp`

```python
@orbit_transformation_tool_mcp(
    data_type: DataType,
    description: str,
    transform_config: Optional[Type[BaseModel]] = None
)
```

**Parameters:**
- `data_type` — `DataType.CSV`, `.JSON`, `.SQL`, etc.
- `description` — Human-readable description for the LLM
- `transform_config` — Optional Pydantic model for validation (auto-extracted from function signature if not provided)

**Returns:**
- Decorated function that Orbit wraps with Station integration

### DataType Enum

```python
from orbit.transformations.base import DataType

DataType.CSV
DataType.JSON
DataType.SQL
DataType.PARQUET
```

## Architecture Details

### Resource URI Handling

Resources are stored in the Station with their execution context. The framework supports:

- **Local file paths**: `/path/to/data.csv`
- **S3 URIs**: `s3://bucket/key/data.csv`
- **HTTP URLs**: `https://example.com/data.csv` (read-only)

URI resolution happens automatically; your function receives the path and can load it with standard I/O.

### TransformationRegistry

The framework maintains a registry of available transformations per data type. When your function is decorated, it's automatically registered:

```python
from orbit.transformations.base import TransformationRegistry

registry = TransformationRegistry()
# Decorated functions are auto-registered by DataType
```

### Station Integration

Transformations store results in the same Station backend as the original data:

```
Station (StationCache or StationDB)
├── tool_call_id: "abc123" → original CSV payload
├── tool_call_id: "filtered_abc123" → filtered CSV payload
└── tool_call_id: "aggregated_abc123" → aggregated CSV payload
```

All transformations are fully logged and retrievable via `await station.get_payload(tool_call_id)`.

## Examples

### Example 1: Simple Filter

```python
@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,
    description="Filter CSV rows by a numeric column"
)
async def filter_by_numeric(
    resource_uri: str,
    column: str,
    min_value: float = None,
    max_value: float = None,
    in_place: bool = False
) -> dict:
    import csv
    
    with open(resource_uri) as f:
        reader = csv.DictReader(f)
        rows = [
            row for row in reader
            if (min_value is None or float(row[column]) >= min_value) and
               (max_value is None or float(row[column]) <= max_value)
        ]
    
    output = resource_uri if in_place else resource_uri + ".filtered"
    with open(output, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    return {
        "type": "text",
        "text": f"Filtered {len(rows)} rows (column '{column}' in range [{min_value}, {max_value}])"
    }
```

### Example 2: Aggregation

```python
@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,
    description="Group CSV by column and compute aggregate"
)
async def aggregate(
    resource_uri: str,
    group_by: str,
    agg_column: str,
    agg_func: str = "sum",  # "sum", "avg", "count", "min", "max"
    in_place: bool = False
) -> dict:
    import csv
    from collections import defaultdict
    
    groups = defaultdict(list)
    with open(resource_uri) as f:
        for row in csv.DictReader(f):
            groups[row[group_by]].append(float(row[agg_column]))
    
    agg_map = {
        "sum": sum,
        "avg": lambda x: sum(x) / len(x),
        "count": len,
        "min": min,
        "max": max
    }
    
    results = {
        key: agg_map[agg_func](values)
        for key, values in groups.items()
    }
    
    output = resource_uri if in_place else resource_uri + ".agg"
    with open(output, 'w') as f:
        f.write(f"{group_by},{agg_func}_{agg_column}\n")
        for key, val in results.items():
            f.write(f"{key},{val}\n")
    
    return {
        "type": "text",
        "text": f"Aggregated: {len(results)} groups, {agg_func}({agg_column})"
    }
```

## Future Plans

### A2A (Agent-to-Agent) Support

Planned feature for passing data references between independent AI agents without serialization or context bloat:

```
Agent A:
  1. Reads data → stored as ref "abc123"
  2. Filters data → stored as ref "filtered_abc123"
  3. Passes ref to Agent B (lightweight reference only)

Agent B:
  1. Receives ref "filtered_abc123"
  2. Calls transformations on that reference
  3. No data serialization or re-transmission
```

This enables:
- Multi-agent pipelines with full data lineage
- Distributed data processing without context transfer
- Agent composition for complex workflows

## Troubleshooting

### "Resource not found in Station"

Ensure the `tool_call_id` (resource reference) is valid:
- Check the agent is using the correct ref from the previous operation
- Verify the Station backend is configured and running

### "Transform returns full data, expected summary"

Always return a summary, not the full dataset:

```python
# ❌ WRONG
return {"type": "text", "text": str(all_rows)}

# ✅ CORRECT
return {
    "type": "text",
    "text": f"Processed {len(rows)} rows, columns: {columns}"
}
```

### "in_place=True doesn't update Station"

The framework handles this automatically. Ensure your function writes to the same `resource_uri` when `in_place=True`:

```python
output_path = resource_uri if in_place else resource_uri + ".new"
save_data(transformed, output_path)
```

## Contributing

To add support for a new data type:

1. Create `orbit/transformations/examples/{datatype}_transforms.py`
2. Implement transformation functions with `@orbit_transformation_tool_mcp`
3. Add corresponding `DataType` enum value
4. Write tests in `tests/test_{datatype}_transforms.py`
5. Update this README with examples

## See Also

- [Launchpad Documentation](../README.md) — payload interception and masking
- [Station Documentation](../station.py) — storage backends
- [MCP Integration](../wrappers/mcp_wrapper.py) — how transformations integrate with MCP

## License

Same as Orbit. See LICENSE file in repository root.
