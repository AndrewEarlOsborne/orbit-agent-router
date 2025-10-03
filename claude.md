Initial notes: to use and replace:

1. artifact == payload. They are synonymous, and describe the non-metadata data returned from a LLM agent's toolcall
2. I want this solution to be bundled in a python module. There should be constructors to enable customization of Launchpads for custom summarization. 
3. I want to be able to wrap locally declared tools in orbit decorators, or to re-construct them a orbit tools. For a given type of tool input, ie MCP Client, Langgraph, the Launchpad should only be passed one type, there may be others, but it needs diffirent logic in the function defenition to handle and switch case to the relevant type passed into the tool wrapper. 
   
Other formatting instructions:
1. DO NOT USE EMOJIS
2. If implementation still needs to be completed or specified, mark as #TODO: <description≥
3. Use typesafe function declarations, item declarations, and else wherever possible.


Inside this line in MCP: ""result = await self.session.call_tool(tool_name, tool_args)"" is where we need to embed logic that executes on the wrapped tools.

Launchpad should start with the following:

1. wrap tools into orbit tools

the wrapped tools will then change the execution time behavior of the tool by wrapping the functionality. This will not impact the tools semantics or functionality, and will only operate on the output to do the following given a result result containing result.content:

1. result will be checked if it is a tool message, and is connected to a valid tool. If not, raise errors as would normally occur.
2. result.content will be copied, and saved to the cache under the tool call id, and include the entire result
3. result.content will be operated on to _generate_summary, which is a customizable method for custom spaceports, but by default will look at each kv pair in the content. If longer than 2048 chars, it will attempt to mask it by returning a summary of the value, including the type. We assume that keys are unique identifiers.


## For summary do:
In MCP, a tool call returns a JSON‑RPC response whose result.content is a list of content objects. 
Model Context Protocol
+2
Model Context Protocol
+2

In Python, result.content is parsed into a list[dict].

Each dict has a "type" key, and fields according to that type (e.g. "text", "data", "resource").

You iterate the list and branch on type to extract the actual data you care about.