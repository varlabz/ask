import json
from collections.abc import Callable

from langfuse._client.datasets import DatasetItemClient

from ask.core.agent import AgentASK
from ask.core.config import Config


def serialize_config(cfg):
    def default(obj):
        if isinstance(obj, type):
            return (
                f"{obj.__module__}.{obj.__name__}"
                if hasattr(obj, "__module__")
                else obj.__name__
            )
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(cfg.model_dump(), indent=2, default=default)


async def task_executor(
    cfg: Config,
    item: DatasetItemClient,
    callback: Callable[[], None],
):
    callback()
    agent = AgentASK.create_from_config(cfg)
    result = await agent.run(item.input)
    return str(result)


async def task_executor_agent(
    agent: AgentASK, item: DatasetItemClient, callback: Callable[[], None]
):
    callback()
    result = await agent.run(item.input)
    return result


function_tools = [
    {
        "name": "echo",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo"}
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        "description": "Echoes back the input",
        "outer_typed_dict_key": None,
        "strict": True,
        "sequential": False,
        "kind": "function",
        "metadata": {"meta": None, "annotations": None, "output_schema": None},
    },
    {
        "name": "add",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        "description": "Adds two numbers",
        "outer_typed_dict_key": None,
        "strict": True,
        "sequential": False,
        "kind": "function",
        "metadata": {"meta": None, "annotations": None, "output_schema": None},
    },
    {
        "name": "longRunningOperation",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "default": 10,
                    "description": "Duration of the operation in seconds",
                },
                "steps": {
                    "type": "number",
                    "default": 5,
                    "description": "Number of steps in the operation",
                },
            },
            "additionalProperties": False,
        },
        "description": "Demonstrates a long running operation with progress updates",
        "outer_typed_dict_key": None,
        "strict": False,
        "sequential": False,
        "kind": "function",
        "metadata": {"meta": None, "annotations": None, "output_schema": None},
    },
    {
        "name": "printEnv",
        "parameters_json_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "description": "Prints all environment variables, helpful for debugging MCP server configuration",
        "outer_typed_dict_key": None,
        "strict": False,
        "sequential": False,
        "kind": "function",
        "metadata": {"meta": None, "annotations": None, "output_schema": None},
    },
    {
        "name": "sampleLLM",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to send to the LLM",
                },
                "maxTokens": {
                    "type": "number",
                    "default": 100,
                    "description": "Maximum number of tokens to generate",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        "description": "Samples from an LLM using MCP's sampling feature",
        "outer_typed_dict_key": None,
        "strict": False,
        "sequential": False,
        "kind": "function",
        "metadata": {"meta": None, "annotations": None, "output_schema": None},
    },
    {
        "name": "getTinyImage",
        "parameters_json_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "description": "Returns the MCP_TINY_IMAGE",
        "outer_typed_dict_key": None,
        "strict": False,
        "sequential": False,
        "kind": "function",
        "metadata": {"meta": None, "annotations": None, "output_schema": None},
    },
    {
        "name": "annotatedMessage",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "messageType": {
                    "type": "string",
                    "enum": ["error", "success", "debug"],
                    "description": "Type of message to demonstrate different annotation patterns",
                },
                "includeImage": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to include an example image",
                },
            },
            "required": ["messageType"],
            "additionalProperties": False,
        },
        "description": "Demonstrates how annotations can be used to provide metadata about content",
        "outer_typed_dict_key": None,
        "strict": False,
        "sequential": False,
        "kind": "function",
        "metadata": {"meta": None, "annotations": None, "output_schema": None},
    },
    {
        "name": "getResourceReference",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "resourceId": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "ID of the resource to reference (1-100)",
                }
            },
            "required": ["resourceId"],
            "additionalProperties": False,
        },
        "description": "Returns a resource reference that can be used by MCP clients",
        "outer_typed_dict_key": None,
        "strict": True,
        "sequential": False,
        "kind": "function",
        "metadata": {"meta": None, "annotations": None, "output_schema": None},
    },
    {
        "name": "getResourceLinks",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "Number of resource links to return (1-10)",
                }
            },
            "additionalProperties": False,
        },
        "description": "Returns multiple resource links that reference different types of resources",
        "outer_typed_dict_key": None,
        "strict": False,
        "sequential": False,
        "kind": "function",
        "metadata": {"meta": None, "annotations": None, "output_schema": None},
    },
    {
        "name": "structuredContent",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "minLength": 1,
                    "description": "City name or zip code",
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        "description": "Returns structured content along with an output schema for client data validation",
        "outer_typed_dict_key": None,
        "strict": False,
        "sequential": False,
        "kind": "function",
        "metadata": {
            "meta": None,
            "annotations": None,
            "output_schema": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "description": "Temperature in celsius",
                    },
                    "conditions": {
                        "type": "string",
                        "description": "Weather conditions description",
                    },
                    "humidity": {
                        "type": "number",
                        "description": "Humidity percentage",
                    },
                },
                "required": ["temperature", "conditions", "humidity"],
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#",
            },
        },
    },
]
