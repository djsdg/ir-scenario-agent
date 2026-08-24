from ir_agent.tools import ToolSpec


def create_plugin(context):
    def plugin_info(_args):
        return {
            "plugin": "example",
            "message": "这是一个可被动态发现的本地插件工具。",
            "user_id": context.user_id,
        }

    return [
        ToolSpec(
            name="example_plugin_info",
            description="Return a small health/info response from the example plugin.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
            input_model=None,
            handler=plugin_info,
        )
    ]
