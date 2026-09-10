from smrt.eval_tools import _extract_model_tool_call, _schema_valid

_WEATHER_SCHEMA = {
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


# If this fails: a well-formed <|tool_call|>...<|/tool_call|> block in the
# model's generated text is no longer being parsed back into a
# (name, arguments) pair — the exact structure eval_tools scores against.
def test_extract_model_tool_call_parses_valid_block():
    text = '<|tool_call|>{"name":"get_weather","arguments":{"location":"Paris"}}<|/tool_call|>'
    result = _extract_model_tool_call(text)
    assert result == ("get_weather", {"location": "Paris"})


# If this fails: text with no tool-call markers at all stopped being
# treated as "no call made" (returning None), which would otherwise crash
# or falsely report a match downstream.
def test_extract_model_tool_call_returns_none_without_markers():
    assert _extract_model_tool_call("just some plain assistant text") is None


# If this fails: a tool-call block with malformed/incomplete JSON inside
# it is no longer being rejected (returning None) instead of raising or
# silently fabricating a partial result.
def test_extract_model_tool_call_returns_none_for_malformed_json():
    text = "<|tool_call|>{not valid json<|/tool_call|>"
    assert _extract_model_tool_call(text) is None


# If this fails: a parsed JSON object missing "name" or "arguments"
# (e.g. the model emitted a differently-shaped blob inside the tool-call
# markers) stopped being rejected (returning None) and instead raised a
# KeyError when the caller indexed into it.
def test_extract_model_tool_call_returns_none_when_missing_required_keys():
    text = '<|tool_call|>{"name":"get_weather"}<|/tool_call|>'
    assert _extract_model_tool_call(text) is None
    text2 = '<|tool_call|>{"arguments":{"location":"Paris"}}<|/tool_call|>'
    assert _extract_model_tool_call(text2) is None


# If this fails: arguments that satisfy the tool's declared JSON Schema
# are being rejected by _schema_valid (a false negative that would
# under-count correct tool calls in the eval).
def test_schema_valid_accepts_conforming_arguments():
    assert _schema_valid("get_weather", {"location": "Paris"}, [_WEATHER_SCHEMA]) is True


# If this fails: arguments missing a required field, or a wrong-typed
# field, are being accepted by _schema_valid (a false positive that would
# over-count incorrect tool calls in the eval).
def test_schema_valid_rejects_missing_required_field():
    assert _schema_valid("get_weather", {}, [_WEATHER_SCHEMA]) is False


def test_schema_valid_rejects_wrong_type():
    assert _schema_valid("get_weather", {"location": 5}, [_WEATHER_SCHEMA]) is False


# If this fails: a tool name with no matching schema in the known list
# stopped being treated as vacuously valid (nothing to check against),
# which would incorrectly fail every call to an unknown/undeclared tool.
def test_schema_valid_true_when_no_schema_for_name():
    assert _schema_valid("unknown_tool", {"anything": 1}, [_WEATHER_SCHEMA]) is True
