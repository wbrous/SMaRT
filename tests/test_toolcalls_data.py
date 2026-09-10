import json

import pytest

from smrt.data.tokenizer import agent_tokenizer
from smrt.data.toolcalls import (
    _extract_json_objects,
    format_trajectory,
    parse_chat_turns,
    parse_system,
)

_SYSTEM_ONE_TOOL = (
    "SYSTEM: You are a helpful assistant with access to the following "
    "functions. Use them if required -\n"
    '{\n    "name": "get_weather",\n    "parameters": {"type": "object"}\n}\n\n'
)

_SYSTEM_TWO_TOOLS = (
    "SYSTEM: You are helpful. functions. Use them if required -\n"
    '{"name": "a", "parameters": {}}\n{"name": "b", "parameters": {}}\n'
)

_CHAT_WITH_CALL = (
    "USER: What is the weather in Paris? <|endoftext|>\n\n\n"
    "ASSISTANT: <functioncall> {\"name\": \"get_weather\", \"arguments\": '{\"location\": \"Paris\"}'} <|endoftext|>\n\n\n"
    "FUNCTION RESPONSE: {\"temp\": 20}\n\n\n"
    "ASSISTANT: It is 20 degrees in Paris. <|endoftext|>\n\n\n"
)

_CHAT_MALFORMED_CALL = (
    "USER: hi <|endoftext|>\n\n\n"
    "ASSISTANT: <functioncall> not-json-at-all <|endoftext|>\n\n\n"
)

_CHAT_INVALID_INNER_ARGUMENTS_JSON = (
    "USER: hi <|endoftext|>\n\n\n"
    "ASSISTANT: <functioncall> {\"name\": \"foo\", \"arguments\": '{not valid json}'} <|endoftext|>\n\n\n"
)

_SYSTEM_INVALID_SCHEMA_JSON = (
    "SYSTEM: You are helpful. functions. Use them if required -\nnot json schemas at all\n"
)


# If this fails: _extract_json_objects stopped raising a bare
# json.JSONDecodeError on genuinely non-JSON input (e.g. it started
# swallowing the error itself) -- callers rely on catching this
# exception at their own boundary rather than _extract_json_objects
# silently returning something misleading.
def test_extract_json_objects_raises_on_non_json_text():
    with pytest.raises(json.JSONDecodeError):
        _extract_json_objects("not json at all")


# If this fails: the two-JSON-blob extraction stopped handling
# back-to-back concatenated objects with no delimiter (raw_decode
# position tracking regressed).
def test_extract_json_objects_handles_multiple_concatenated_blobs():
    text = '{"a": 1}\n{"b": 2}'
    objs = _extract_json_objects(text)
    assert objs == [{"a": 1}, {"b": 2}]


# If this fails: an empty/whitespace-only string is no longer handled
# (should yield zero objects, not raise).
def test_extract_json_objects_empty_string_yields_nothing():
    assert _extract_json_objects("   \n  ") == []


# If this fails: parse_system stopped extracting exactly one tool schema
# for a single-tool row, or the "SYSTEM: " prefix is leaking into the
# intro text.
def test_parse_system_single_tool():
    intro, schemas = parse_system(_SYSTEM_ONE_TOOL)
    assert not intro.startswith("SYSTEM:")
    assert len(schemas) == 1
    assert schemas[0]["name"] == "get_weather"


# If this fails: parse_system stopped extracting all N concatenated tool
# schemas for a multi-tool row (only picked up the first one).
def test_parse_system_multiple_tools():
    _intro, schemas = parse_system(_SYSTEM_TWO_TOOLS)
    assert [s["name"] for s in schemas] == ["a", "b"]


# If this fails: a system block with no "functions. Use them if required"
# marker (no tools offered) stopped being treated as tool-free text.
def test_parse_system_no_tools_marker():
    intro, schemas = parse_system("SYSTEM: just be nice.")
    assert intro == "just be nice."
    assert schemas == []


# If this fails: the "\n\n\n"-separated turn splitter stopped correctly
# tagging each segment's role (user/assistant/tool_result), or started
# leaking the role prefix into the body text.
def test_parse_chat_turns_roles_and_bodies():
    turns = parse_chat_turns(_CHAT_WITH_CALL)
    roles = [role for role, _ in turns]
    assert roles == ["user", "assistant", "tool_result", "assistant"]
    assert turns[0][1] == "What is the weather in Paris?"
    assert not turns[1][1].startswith("ASSISTANT:")


# If this fails: format_trajectory stopped correctly round-tripping a
# glaive-shaped row's function call into the <|tool_call|> special-token
# wrapping, or the outer single-quote-wrapped arguments are no longer
# being unwrapped into a valid nested JSON object.
def test_format_trajectory_extracts_function_call():
    tokenizer = agent_tokenizer()
    result = format_trajectory(_SYSTEM_ONE_TOOL, _CHAT_WITH_CALL, tokenizer)
    assert result is not None
    ids, mask = result
    assert len(ids) == len(mask)
    text = tokenizer.decode(ids)
    assert '<|tool_call|>{"name":"get_weather","arguments":{"location":"Paris"}}<|/tool_call|>' in text


# If this fails: the loss mask is not actually zero on system/user/
# tool_result content and one on assistant content — the core contract
# the SFT loop's ignore_index masking depends on.
def test_format_trajectory_mask_only_covers_assistant_turns():
    tokenizer = agent_tokenizer()
    ids, mask = format_trajectory(_SYSTEM_ONE_TOOL, _CHAT_WITH_CALL, tokenizer)
    assert any(m == 1 for m in mask)
    assert any(m == 0 for m in mask)

    # Every position tagged as loss-target (mask==1) must fall strictly
    # after the first <|assistant|> token, never inside the masked
    # system/user preamble.
    assistant_id = tokenizer.encode("<|assistant|>")[0]
    first_assistant_pos = ids.index(assistant_id)
    unmasked_positions = [i for i, m in enumerate(mask) if m == 1]
    assert min(unmasked_positions) >= first_assistant_pos


# If this fails: a functioncall body that doesn't match the expected
# single-quoted-arguments shape is silently emitted as a training target
# instead of causing the whole row to be dropped (returning None).
def test_format_trajectory_returns_none_for_malformed_call():
    tokenizer = agent_tokenizer()
    result = format_trajectory(_SYSTEM_ONE_TOOL, _CHAT_MALFORMED_CALL, tokenizer)
    assert result is None


# If this fails: a <functioncall> whose inner `arguments` string matches
# the expected outer shape but isn't valid JSON on its own stops being
# dropped (returning None) and instead crashes format_trajectory with an
# uncaught json.JSONDecodeError -- which would kill the entire
# multi-hundred-thousand-row load_toolcall_sft_stream on one bad row.
# Regression test for a real bug found and fixed this session.
def test_format_trajectory_returns_none_for_invalid_inner_arguments_json():
    tokenizer = agent_tokenizer()
    result = format_trajectory(_SYSTEM_ONE_TOOL, _CHAT_INVALID_INNER_ARGUMENTS_JSON, tokenizer)
    assert result is None


# If this fails: a system block whose tool-schema section isn't valid
# JSON stops being dropped (returning None) and instead crashes
# format_trajectory with an uncaught json.JSONDecodeError from
# parse_system -- the same crash-the-whole-stream failure mode as above,
# just triggered from the system side instead of the assistant side.
# Regression test for a real bug found and fixed this session.
def test_format_trajectory_returns_none_for_invalid_system_schema_json():
    tokenizer = agent_tokenizer()
    result = format_trajectory(_SYSTEM_INVALID_SCHEMA_JSON, _CHAT_WITH_CALL, tokenizer)
    assert result is None
