import json
from mlx_lm.server import ToolCallFormatter, process_message_content
from mlx_lm.tool_parsers.json_tools import parse_tool_call

formatter = ToolCallFormatter(parse_tool_call, [])
valid = '{"name":"edit_file","arguments":{"new":"line one\nline two"}}'
parsed = formatter([valid])
assert len(parsed) == 1 and json.loads(parsed[0]['function']['arguments'])['new'] == 'line one\nline two'
assert formatter.emitted == 1 and formatter.take_failed_text() == ''
invalid = '{"name":"edit_file","arguments":{"new":"unescaped "quote""}}'
failed = ToolCallFormatter(parse_tool_call, [])
assert failed([invalid]) == [] and failed.emitted == 0
assert failed.take_failed_text() == '<tool_call>' + invalid + '</tool_call>'
assert failed.take_failed_text() == ''
print('Patched parser accepts literal newlines and preserves invalid quote text')

messages = [{'role':'assistant','content':None,'reasoning':'prior thought'}]
process_message_content(messages)
assert messages[0]['reasoning_content'] == 'prior thought'
print('Patched server preserves prior reasoning for the template')
