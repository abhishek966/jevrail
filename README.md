# jevrail

Input and output guardrails for LangGraph agents. [Jev](https://typesafe.ai/) scores each check and returns a probability. jevrail applies your thresholds and decides whether to allow, block, redact, or flag.

Jev does not write the refusal. One call per stage asks every enabled check.

## Install

```bash
pip install jevrail
pip install "jevrail[langgraph,typesafe]"
```

Set `TYPESAFE_API_KEY` before a live agent runs. Tests and the decision engine accept a stand-in classifier, so the package does not call TypeSafe unless you use the default one.

## Place the guard where you need it

Checks you leave out do not run. The default policy guards user input and the model answer. Tool calls and tool results stay off until you list them.

```python
from jevrail import GuardPolicy, JevRail, guard

policy = GuardPolicy(
    stages={
        "input": ["prompt_injection", "jailbreak", "content_safety"],
        "output": ["content_safety", "pii", "secrets", "system_prompt_leak"],
        # "tool_call": ["excessive_agency"],
        # "tool_result": ["indirect_injection"],
    },
    thresholds={"prompt_injection": 0.7, "pii": 0.8},
    actions={"pii": "redact", "prompt_injection": "block"},
    on_block="refuse",   # refuse | raise
    fail_mode="closed",  # closed | open
)
```

`create_agent` middleware:

```python
from langchain.agents import create_agent

agent = create_agent(model, tools=tools, middleware=[JevRail(policy)])
```

A compiled graph, checked on the way in and on the way out:

```python
app = guard(compiled, policy)
app.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

`app.stream` and `app.astream` run the whole graph, check the final answer, then yield the chunks, with redactions applied or a single refusal in their place. They accept `stream_mode` `"values"` or `"updates"`, alone or as a list, and `subgraphs=True`. Token streaming (`"messages"`) and `"custom"` raise `ValueError`, because those chunks would reach the caller before the output check.

A custom graph, on the edges you choose:

```python
from jevrail import input_guard, output_guard

graph.add_node("input_guard", input_guard(policy))
graph.add_node("output_guard", output_guard(policy))
```

`import jevrail.auto` is optional. It patches `create_agent` and messages-based `StateGraph.compile` using `JEVRAIL_POLICY` when that variable points at a YAML file. Import it before you build the agent.

## What each stage can check

| Check | Typical stage | Action |
| --- | --- | --- |
| `prompt_injection` | input, tool result | block |
| `jailbreak` | input | block |
| `content_safety` | input, output | block |
| `system_prompt_extraction` | input | block |
| `system_prompt_leak` | output | block |
| `pii` | output | redact |
| `secrets` | output | redact |
| `data_exfiltration` | output | block |
| `excessive_agency` | tool call | block |
| `indirect_injection` | tool result | block |
| `topic_scope` | input or output, with `allowed_topics` | block |

`denied_tools` blocks a tool by name before it runs, without a Jev call. It applies whether or not `tool_call` is listed in `stages`.

When `pii` or `secrets` is enabled, a local scanner masks the emails, SSNs, card numbers, API keys, and private keys it recognizes on every pass, whatever Jev scores. Jev covers what a pattern cannot see. If Jev flags PII or a secret and the scanner finds no span to mask, the action becomes a block.

If the TypeSafe API fails, `fail_mode="closed"` blocks the turn. `fail_mode="open"` lets it through.

## Publish

Do not create an empty project on PyPI. The name is claimed by the first upload.

1. Register at pypi.org and test.pypi.org and turn on 2FA.
2. `uv build` or `python -m build`.
3. `twine upload --repository testpypi dist/*` and install it in a clean virtualenv.
4. `twine upload dist/*` to create the `jevrail` project.

## Development

```bash
pip install -e ".[dev,typesafe]"
pytest
```
