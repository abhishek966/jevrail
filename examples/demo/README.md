# jevrail demo

A two-node LangGraph agent with [jevrail](https://pypi.org/project/jevrail/) on the input and the output.

```bash
cd examples/demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python agent.py
```

`python agent.py` uses a stand-in classifier, so it does not call TypeSafe. It checks three paths:

- a normal question is answered
- an instruction-override attempt is refused before either node runs
- an email address in the model answer is replaced with `[REDACTED]`

It checks each path with `invoke`, then again with `stream` in the `updates` and `values` modes, and confirms that token streaming is refused.

`requirements.txt` installs jevrail from PyPI. To run against this checkout instead:

```bash
pip install -e "../..[langgraph]"
```

To call Jev itself:

```bash
pip install "jevrail[langgraph,typesafe]"
export TYPESAFE_API_KEY=...
python agent.py --live
```
