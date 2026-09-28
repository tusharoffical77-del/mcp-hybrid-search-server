"""
Tool implementations.

Each function takes a dict of arguments (as sent by the model) and
returns a JSON-serializable result. Every function catches its own
errors and returns a structured {"error": "..."} dict instead of
raising — the agent loop should never crash because a tool failed.
"""

import ast
import math
import operator
import os

# ---------------------------------------------------------------------------
# Safe calculator (no eval() — arithmetic + a whitelisted set of math
# functions only, via an AST whitelist)
# ---------------------------------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Only these specific function names are callable from an expression.
# This is a strict allowlist, not a general-purpose function call —
# anything not in this dict (e.g. os.system, __import__, eval) is
# rejected before it ever gets a chance to run.
_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "pow": math.pow,
    "abs": abs,
    "log": math.log,       # log(x) natural log, or log(x, base)
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "factorial": math.factorial,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        # Only allow calls where the function is a bare name (not an
        # attribute like os.system) AND that name is in our allowlist.
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct calls to whitelisted functions are allowed")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        args = [_eval_node(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[func_name](*args)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculator(args: dict) -> dict:
    expression = args.get("expression", "")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return {"result": result}
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": f"Could not evaluate expression: {e}"}


# ---------------------------------------------------------------------------
# Mock knowledge base
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE = {
    "pricing": "Pro plan is $20/month, billed monthly or annually.",
    "refund policy": "Refunds are available within 14 days of purchase, no questions asked.",
    "support hours": "Support is available Mon-Fri, 9am-6pm IST.",
    "api rate limits": "Free tier: 50 requests/min. Pro tier: 1000 requests/min.",
    "data retention": "User data is retained for 90 days after account deletion.",
}


def knowledge_base_lookup(args: dict) -> dict:
    query = args.get("query", "").lower().strip()
    if not query:
        return {"error": "No query provided"}

    # Simple keyword match against the mock KB
    matches = {
        topic: answer
        for topic, answer in _KNOWLEDGE_BASE.items()
        if query in topic or topic in query
    }

    if not matches:
        return {"result": "No matching entry found in knowledge base.", "matches": {}}

    return {"result": "Found matching entries.", "matches": matches}


# ---------------------------------------------------------------------------
# File reader (sandboxed to a workspace directory — no path traversal)
# ---------------------------------------------------------------------------

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "..", "workspace")


def read_file(args: dict) -> dict:
    filename = args.get("filename", "")
    if not filename:
        return {"error": "No filename provided"}

    # Prevent path traversal outside the workspace directory
    safe_name = os.path.basename(filename)
    path = os.path.abspath(os.path.join(WORKSPACE_DIR, safe_name))
    workspace_abs = os.path.abspath(WORKSPACE_DIR)

    if not path.startswith(workspace_abs):
        return {"error": "Invalid filename"}

    if not os.path.isfile(path):
        return {"error": f"File not found: {safe_name}"}

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Cap size returned to the model so one file can't blow the context
        if len(content) > 4000:
            content = content[:4000] + "\n...[truncated]"
        return {"result": content}
    except Exception as e:
        return {"error": f"Could not read file: {e}"}


# ---------------------------------------------------------------------------
# Registry mapping tool name -> function
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "knowledge_base_lookup": knowledge_base_lookup,
    "read_file": read_file,
}