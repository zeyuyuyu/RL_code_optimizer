"""
agent.py  – CodeTransformationAgent
Wraps ACTIONS so policy can call propose(code, action_id).
"""
import ast
from transformations import ACTIONS

class CodeTransformationAgent:
    def __init__(self):
        self.n_actions = len(ACTIONS)

    def propose(self, code_str: str, action_id: int):
        if action_id < 0 or action_id >= self.n_actions:
            raise ValueError("invalid action id")
        tree = ast.parse(code_str)
        fn = tree.body[0]
        changed = ACTIONS[action_id](fn)
        return ast.unparse(fn) if changed else code_str, changed
