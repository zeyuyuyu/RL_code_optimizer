"""
env.py
Gym-like environment for RL code optimizer.
"""
import ast, time, random
import numpy as np
from transformations import ACTIONS

class CodeOptimizeEnv:
    def __init__(self, functions_dict, max_steps=10):
        self.functions = functions_dict
        self.max_steps = max_steps
        self.action_space = list(range(len(ACTIONS)))
        self._prepare_refs()

    # ---------- 测试&性能输入 ----------
    def _prepare_refs(self):
        self.original, self.tests, self.perfs = {}, {}, {}
        for name, code in self.functions.items():
            ns = {}; exec(code, ns)
            self.original[name] = ns[name]
            if name in ('sum_list', 'max_list', 'double_list'):
                self.tests[name] = [([1,2,3],), ([],)]
                self.perfs[name] = [([i for i in range(10000)],)]
            elif name == 'check_positive':
                self.tests[name] = [(-1,), (0,), (5,)]
                self.perfs[name] = [(9999999,)]
            elif name == 'greet':
                self.tests[name] = [("Bob",)]
                self.perfs[name] = [("X"*16,)]
            else:
                self.tests[name] = [()]
                self.perfs[name] = [()]

    # ---------- 环境交互 ----------
    def reset(self):
        self.name = random.choice(list(self.functions))
        self.tree = ast.parse(self.functions[self.name])
        self.node = self.tree.body[0]
        self.steps = 0
        return self._feature()

    def _feature(self):
        code_len = len(ast.unparse(self.node).replace(" ", "").replace("\n", "")) / 100
        doc = 1 if (self.node.body and isinstance(self.node.body[0], ast.Expr)
                    and isinstance(self.node.body[0].value, ast.Constant)
                    and isinstance(self.node.body[0].value.value, str)) else 0
        vars_cnt = len({arg.arg for arg in self.node.args.args})
        return np.array([code_len, doc, vars_cnt], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        applied = ACTIONS[action](self.node)
        if not self._correct():
            return self._feature(), -10.0, True, {}
        reward = 0.0
        if applied:
            len_before, t_before = self._code_len(), self._perf()
            len_after,  t_after  = self._code_len(), self._perf()
            reward = (len_before - len_after) + 20*((t_before/t_after)-1 if t_after>0 else 0)
        done = self.steps >= self.max_steps
        return self._feature(), reward, done, {}

    # ---------- 工具函数 ----------
    def _code_len(self):
        return len(ast.unparse(self.node).replace(" ", "").replace("\n", ""))

    def _compile(self):
        ns = {"print": lambda *a, **kw: None}
        exec(compile(ast.Module([self.node], []), "<ast>", "exec"), ns)
        return ns[self.name]

    def _correct(self):
        f = self._compile()
        for case in self.tests[self.name]:
            if f(*case) != self.original[self.name](*case):
                return False
        return True

    def _perf(self):
        f = self._compile()
        args = self.perfs[self.name][0]
        s = time.time()
        f(*args)
        return time.time() - s
