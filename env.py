"""
env.py  – mix heuristic & learned reward; uses CodeTransformationAgent.
"""
import ast, time, random, numpy as np
from agent import CodeTransformationAgent
from reward_model import PairwiseRewardModel

class CodeOptimizeEnv:
    def __init__(self, funcs, cfg):
        self.funcs = funcs
        self.cfg = cfg
        self.ct = CodeTransformationAgent()
        self.rm = PairwiseRewardModel()
        self.max_steps = cfg.get("max_steps", 10)
        self._prepare_refs()

    # -------- dataset --------
    def _prepare_refs(self):
        self.names = list(self.funcs)
        self.ref_funcs, self.tests, self.perfs = {}, {}, {}
        for n, code in self.funcs.items():
            ns={"print": lambda *a, **k: None}
            exec(code, ns)
            self.ref_funcs[n] = ns[n]
            if n in ("sum_list","double_list"):
                self.tests[n]=[([1,2,3],), ([],)]
                self.perfs[n]=[([i for i in range(5000)],)]
            elif n=="max_list":
                self.tests[n]=[([3,1,2],), ([42],)]
                self.perfs[n]=[([i for i in range(5000)],)]
            elif n=="check_positive":
                self.tests[n]=[(5,), (-1,)]
                self.perfs[n]=[(9999999,)]
            else:
                self.tests[n]=[("Bob",)]
                self.perfs[n]=[("Bob"*4,)]

    # -------- RL interface --------
    def reset(self):
        self.idx = random.randrange(len(self.names))
        self.name = self.names[self.idx]
        self.code = self.funcs[self.name]
        self.steps = 0
        return self._obs()

    def _obs(self):
        return np.array([len(self.code)/100, self.idx/len(self.names)],
                        dtype=np.float32)

    def step(self, action):
        self.steps += 1
        new_code, changed = self.ct.propose(self.code, action)

        if not changed:
            done = self.steps >= self.max_steps
            return self._obs(), -1.0, done, {"heu": 0.0, "lr": 0.0}

        # correctness
        if not self._correct(new_code):
            return self._obs(), -10.0, True, {}

        # heuristic reward
        heu = 0.0
        if changed:
            l_prev, l_new = len(self.code), len(new_code)
            scale = 1.0 if action == 1 else 5.0
            heu = scale * float(l_prev - l_new)

        # learned reward
        lr = self.rm.score(self.code, new_code)
        self.rm.add(self.code, new_code, heu > 0)
        self.code = new_code

        mixed = self.cfg["alpha"]*heu + self.cfg["beta"]*lr
        done = self.steps >= self.max_steps or (heu > 0 and action != 1)
        return self._obs(), mixed, done, {"heu": heu, "lr": lr}

    # -------- helpers --------
    def _compile(self, code):
        ns={"print":lambda *a,**k:None}; exec(code, ns); return ns[self.name]

    def _rt(self, code):
        fn = self._compile(code)
        args = self.perfs[self.name][0]
        times = []
        for _ in range(5):
            s = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - s)
        return sum(times) / len(times)

    def _correct(self, code):
        ref = self.ref_funcs[self.name]; new = self._compile(code)
        for case in self.tests[self.name]:
            try: r_ref, e_ref = ref(*case), None
            except Exception as e: e_ref = type(e)
            try: r_new, e_new = new(*case), None
            except Exception as e: e_new = type(e)
            if e_ref or e_new:
                if e_ref != e_new: return False
            elif r_ref != r_new: return False
        return True
