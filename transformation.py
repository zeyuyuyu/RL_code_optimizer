"""
AST-based code transformations (6 actions).
"""
import ast

# 0. 删除 docstring
def remove_docstring(fn: ast.FunctionDef) -> bool:
    if (fn.body and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)):
        fn.body.pop(0); return True
    return False

# 1. 变量重命名（最长名 → 单字符）— 同步形参 
def rename_one_variable(fn: ast.FunctionDef) -> bool:
    names = {a.arg for a in fn.args.args}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
    longest = max((n for n in names if len(n) > 1), key=len, default=None)
    if not longest: return False
    new = next((c for c in "abcdefghijklmnopqrstuvwxyz" if c not in names), None)
    if not new: return False

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == longest: node.id = new
            return node
        def visit_arg(self, node):
            if node.arg == longest: node.arg = new
            return node
    Renamer().visit(fn)
    return True

# 2. 循环累加 → sum()
def _match_sum(body, i):
    if i + 2 >= len(body): return None
    a, loop, ret = body[i:i+3]
    if (isinstance(a, ast.Assign) and isinstance(loop, ast.For) and isinstance(ret, ast.Return)
        and len(a.targets) == 1 and isinstance(a.value, ast.Constant) and a.value.value == 0
        and isinstance(ret.value, ast.Name) and ret.value.id == a.targets[0].id):
        return loop.iter
    return None

def transform_loop_sum(fn):
    body = fn.body
    for i in range(len(body)-2):
        it = _match_sum(body, i)
        if it is not None:
            body[i:i+3] = [ast.Return(value=ast.Call(func=ast.Name(id="sum", ctx=ast.Load()),
                                                     args=[it], keywords=[]))]
            ast.fix_missing_locations(fn); return True
    return False

# 3. 循环取最大 → max()
def _match_max(body, i):
    if i + 2 >= len(body): return None
    a, loop, ret = body[i:i+3]
    if (isinstance(a, ast.Assign) and isinstance(loop, ast.For) and isinstance(ret, ast.Return)
        and isinstance(a.value, ast.Subscript) and isinstance(ret.value, ast.Name)
        and ret.value.id == a.targets[0].id):
        return a.value.value
    return None

def transform_loop_max(fn):
    body = fn.body
    for i in range(len(body)-2):
        it = _match_max(body, i)
        if it is not None:
            body[i:i+3] = [ast.Return(value=ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                                                     args=[it], keywords=[]))]
            ast.fix_missing_locations(fn); return True
    return False

# 4. if-return True/False → 布尔表达式
def transform_if_return_bool(fn):
    new, changed = [], False
    for n in fn.body:
        if (isinstance(n, ast.If) and len(n.body)==1 and len(n.orelse)==1
            and isinstance(n.body[0], ast.Return) and isinstance(n.orelse[0], ast.Return)
            and isinstance(n.body[0].value, ast.Constant) and isinstance(n.orelse[0].value, ast.Constant)):
            t, f = n.body[0].value.value, n.orelse[0].value.value
            if t is True and f is False:
                new.append(ast.Return(value=n.test)); changed=True; continue
            if t is False and f is True:
                new.append(ast.Return(value=ast.UnaryOp(op=ast.Not(), operand=n.test))); changed=True; continue
        new.append(n)
    if changed:
        fn.body = new; ast.fix_missing_locations(fn)
    return changed

# 5. append 循环 → 列表推导式
def transform_list_append(fn):
    body = fn.body
    for i in range(len(body)-2):
        ass, loop, ret = body[i:i+3]
        if (isinstance(ass, ast.Assign) and isinstance(loop, ast.For)
            and isinstance(ret, ast.Return) and isinstance(ass.value, ast.List)
            and ass.value.elts == []):
            lst = ass.targets[0].id
            call = loop.body[0].value if loop.body else None
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.value.id == lst and call.func.attr == "append"):
                comp = ast.ListComp(elt=call.args[0],
                                    generators=[ast.comprehension(target=loop.target,
                                                                  iter=loop.iter,
                                                                  ifs=[], is_async=0)])
                body[i:i+3] = [ast.Return(value=comp)]
                ast.fix_missing_locations(fn); return True
    return False

# 动作列表
ACTIONS = [remove_docstring, rename_one_variable, transform_loop_sum,
           transform_loop_max, transform_if_return_bool, transform_list_append]
