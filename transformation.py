
---

## 3 📄 transformations.py

```python
"""
transformations.py
AST-based code transformation rules for code optimization.
"""
import ast

# ---------- 基础变换 ----------

def remove_docstring(func_node):
    if (func_node.body
        and isinstance(func_node.body[0], ast.Expr)
        and isinstance(func_node.body[0].value, ast.Constant)
        and isinstance(func_node.body[0].value.value, str)):
        func_node.body.pop(0)
        return True
    return False

def rename_one_variable(func_node):
    names = {arg.arg for arg in func_node.args.args}
    for n in ast.walk(func_node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
    longest = max((v for v in names if len(v) > 1), key=len, default=None)
    if not longest:
        return False
    new_name = next((c for c in "abcdefghijklmnopqrstuvwxyz" if c not in names), None)
    if not new_name:
        return False

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == longest:
                return ast.copy_location(ast.Name(id=new_name, ctx=node.ctx), node)
            return node
    Renamer().visit(func_node)
    return True

# ---------- 循环 → 内建函数 ----------

def _match_sum_loop(body, i):
    if i + 2 >= len(body):
        return None
    assign, loop, ret = body[i:i+3]
    if (isinstance(assign, ast.Assign) and isinstance(loop, ast.For)
        and isinstance(ret, ast.Return) and len(assign.targets) == 1
        and isinstance(assign.targets[0], ast.Name)
        and isinstance(assign.value, ast.Constant) and assign.value.value == 0
        and isinstance(ret.value, ast.Name)
        and ret.value.id == assign.targets[0].id and len(loop.body) == 1):
        return loop.iter
    return None

def transform_loop_sum(func_node):
    body = func_node.body
    for i in range(len(body) - 2):
        iterable = _match_sum_loop(body, i)
        if iterable is not None:
            body[i:i+3] = [ast.Return(
                value=ast.Call(func=ast.Name(id="sum", ctx=ast.Load()),
                               args=[iterable], keywords=[]))]
            ast.fix_missing_locations(func_node)
            return True
    return False

def _match_max_loop(body, i):
    if i + 2 >= len(body):
        return None
    assign, loop, ret = body[i:i+3]
    if (isinstance(assign, ast.Assign) and isinstance(loop, ast.For)
        and isinstance(ret, ast.Return) and len(assign.targets) == 1
        and isinstance(assign.targets[0], ast.Name)
        and isinstance(assign.value, ast.Subscript)
        and isinstance(ret.value, ast.Name)
        and ret.value.id == assign.targets[0].id
        and any(isinstance(n, ast.If) for n in loop.body)):
        return assign.value.value
    return None

def transform_loop_max(func_node):
    body = func_node.body
    for i in range(len(body) - 2):
        iterable = _match_max_loop(body, i)
        if iterable is not None:
            body[i:i+3] = [ast.Return(
                value=ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                               args=[iterable], keywords=[]))]
            ast.fix_missing_locations(func_node)
            return True
    return False

# ---------- 逻辑简化 ----------

def transform_if_return_bool(func_node):
    new_body, changed = [], False
    for node in func_node.body:
        if (isinstance(node, ast.If) and len(node.body) == 1 and len(node.orelse) == 1
            and isinstance(node.body[0], ast.Return) and isinstance(node.orelse[0], ast.Return)
            and isinstance(node.body[0].value, ast.Constant) and isinstance(node.orelse[0].value, ast.Constant)):
            rb, ro = node.body[0].value.value, node.orelse[0].value.value
            if rb is True and ro is False:
                new_body.append(ast.Return(value=node.test)); changed = True; continue
            if rb is False and ro is True:
                new_body.append(ast.Return(
                    value=ast.UnaryOp(op=ast.Not(), operand=node.test))); changed = True; continue
        new_body.append(node)
    if changed:
        func_node.body = new_body
        ast.fix_missing_locations(func_node)
    return changed

# ---------- 列表推导 ----------

def transform_list_append(func_node):
    body = func_node.body
    for i in range(len(body) - 2):
        assign, loop, ret = body[i:i+3]
        if (isinstance(assign, ast.Assign) and len(assign.targets)==1
            and isinstance(assign.value, ast.List) and assign.value.elts == []):
            list_var = assign.targets[0].id
            if (isinstance(loop, ast.For) and len(loop.body) == 1
                and isinstance(loop.body[0], ast.Expr)
                and isinstance(loop.body[0].value, ast.Call)):
                call = loop.body[0].value
                if (isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == list_var and call.func.attr == "append"):
                    if (isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
                        and ret.value.id == list_var):
                        elem = call.args[0]
                        comp = ast.ListComp(elt=elem,
                                            generators=[ast.comprehension(
                                                target=loop.target, iter=loop.iter,
                                                ifs=[], is_async=0)])
                        body[i:i+3] = [ast.Return(value=comp)]
                        ast.fix_missing_locations(func_node)
                        return True
    return False

# 动作列表（索引即动作编号）
ACTIONS = [
    remove_docstring,
    rename_one_variable,
    transform_loop_sum,
    transform_loop_max,
    transform_if_return_bool,
    transform_list_append,
]
