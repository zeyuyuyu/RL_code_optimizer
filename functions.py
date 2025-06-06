
functions = {
"sum_list": """
def sum_list(lst):
    total = 0
    for x in lst:
        total += x
    return total
""",
"max_list": """
def max_list(lst):
    current_max = lst[0]
    for x in lst:
        if x > current_max:
            current_max = x
    return current_max
""",
"check_positive": """
def check_positive(x):
    if x > 0:
        return True
    else:
        return False
""",
"greet": """
def greet(name):
    \"\"\"greet\"\"\"
    message = "Hello " + name
    print(message)
    return message
""",
"double_list": """
def double_list(lst):
    res = []
    for x in lst:
        res.append(x * 2)
    return res
"""
}
