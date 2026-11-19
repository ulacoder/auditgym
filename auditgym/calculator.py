"""Калькулятор: только арифметика над десятичными числами. Никаких имён, вызовов и атрибутов."""
from __future__ import annotations

import ast
import re
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

_ALLOWED = re.compile(r"[0-9.\s+\-*/%()]*")
MAX_LEN = 300
MAX_ABS = Decimal("1e30")


class CalcError(ValueError):
    pass


def evaluate(expression) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise CalcError("expression must be a non-empty string")
    if len(expression) > MAX_LEN:
        raise CalcError(f"expression longer than {MAX_LEN} chars")
    if not _ALLOWED.fullmatch(expression):
        raise CalcError("only digits, '.', '+', '-', '*', '/', '%', '(', ')' are allowed")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise CalcError("syntax error") from None

    def ev(node, depth=0):
        if depth > 60:
            raise CalcError("expression too deep")
        if isinstance(node, ast.Expression):
            return ev(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            seg = ast.get_source_segment(expression, node)
            return Decimal(seg.strip())
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand, depth + 1)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left, depth + 1), ev(node.right, depth + 1)
            op = node.op
            if isinstance(op, ast.Add):
                r = a + b
            elif isinstance(op, ast.Sub):
                r = a - b
            elif isinstance(op, ast.Mult):
                r = a * b
            elif isinstance(op, ast.Div):
                r = a / b
            elif isinstance(op, ast.FloorDiv):
                r = a // b
            elif isinstance(op, ast.Mod):
                r = a % b
            elif isinstance(op, ast.Pow):
                if b != b.to_integral_value() or abs(b) > 12:
                    raise CalcError("power exponent must be an integer with |n| <= 12")
                r = a ** int(b)
            else:
                raise CalcError("operator not allowed")
            if abs(r) > MAX_ABS:
                raise CalcError("result too large")
            return r
        raise CalcError("only arithmetic on decimal numbers is allowed")

    with localcontext() as ctx:
        ctx.prec = 34
        ctx.traps[DivisionByZero] = True
        ctx.traps[InvalidOperation] = True
        try:
            r = ev(tree)
        except (DivisionByZero, InvalidOperation, ZeroDivisionError):
            raise CalcError("division by zero or invalid operation") from None
        r = r.normalize()
        s = format(r, "f")
    return s
