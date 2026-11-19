import sys, traceback
sys.path.insert(0, "/home/claude/auditgym")
from auditgym import calculator
cases = ["1e999", "0**-1", "0**0", "١٢", "1_000", "9"*300, "-"*298+"1", "("*100+"1"+")"*100,
         "(.1**12)**12", "((.1**12)**12)**12", "(((.1**12)**12)**12)**12", "((((.1**12)**12)**12)**12)**12",
         "(((((.1**12)**12)**12)**12)**12)**12", "+".join(["1"]*70), "1/3", "10**31", "007.5", "...", "1\n+2", "(1\n+2)",
         "1\x0c+2", "(1\r+2)", "2**0.5", "(-8)**(1/3)", "1e30", "1//0", "5%0", "10**30*10", "(10**12)**12",
         "9"*40+"//0.0000001", "1"+" "*298, "  ", "", None, 5, "1 2", "0x10", "1　+1", "1\xa0+1", "2**-12", ".5", "1.", "1..2",
         "9"*299+"*1", "-"+"9"*299, "%", "()", "(1)(2)", "1if 1else 2"]
for c in cases:
    try:
        r = calculator.evaluate(c)
        print(repr(c)[:40], "->", (r[:60] + ("...len=%d" % len(r) if len(r) > 60 else "")))
    except calculator.CalcError as e:
        print(repr(c)[:40], "CalcError", e)
    except BaseException as e:
        print(repr(c)[:40], "UNCAUGHT", type(e).__name__, str(e)[:80])
