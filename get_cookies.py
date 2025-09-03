# get_cookies.py
import time, os
import browser_cookie3

DOMAINS = ["youtube.com",".youtube.com","google.com",".google.com","accounts.google.com",".accounts.google.com"]

def dump(cj, out="yt_cookies.txt"):
    lines = ["# Netscape HTTP Cookie File"]
    now = int(time.time())
    for c in cj:
        if not any(d in c.domain for d in DOMAINS):
            continue
        domain = c.domain
        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.path
        secure = "TRUE" if c.secure else "FALSE"
        expires = int(getattr(c, "expires", 0) or now + 3600*24*30)
        lines.append("\t".join([domain, flag, path, secure, str(expires), c.name, c.value]))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")

try:
    cj = browser_cookie3.chrome()
    dump(cj)
except Exception as e:
    print("Export failed:", e)
