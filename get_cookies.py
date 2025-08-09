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
    # 任选一个：
    # 1) 仅用 Firefox（常见地不需要管理员，只要完全退出 Firefox）
    # cj = browser_cookie3.firefox()      # ← 推荐先试这个
    # 2) 或仅用 Chrome 默认配置
    cj = browser_cookie3.chrome()
    # 3) 或指定 Chrome 某个 profile 的 cookie 文件（需要关闭浏览器）
    # cookie_file = os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\User Data\Profile 1\Network\Cookies')
    # cj = browser_cookie3.chrome(cookie_file=cookie_file)

    dump(cj)
except Exception as e:
    print("Export failed:", e)
