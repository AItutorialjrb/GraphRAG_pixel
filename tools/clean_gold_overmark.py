import json, re

inp="data/eval/gold_review_v3_expanded_fast_corrected.json"
out="data/eval/gold_review_clean.json"

d=json.load(open(inp))

def norm(s):
    return re.sub(r"\s+", " ", str(s or "").lower())

for x in d:
    q = norm(x.get("query"))

    for c in x["candidates"]:
        t = norm(c.get("text"))

        # === HV-MAPS ===
        if "hv-maps" in q:
            c["gold"] = (
                "high-voltage monolithic active pixel sensors" in t
                or "high voltage monolithic active pixel sensors" in t
            )

        # === LGAD ===
        elif "lgad" in q:
            c["gold"] = (
                "low-gain avalanche detector" in t
                or "low gain avalanche detector" in t
            )

        # === ToT ===
        elif "time-over-threshold" in q or "tot" in q:
            c["gold"] = (
                ("time over threshold" in t or "tot" in t)
                and ("charge" in t or "amplitude" in t)
            )

        # === ToA ===
        elif "time-of-arrival" in q or "toa" in q:
            c["gold"] = (
                ("time of arrival" in t or "toa" in t)
                and ("time" in t or "timestamp" in t)
            )

        # === 其他问题：先全部关掉（后面人工补）
        else:
            c["gold"] = False

json.dump(d, open(out,"w"), indent=2, ensure_ascii=False)
print("wrote", out)
