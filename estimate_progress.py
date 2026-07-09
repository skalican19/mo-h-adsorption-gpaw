import os, json, datetime as dt
from pathlib import Path

root = Path("data/outputs/gpaw_calculations")
res_json = Path("data/outputs/gpaw_h_adsorption_results.json")
res_csv  = Path("data/outputs/gpaw_h_adsorption_results.csv")

def nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0

# --- TOTAL ---
total = None
if os.environ.get("TOTAL"):
    total = int(os.environ["TOTAL"])
else:
    # try infer from results files, but don't trust if it's < dirs_seen
    if res_json.exists():
        try:
            data = json.loads(res_json.read_text())
            if isinstance(data, (dict, list)):
                total = len(data)
        except Exception:
            pass
    if total is None and res_csv.exists():
        try:
            n = sum(1 for _ in res_csv.open(errors="ignore")) - 1
            if n > 0:
                total = n
        except Exception:
            pass

# --- start time ---
start = None
if os.environ.get("START"):
    s = os.environ["START"].strip()
    try:
        start = dt.datetime.fromisoformat(s)
    except Exception:
        start = dt.datetime.strptime(s, "%Y-%m-%d %H:%M")
else:
    all_txt = list(root.rglob("*.txt")) if root.exists() else []
    if all_txt:
        start = dt.datetime.fromtimestamp(min(p.stat().st_mtime for p in all_txt))

now = dt.datetime.now()

dirs = [d for d in root.iterdir() if d.is_dir()] if root.exists() else []
done = []
inprog = []
all_txt = []

for d in dirs:
    clean = d / "clean_slab.txt"
    slab  = d / "slab_with_h.txt"
    h2    = d / "h2_molecule.txt"

    have = [nonempty(clean), nonempty(slab), nonempty(h2)]
    if any(have) and not all(have):
        inprog.append((sum(have), d))
    if all(have):
        t = max(x.stat().st_mtime for x in (clean, slab, h2))
        done.append((dt.datetime.fromtimestamp(t), d))
    for x in (clean, slab, h2):
        if nonempty(x):
            all_txt.append(x)

done.sort(key=lambda x: x[0])

# fix total if inferred is nonsense
dirs_seen = len(dirs)
if total is None or total < dirs_seen:
    total = dirs_seen

print("============================================================")
print("Progress from gpaw_calculations/")
print("============================================================")
print(f"Now:       {now}")
if start:
    print(f"Start:     {start}")
    print(f"Elapsed:   {now-start}")
else:
    print("Start:     (unknown)  -> set START='YYYY-MM-DD HH:MM' for ETA")

print(f"Dirs seen: {dirs_seen}")
print(f"Done:      {len(done)}/{total}")
print(f"In-progress dirs: {len(inprog)}")
print(f"Remaining: {total - len(done)}")

if all_txt:
    latest = max(all_txt, key=lambda p: p.stat().st_mtime)
    lt = dt.datetime.fromtimestamp(latest.stat().st_mtime)
    print(f"\nLatest file: {latest}")
    print(f"Updated at:  {lt}")
    print(f"Likely current case dir: {latest.parent}")

if inprog:
    print("\nIn-progress (missing some outputs):")
    for k, d in sorted(inprog, key=lambda x: (-x[0], x[1].name))[:20]:
        print(f"  {d.name}  ({k}/3 files present)")

if start and (total - len(done)) > 0 and len(done) > 0:
    elapsed_s = (now-start).total_seconds()
    rate_per_day = (len(done) / elapsed_s) * 86400
    rem = total - len(done)
    eta = now + dt.timedelta(days=(rem / rate_per_day)) if rate_per_day > 0 else None
    print(f"\nRate:      {rate_per_day:.2f} surfaces/day")
    if eta:
        print(f"ETA:       {eta}")

if done:
    print("\nLast completed:")
    for t, d in done[-5:]:
        print(f"  {t}  {d.name}")
