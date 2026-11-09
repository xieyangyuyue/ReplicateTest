import os, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAM  = ROOT / "CamVid"
RAW  = CAM / "701_StillsRaw_full"
LAB  = CAM / "LabeledApproved_full"
RAW.mkdir(exist_ok=True); LAB.mkdir(exist_ok=True)

# 1) class_dict.csv → label_colors.txt（格式：r g b ClassName）
with open(CAM / "class_dict.csv", newline="") as f:
    rows = list(csv.reader(f))
header = [h.strip().lower() for h in rows[0]]
def col(*names, default=None):
    for n in names:
        if n in header: return header.index(n)
    return default
ci = col("name", "class_name", "label", "class", default=0)
ri = col("r", "red", default=1); gi = col("g", "green", default=2); bi = col("b", "blue", default=3)

classes = []
with open(CAM / "label_colors.txt", "w") as f:
    for r in rows[1:]:
        if not r or not r[ci].strip(): continue
        name = r[ci].strip().replace(" ", "_")   # 名字不能带空格，否则解析会错
        f.write(f"{int(r[ri])} {int(r[gi])} {int(r[bi])} {name}\n")
        classes.append(name)
print(f"classes = {len(classes)}  -> {classes[:5]} ...")

# 2) 汇总三个 split，标签统一命名成 <stem>_L.png
pairs = []
for split in ("train", "val", "test"):
    img_dir, lab_dir = CAM / split, CAM / f"{split}_labels"
    if not img_dir.is_dir(): continue
    for p in sorted(img_dir.glob("*.png")):
        stem = p.stem
        cands = [lab_dir / f"{stem}_L.png", lab_dir / p.name, lab_dir / f"{stem}.png"]
        q = next((c for c in cands if c.is_file()), None)
        if q is None:
            print("!! 缺标签:", p); continue
        pairs.append((p, q, stem))
print("pairs =", len(pairs))

for p, q, stem in pairs:
    di, dl = RAW / f"{stem}.png", LAB / f"{stem}_L.png"
    if not di.exists(): di.symlink_to(p.resolve())
    if not dl.exists(): dl.symlink_to(q.resolve())

print("done. 现在检查：")
print(" ", len(list(RAW.glob('*.png'))), "张原图 /", len(list(LAB.glob('*_L.png'))), "张标签")
