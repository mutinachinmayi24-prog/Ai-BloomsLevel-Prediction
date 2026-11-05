"""Extract the textbook's real table of contents (the PDF bookmarks) into backend/data/syllabus.json.
Usage: python scripts/build_syllabus.py path/to/textbook.pdf
Structure: units (Parts + appendices) -> chapters -> sections, each with its PDF page range.
"""
import re, sys, json, pathlib
from pypdf import PdfReader

pdf = sys.argv[1]
out = pathlib.Path(__file__).resolve().parent.parent / "backend" / "data" / "syllabus.json"
r = PdfReader(pdf)


def flat(items, depth=0, acc=None):
    acc = [] if acc is None else acc
    for x in items:
        if isinstance(x, list):
            flat(x, depth + 1, acc)
        else:
            acc.append((depth, re.sub(r"[\x00-\x1f]", "", x.title).strip(), r.get_destination_page_number(x) + 1))     # 1-based PDF page
    return acc


ol = flat(r.outline)
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}
biblio = next(p for d, t, p in ol if d == 0 and t.startswith("Bibliography"))
units, unit, chap = [], None, None
for d, title, page in ol:
    m = re.match(r"^([IVX]+): (.+)$", title) if d == 0 else None
    a = re.match(r"^([AB]): (.+)$", title) if d == 0 else None
    if m and m.group(1) in ROMAN:
        unit = {"name": f"Part {m.group(1)}: {m.group(2)}", "chapters": []}
        units.append(unit)
        chap = None
    elif a:
        unit = {"name": f"Appendix {a.group(1)}: {a.group(2)}", "chapters": []}
        units.append(unit)
        chap = {"id": 28 if a.group(1) == "A" else 29, "number": a.group(1), "title": a.group(2), "start": page, "sections": []}   # an appendix is one 'chapter'
        unit["chapters"].append(chap)
    elif d == 0:
        unit = chap = None                                         # Cover, Contents, Bibliography, Index ...
    elif unit is not None and d == 1 and re.match(r"^\d+ ", title):
        n, t = title.split(" ", 1)
        chap = {"id": int(n), "number": n, "title": t.strip(), "start": page, "sections": []}
        unit["chapters"].append(chap)
    elif unit is not None and chap is not None and d >= 1 and re.match(r"^(\d+|[AB])\.\d+ ", title):
        n, t = title.split(" ", 1)
        if re.search(r"Summary|Bibliographical|Exercises", t):      # end-of-chapter study aids, not topics
            continue
        chap["sections"].append({"number": n, "title": t.strip(), "start": page})

# page ranges: each node ends where the next one starts
marks = sorted({c["start"] for u in units for c in u["chapters"]} | {s["start"] for u in units for c in u["chapters"] for s in c["sections"]} | {biblio})
nxt = lambda p: next((q for q in marks if q > p), biblio) - 1
for u in units:
    for c in u["chapters"]:
        c["end"] = nxt(c["start"]) if not c["sections"] else max(nxt(c["sections"][-1]["start"]), c["sections"][-1]["start"])
        for s in c["sections"]:
            s["end"] = max(nxt(s["start"]), s["start"])
first = min(c["start"] for u in units for c in u["chapters"])
data = {"units": units, "content_pages": [first, biblio - 1]}
out.write_text(json.dumps(data, indent=1))
nc = sum(len(u["chapters"]) for u in units); ns = sum(len(c["sections"]) for u in units for c in u["chapters"])
print(f"{len(units)} units, {nc} chapters, {ns} sections; content pages {first}-{biblio - 1}")
