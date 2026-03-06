#!/usr/bin/env python3
'''
for d in *_TEMP2; do prefix="${d%_TEMP2}"; infile="$d/$prefix.insertion.bed"; outfile="$d/$prefix.insertion.fam.bed"; [[ -f "$infile" ]] || { echo "Missing $infile, skipping"; continue; }; python add_transposon_family.py "$infile" family_list2.tsv > "$outfile"; done
'''

import sys
import re

def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def load_family_map(family_path: str) -> dict:
    """
    Parse family lines like either:
      >NC_057762.1:12967042-12972070#LTR/Copia/SIRE
      >NC_057762.1:12967042..12972070#LTR/Copia/SIRE

    Build mapping keyed ONLY by numeric coordinates:
      (start, end) -> family

    If the same (start,end) appears with multiple families, keep the LONGEST
    family string (most characters). If tied, keep the first seen.
    """
    fam = {}
    pat = re.compile(r'^>([^:]+):(\d+)(?:-|\.\.)(\d+)#(.+?)\s*$')

    with open(family_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = pat.match(line)
            if not m:
                continue

            start, end, family = m.group(2), m.group(3), m.group(4)
            key = (start, end)

            if key not in fam:
                fam[key] = family
            else:
                prev = fam[key]
                if family != prev:
                    # Keep the longest family string
                    if len(family) > len(prev):
                        fam[key] = family
                    # Warn if there is a discrepancy (and report which one we kept)
                    kept = fam[key]
                    print(
                        f"WARNING: coordinate {start}-{end} has multiple families: "
                        f"{prev} vs {family}. Keeping longest: {kept}.",
                        file=sys.stderr
                    )

    return fam

def annotate_hit(hit: str, fam_map: dict) -> str:
    """
    Hit examples:
      NC_057765.1_11619413-11623529_n=1:3790:3755:+
      NC_057765.1_11619413..11623529_n=1:3790:3755:+

    We match:
      _<start><sep><end>_
    where <sep> is '-' or '..'

    Output rules:
      1) Insert #<family> right after <end>
      2) Normalize any '..' separator to '-' in the output for consistency
    """
    m = re.search(r'_(\d+)(-|\.\.)(\d+)(?=_)', hit)
    if not m:
        return hit

    start, sep, end = m.group(1), m.group(2), m.group(3)
    family = fam_map.get((start, end))

    # Always normalize ".." -> "-" in the coordinate span we matched
    if sep == "..":
        hit = hit[:m.start(2)] + "-" + hit[m.end(2):]
        # refresh match on normalized string
        m2 = re.search(r'_(\d+)-(\d+)(?=_)', hit)
        if not m2:
            return hit
        start, end = m2.group(1), m2.group(2)
        if family is None:
            return hit
        end_pos = m2.end(2)
        return hit[:end_pos] + "#" + family + hit[end_pos:]

    # sep is already '-'
    if family is None:
        return hit

    end_pos = m.end(3)  # end coordinate end index in original hit
    return hit[:end_pos] + "#" + family + hit[end_pos:]

def main():
    if len(sys.argv) != 3:
        die("Usage: python add_transposon_family.py transposon_file.tsv family_file.list")

    transposon_path = sys.argv[1]
    family_path = sys.argv[2]

    fam_map = load_family_map(family_path)
    if not fam_map:
        print("WARNING: no family records loaded (check family file format).", file=sys.stderr)

    with open(transposon_path, "r", encoding="utf-8") as tfh:
        for line in tfh:
            if line.startswith("#") or not line.strip():
                sys.stdout.write(line)
                continue

            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                sys.stdout.write(line)
                continue

            # Column 4 (1-based) == index 3 (0-based)
            hits = cols[3].split(",")
            hits_mod = [annotate_hit(h, fam_map) for h in hits]
            cols[3] = ",".join(hits_mod)

            sys.stdout.write("\t".join(cols) + "\n")

if __name__ == "__main__":
    main()
