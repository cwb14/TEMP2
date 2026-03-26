#!/usr/bin/env python3
"""
https://www.nature.com/articles/s41467-025-64944-4
https://onlinelibrary.wiley.com/doi/epdf/10.1111/pbi.13926
https://academic.oup.com/plcell/article/36/4/840/7456361 # 441 TE presences;  5,306 to 6,528 polymorphic TEs.
https://cdn.elifesciences.org/articles/15716/elife-15716-v2.pdf # 2835 non-reference TE insertions with TSDs identified in total.
https://www.nature.com/articles/s41467-020-17874-2.pdf # 6906
https://link.springer.com/article/10.1186/s12864-017-4103-x # 274,408
https://academic.oup.com/nsr/article/11/6/nwae188/7687832?guestAccessKey=

TE Insertion Analysis – Publication Figure Generator
====================================================
Five-page multi-panel PDF from TEMP2 TE insertion BED files + master TSV.

  Page 1  TE composition & group comparisons      (panels A–D)
  Page 2  Regression analyses                      (panels A–C: Bio25, Bio30, CWD)
  Page 3  Glyphosate violin plots & BioClim corr  (2 rows)
  Page 4  Per-sample stacked bar by superfamily    (full width)
  Page 5  Per-sample stacked bar by LTR-RT family (full width)

Recommended command for marestail dataset:
    conda run -n bioinfo python te_figures.py \\
        --master MASTER_MARESTAIL_MERGED_less_China_Mar2026.tsv \\
        --te-pattern '{sample}_TEMP2/{sample}.insertion.fam.bed' \\
        --output te_figures.pdf \\
        --awk '$5 >= 0.1 && $8 >= 3 && $7 == "1p1"'

AWK filter rationale:
    $5 >= 0.1   : insertion frequency >= 10 %
    $8 >= 3     : >= 3 support reads
    $7 == "1p1" : split-read evidence from BOTH TE ends (most precise,
                  lowest false-positive rate; excludes discordant-pair-only calls)

General usage:
    python te_figures.py \\
        --master MASTER.tsv \\
        --te-pattern '{sample}_TEMP2/{sample}.insertion.fam.bed' \\
        --output te_figures.pdf \\
        [--awk '$5 >= 0.1 && $8 >= 3 && $7 == "1p1"'] \\
        [--fai ref.fa.fai] \\
        [--sample-col Sample]
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Publication rcParams ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           8.5,
    "axes.titlesize":      10,
    "axes.titleweight":    "bold",
    "axes.labelsize":      8.5,
    "xtick.labelsize":     7.5,
    "ytick.labelsize":     7.5,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.8,
    "axes.grid":           True,
    "grid.alpha":          0.22,
    "grid.linewidth":      0.5,
    "grid.color":          "#bbbbbb",
    "figure.dpi":          120,
    "savefig.dpi":         300,
    "legend.fontsize":     7,
    "legend.framealpha":   0.88,
    "legend.edgecolor":    "#cccccc",
    "legend.borderpad":    0.4,
    "patch.linewidth":     0.5,
})

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#BCBD22", "#17BECF", "#AEC6CF", "#FFB347",
]

# Per-category palettes — no shared colors between superfamily and family sets.
#
# FAM_PALETTE  (10): ColorBrewer Set1 — bold, highly saturated primaries.
# CLADE_PALETTE(24): ColorBrewer Paired(12) + Pastel2(8) + 4 extras —
#                    light/medium tones clearly distinct from Set1 saturation.

FAM_PALETTE = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#A65628",  # brown
    "#F781BF",  # pink
    "#FFFF33",  # yellow
    "#999999",  # grey
    "#1B9E77",  # teal-green (Dark2)
]

CLADE_PALETTE = [
    # Paired (12 light/dark pairs — muted, clearly unlike Set1)
    "#A6CEE3", "#1F78B4",   # light/dark sky-blue
    "#B2DF8A", "#33A02C",   # light/dark leaf-green
    "#FB9A99", "#E31A1C",   # light/dark rose-red  — distinct from Set1 red by tone
    "#FDBF6F", "#FF7F00",   # light/dark peach-orange  (different value from Set1)
    "#CAB2D6", "#6A3D9A",   # light/dark lavender
    "#FFFF99", "#B15928",   # light/dark straw + sienna
    # Pastel2 (8)
    "#B3E2CD", "#FDCDAC", "#CBD5E8", "#F4CAE4",
    "#E6F5C9", "#FFF2AE", "#F1E2CC", "#CCCCCC",
    # 4 additional muted tones to reach 24
    "#80CDC1", "#DFC27D", "#A6611A", "#762A83",
]

ANNOT = dict(
    fontsize=7.5, va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.35", fc="white",
              ec="#bbbbbb", alpha=0.93, lw=0.6),
)

BIO_KEY = {
    "Bio01": "Annual Mean Temperature",
    "Bio02": "Mean Diurnal Temp Range",
    "Bio03": "Isothermality",
    "Bio04": "Temperature Seasonality",
    "Bio05": "Max Temp, Warmest Month",
    "Bio06": "Min Temp, Coldest Month",
    "Bio07": "Temp Annual Range",
    "Bio08": "Mean Temp, Wettest Quarter",
    "Bio09": "Mean Temp, Driest Quarter",
    "Bio10": "Mean Temp, Warmest Quarter",
    "Bio11": "Mean Temp, Coldest Quarter",
    "Bio12": "Annual Precipitation (mm)",
    "Bio13": "Precip, Wettest Month",
    "Bio14": "Precip, Driest Month",
    "Bio15": "Precipitation Seasonality (CV)",
    "Bio16": "Precip, Wettest Quarter",
    "Bio17": "Precip, Driest Quarter",
    "Bio18": "Precip, Warmest Quarter",
    "Bio19": "Precip, Coldest Quarter",
    "Bio20": "Solar Radiation, annual (W m⁻²)",
    "Bio21": "Solar Radiation, max monthly",
    "Bio22": "Solar Radiation, min monthly",
    "Bio23": "Solar Radiation, seasonality",
    "Bio24": "Solar Radiation, wettest quarter",
    "Bio25": "Solar Radiation, driest quarter",
    "Bio26": "Solar Radiation, warmest quarter",
    "Bio27": "Solar Radiation, coldest quarter",
    "Bio28": "Vapour Pressure Deficit, seasonality",
    "Bio29": "VPD, wettest quarter (kPa)",
    "Bio30": "VPD, driest quarter (kPa)",
    "Bio31": "VPD, warmest quarter (kPa)",
    "Bio32": "VPD, coldest quarter (kPa)",
    "Bio33": "Mean wind speed (m s⁻¹)",
    "Bio34": "Max monthly wind speed",
    "Bio35": "Min monthly wind speed",
    "Bio36": "Actual Evapotranspiration, annual (mm)",
    "Bio37": "AET, max monthly",
    "Bio38": "AET, min monthly",
    "Bio39": "AET, seasonality",
    "Bio40": "AET, driest quarter",
}

VAR_DESC = {
    **{k: f"{k}: {v}" for k, v in BIO_KEY.items()},
    "CWD_All_Annual": (
        "CWD_All_Annual: Climatic Water Deficit – annual (mm)\n"
        "Potential − actual evapotranspiration.  Higher = drier / more drought stress."
    ),
}



# Fixed display order for Panel E (K2P age) — mirrors Panel C top-to-bottom.
# Families absent from the age file are silently skipped.
LTR_AGE_FAMILY_ORDER = [
    "SIRE", "Tekay", "unknown", "Ale", "Retand", "Athila",
    "Angela", "TAR", "Ikeros", "Ivana", "Bianca", "Reina", "CRM", "mixture",
]

GENE_CONTEXT_WINDOW_BP = 1000
#GENE_CONTEXT_WINDOW_BP = 2000
#GENE_CONTEXT_WINDOW_LABEL = "2kb"
GENE_CONTEXT_WINDOW_LABEL = "1kb"

CONTEXT_ORDER = [
    "Exonic",
    "Intronic",
    f"Upstream {GENE_CONTEXT_WINDOW_LABEL}",
    f"Downstream {GENE_CONTEXT_WINDOW_LABEL}",
    "Intergenic",
]

def get_te_cols(merged, col_prefix, exclude_mixture=True):
    cols = [c for c in merged.columns if c.startswith(col_prefix)]
    if exclude_mixture and col_prefix == "te_fam_":
        cols = [c for c in cols if c != "te_fam_mixture"]
    return cols

# ── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--master",     required=True)
    p.add_argument("--te-pattern", required=True)
    p.add_argument("--output",     default="te_figures.pdf")
    p.add_argument("--awk",        default=None, dest="awk_filter")
    p.add_argument("--sample-col", default="Sample")
    p.add_argument("--fai",        default=None,
                   help="Samtools FAI index (e.g. ref.fa.fai) for exact "
                        "chromosome lengths on the karyotype plot.")
    p.add_argument("--gff",        default=None,
                   help="GFF3 annotation file to check whether enriched "
                        "insertions fall inside a gene or its promoter (1 kb).")
    p.add_argument("--crm",        default=None,
                   help="CRM element file (one entry per line, format "
                        "'chr:start-end#LTR/Gypsy/CRM') used to overlay "
                        "centromere-proximal regions on the karyotype (page 5).")
    p.add_argument("--ltr-age",    default=None, dest="ltr_age",
                   help="LTR-RT age file (col1=locus#LTR/Superfamily/Family, "
                        "col11=K2P divergence between the two LTRs). Families "
                        "with >100 elements are plotted as stacked K2P density "
                        "histograms on the gene-context page (requires --gff).")
    return p.parse_args()


# ── CRM loader ──────────────────────────────────────────────────────────────

def parse_crm(path):
    """Parse a CRM element file into a dict {chrom: [(start, end), ...]}.

    Accepted line format (one per line):
        chr:start-end#anything      e.g. NC_057761.1:498719-504450#LTR/Gypsy/CRM
    Lines that don't match are silently skipped.
    """
    import re as _re
    pattern = _re.compile(r'^([^:]+):(\d+)-(\d+)')
    intervals = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                continue
            chrom = m.group(1)
            start, end = int(m.group(2)), int(m.group(3))
            intervals.setdefault(chrom, []).append((start, end))
    return intervals


# ── LTR-RT age loader ───────────────────────────────────────────────────────

def load_ltr_age(path):
    """Parse an LTR-RT age file.  Returns dict {family_name: [k2p_values]}.

    Expected format (whitespace-delimited, one element per line):
        col 1  : locus identifier, e.g. NC_057761.1:45750-55571#LTR/Copia/SIRE
        col 11 : K2P divergence between the element's two LTRs (float)
    The family name is the last '/'-separated token after '#' in col 1.
    Lines with < 11 columns or non-numeric col 11 are silently skipped.
    """
    family_k2p = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            col1 = parts[0]
            if '#' in col1:
                family_path = col1.split('#', 1)[1]   # e.g. "LTR/Copia/SIRE"
                family_name = family_path.split('/')[-1]
            else:
                family_name = col1
            try:
                k2p = float(parts[10])                 # 11th column (0-indexed: 10)
            except (ValueError, IndexError):
                continue
            family_k2p.setdefault(family_name, []).append(k2p)
    return family_k2p


# ── Data loading ────────────────────────────────────────────────────────────

def load_master(path, sample_col):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if str(df.columns[0]) in ("NA", "Unnamed: 0", "", "nan", "NaN"):
        df = df.drop(columns=[df.columns[0]])

    def _clean(v):
        try:
            f = float(v)
            if f == int(f):
                return str(int(f))
        except Exception:
            pass
        return str(v)

    df[sample_col] = df[sample_col].apply(_clean)
    return df.set_index(sample_col)


def load_te_bed(filepath, awk_filter=None):
    if not os.path.isfile(filepath):
        return None
    if awk_filter:
        cmd = f"awk 'BEGIN{{FS=OFS=\"\\t\"}} /^#/{{next}} {awk_filter}' '{filepath}'"
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=30)
            lines = r.stdout.strip().split("\n")
            if not lines or lines == [""]:
                return pd.DataFrame()
        except Exception:
            return None
    else:
        try:
            lines = [l.rstrip("\n") for l in open(filepath)
                     if not l.startswith("#")]
        except Exception:
            return None
    if not lines:
        return pd.DataFrame()

    rows = [l.split("\t") for l in lines]
    cols = ["Chr", "Start", "End", "TE_ID", "Frequency", "Strand",
            "Type", "SupportReads", "UnsupportReads",
            "5primeSR", "3primeSR", "TSD", "ConfSomatic", "Spl5", "Spl3"]
    mc = max(len(r) for r in rows)
    while len(cols) < mc:
        cols.append(f"x{len(cols)}")
    df = pd.DataFrame(rows, columns=cols[:mc])
    for c in ["Start", "End", "Frequency", "SupportReads"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_te_levels(te_id):
    """
    Parse TE_ID field.  Returns list of level tuples (order, superfamily, family) where:
      order       = parts[0]  (LTR, DNA, LINE …)                top-level order
      superfamily = parts[1]  (Copia, Gypsy, hAT, Helitron …)  superfamily level
      family      = parts[-1] (SIRE, TAR, Ale, Helitron, DTH …) family level
    """
    if not isinstance(te_id, str):
        return []
    results = []
    for part in te_id.split(","):
        m = re.search(r"#([^:_\s]+(?:/[^:_\s]+)*)", part)
        if m:
            lvls  = m.group(1).split("/")
            order = lvls[0]
            fam   = lvls[1] if len(lvls) >= 2 else lvls[0]
            clade = lvls[-1]   # most specific available level
            results.append((order, fam, clade))
    return results


def summarise_sample(sid, te_pattern, awk_filter):
    fp = te_pattern.replace("{sample}", str(sid))
    df = load_te_bed(fp, awk_filter)
    if df is None or df.empty:
        return None

    fam_counts   = Counter()
    clade_counts = Counter()

    if "TE_ID" in df.columns:
        for te_id in df["TE_ID"]:
            for order, fam, clade in parse_te_levels(te_id):
                fam_counts[fam] += 1
                if order == "LTR":          # clades only tracked for LTR-RTs
                    clade_counts[clade] += 1

    res = {
        "n_te_total":            len(df),
        "te_freq_mean":          df["Frequency"].mean() if "Frequency" in df.columns else np.nan,
        "te_support_reads_mean": df["SupportReads"].mean() if "SupportReads" in df.columns else np.nan,
    }
    for k, v in fam_counts.items():
        res[f"te_fam_{k}"] = v
    for k, v in clade_counts.items():
        res[f"te_clade_{k}"] = v
    return res


def load_gff_genes(path, upstream_bp=1000):
    """
    Parse a GFF3 file and return a DataFrame of gene features with
    precomputed promoter windows (strand-aware, upstream_bp upstream of TSS).

    Columns: chr, start, end, strand, gene_name, biotype, prom_start, prom_end
    """
    genes = []
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9 or parts[2] != "gene":
                    continue
                seqname, _, _, start, end, _, strand, _, attrs = parts
                start, end = int(start), int(end)

                attr = {}
                for seg in attrs.split(";"):
                    if "=" in seg:
                        k, v = seg.split("=", 1)
                        attr[k.strip()] = v.strip()

                gene_name = (attr.get("Name") or attr.get("gene")
                             or attr.get("ID", "unknown"))
                biotype   = attr.get("gene_biotype", "")

                if strand == "+":
                    prom_start, prom_end = max(0, start - upstream_bp), start
                elif strand == "-":
                    prom_start, prom_end = end, end + upstream_bp
                else:
                    prom_start = max(0, start - upstream_bp)
                    prom_end   = end + upstream_bp

                genes.append(dict(chr=seqname, start=start, end=end,
                                  strand=strand, gene_name=gene_name,
                                  biotype=biotype,
                                  prom_start=prom_start, prom_end=prom_end))
    except Exception as e:
        print(f"    WARNING: GFF parsing error: {e}")
        return pd.DataFrame()
    return pd.DataFrame(genes)


def load_gff_full(path):
    """Parse GFF3 for gene-disruption analysis.

    Returns
    -------
    genes_df : DataFrame [chr, start, end, strand, gene_id]  (1-based, inclusive)
    exon_intervals : dict {gene_id: np.ndarray shape (N,2)}
        Union of all exon intervals per gene, sorted and merged across transcripts.
    """
    gene_rows = []
    exon_raw  = {}   # {gene_id: [(start, end), ...]}

    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                chrom, _, feat, start, end, _, strand, _, attrs = parts
                try:
                    start, end = int(start), int(end)
                except ValueError:
                    continue

                attr = {}
                for seg in attrs.split(";"):
                    if "=" in seg:
                        k, v = seg.split("=", 1)
                        attr[k.strip()] = v.strip()

                if feat == "gene":
                    # Canonical gene_id = Name= (e.g. LOC122597209)
                    gene_id = (attr.get("Name") or
                               attr.get("ID", "").replace("gene-", ""))
                    if gene_id:
                        gene_rows.append(dict(chr=chrom, start=start, end=end,
                                              strand=strand, gene_id=gene_id))

                elif feat == "exon":
                    # Exon rows carry gene= attribute directly
                    gene_id = attr.get("gene", "")
                    if gene_id:
                        exon_raw.setdefault(gene_id, []).append((start, end))
    except Exception as e:
        print(f"    WARNING: GFF parsing error in load_gff_full: {e}")
        return pd.DataFrame(), {}

    genes_df = pd.DataFrame(gene_rows) if gene_rows else pd.DataFrame(
        columns=["chr", "start", "end", "strand", "gene_id"])

    # Merge overlapping exon intervals per gene
    exon_intervals = {}
    for gene_id, ivs in exon_raw.items():
        arr = sorted(ivs, key=lambda x: x[0])
        merged = [list(arr[0])]
        for s, e in arr[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        exon_intervals[gene_id] = np.array(merged, dtype=np.int64)

    return genes_df, exon_intervals


def compute_context_lengths(genes_df, exon_intervals, window=GENE_CONTEXT_WINDOW_BP,
                            fai_lengths=None):
    """Return total base-pairs (in kb) for each gene context across all genes.

    Used to normalise raw insertion counts to density (insertions per kb) so
    that contexts of different lengths are directly comparable.

    Keys match CONTEXT_ORDER: Exonic, Intronic, Upstream …, Downstream …, Intergenic.

    Intergenic = everything farther than `window` bp from any gene (strand-agnostic).
    Computed as: total_genome_bp minus the merged union of [start-window, end+window]
    across all genes (per chromosome, capped at chromosome boundaries).  This correctly
    handles overlapping genes and windows without double-counting.
    Requires fai_lengths dict {chr: length}; falls back to 1 kb if unavailable.
    """
    total_exon_bp   = 0
    total_intron_bp = 0
    for row in genes_df.itertuples(index=False):
        gene_bp = int(row.end) - int(row.start) + 1
        exons   = exon_intervals.get(row.gene_id)
        if exons is not None and len(exons):
            exon_bp = int(np.sum(exons[:, 1] - exons[:, 0] + 1))
        else:
            exon_bp = 0
        total_exon_bp   += exon_bp
        total_intron_bp += max(0, gene_bp - exon_bp)

    # Upstream / downstream window lengths: use merged intervals per chromosome so
    # overlapping windows between adjacent genes are not double-counted.
    # Strandedness is irrelevant here — the physical proximal zone around each gene
    # is [start - window, end + window] regardless of strand.
    total_proximal_bp = 0
    for chrom, grp in genes_df.groupby("chr"):
        chr_len = fai_lengths.get(chrom, None) if fai_lengths else None
        ivs = sorted(
            (max(0, int(r.start) - window),
             (min(int(r.end) + window, chr_len - 1) if chr_len else int(r.end) + window))
            for r in grp.itertuples(index=False)
        )
        # Merge overlapping intervals
        merged = []
        for s, e in ivs:
            if merged and s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        total_proximal_bp += sum(e - s + 1 for s, e in merged)

    # Upstream and downstream windows are each half of the proximal zone around gene
    # ends; report them separately as n_genes * window (un-merged) for panel labels,
    # but use the merged proximal total for intergenic.
    n_genes       = len(genes_df)
    upstream_bp   = n_genes * window
    downstream_bp = n_genes * window

    if fai_lengths:
        total_genome_bp = sum(fai_lengths.values())
        intergenic_bp   = max(total_genome_bp - total_proximal_bp, 1)
    else:
        intergenic_bp   = 1

    return {
        "Exonic":                                  max(total_exon_bp,   1) / 1000,
        "Intronic":                                max(total_intron_bp, 1) / 1000,
        f"Upstream {GENE_CONTEXT_WINDOW_LABEL}":   upstream_bp            / 1000,
        f"Downstream {GENE_CONTEXT_WINDOW_LABEL}": downstream_bp          / 1000,
        "Intergenic":                              intergenic_bp          / 1000,
    }


def compute_gene_disruptions(pos_df, genes_df, exon_intervals):
    """For each sample, count unique genes disrupted exonically vs intronically.

    Returns DataFrame [sample_id, n_exonic, n_intronic].
    """
    if genes_df.empty or pos_df.empty:
        return pd.DataFrame(columns=["sample_id", "n_exonic", "n_intronic"])

    # Build per-chromosome sorted gene arrays
    chr_gene = {}
    for chrom, grp in genes_df.groupby("chr"):
        grp_s = grp.sort_values("start")
        chr_gene[chrom] = {
            "starts":   grp_s["start"].values.astype(np.int64),
            "ends":     grp_s["end"].values.astype(np.int64),
            "gene_ids": grp_s["gene_id"].values,
        }

    results = []
    for sid, sdf in pos_df.groupby("sample_id"):
        n_exonic = 0
        n_intronic = 0
        for chrom, cdf in sdf.groupby("Chr"):
            if chrom not in chr_gene:
                continue
            cg = chr_gene[chrom]
            starts   = cg["starts"]
            ends     = cg["ends"]
            gene_ids = cg["gene_ids"]
            te_pos   = cdf["Start"].values.astype(np.int64)

            # Vectorised overlap: te_pos[i] overlaps gene[j] iff
            # starts[j] <= te_pos[i] <= ends[j]
            # shape: (n_te, n_genes)
            overlap = (
                (starts[np.newaxis, :] <= te_pos[:, np.newaxis]) &
                (te_pos[:, np.newaxis] <= ends[np.newaxis, :])
            )
            hit_gene_idx = np.where(overlap.any(axis=0))[0]

            for g_idx in hit_gene_idx:
                gene_id  = gene_ids[g_idx]
                gs, ge   = int(starts[g_idx]), int(ends[g_idx])
                # TE positions that hit this gene
                te_hitting = te_pos[overlap[:, g_idx]]
                exons = exon_intervals.get(gene_id)
                if exons is None:
                    n_intronic += 1
                    continue
                is_exonic = False
                for tp in te_hitting:
                    j = np.searchsorted(exons[:, 0], tp, side="right") - 1
                    if j >= 0 and exons[j, 1] >= tp:
                        is_exonic = True
                        break
                if is_exonic:
                    n_exonic += 1
                else:
                    n_intronic += 1

        results.append({"sample_id": sid,
                        "n_exonic": n_exonic,
                        "n_intronic": n_intronic})

    return pd.DataFrame(results)


def compute_metagene_profile(pos_df, genes_df, window=2000, bin_size=50,
                             max_genes=5000, seed=42):
    """Compute TE insertion frequency per bin relative to gene boundaries.

    Strand-aware: upstream = 5' of TSS, downstream = 3' of TES.
      + strand: TSS = gene_start, TES = gene_end
                upstream window [start-window, start-1], rel = pos - start (neg)
                downstream window [end+1, end+window],   rel = pos - end   (pos)
      − strand: TSS = gene_end,   TES = gene_start
                upstream window [end+1, end+window],     rel = end - pos   (neg)
                downstream window [start-window, start-1], rel = start-pos (pos)

    Frequency = (genes with ≥1 TE in that bin) / total_genes.
    Returns DataFrame [sample_id, bin_center, freq].
    """
    if genes_df.empty or pos_df.empty:
        return pd.DataFrame(columns=["sample_id", "bin_center", "freq"])

    # Subsample genes for performance
    rng_sub = np.random.default_rng(seed)
    if len(genes_df) > max_genes:
        idx = rng_sub.choice(len(genes_df), max_genes, replace=False)
        genes_sub = genes_df.iloc[idx].reset_index(drop=True)
    else:
        genes_sub = genes_df.reset_index(drop=True)
    n_genes = len(genes_sub)

    n_bins = (2 * window) // bin_size
    bin_centers = np.arange(-window + bin_size // 2, window, bin_size)

    # Pre-build per-chromosome, per-sample sorted TE position arrays
    sample_chr_pos = {}
    for sid, sdf in pos_df.groupby("sample_id"):
        sample_chr_pos[sid] = {}
        for chrom, cdf in sdf.groupby("Chr"):
            sample_chr_pos[sid][chrom] = np.sort(
                cdf["Start"].values.astype(np.int64))

    # Pre-group genes by chromosome for fast iteration
    genes_by_chr = {c: g for c, g in genes_sub.groupby("chr")}

    results = []
    for sid, chr_pos in sample_chr_pos.items():
        bin_hits = np.zeros(n_bins, dtype=np.int64)

        for chrom, gene_grp in genes_by_chr.items():
            pos_arr = chr_pos.get(chrom)
            if pos_arr is None or len(pos_arr) == 0:
                continue

            for gene in gene_grp.itertuples(index=False):
                gs     = int(gene.start)
                ge     = int(gene.end)
                strand = getattr(gene, "strand", "+")

                if strand == "-":
                    # TSS at ge, TES at gs
                    # Upstream:   pos in [ge+1, ge+window], rel = ge - pos (neg)
                    lo = int(np.searchsorted(pos_arr, ge + 1,          side="left"))
                    hi = int(np.searchsorted(pos_arr, ge + window + 1, side="left"))
                    if lo < hi:
                        rel  = ge - pos_arr[lo:hi]
                        bidx = ((rel + window) // bin_size).astype(int)
                        bidx = bidx[(bidx >= 0) & (bidx < n_bins)]
                        if bidx.size:
                            bin_hits[np.unique(bidx)] += 1

                    # Downstream: pos in [gs-window, gs-1], rel = gs - pos (pos)
                    lo = int(np.searchsorted(pos_arr, gs - window, side="left"))
                    hi = int(np.searchsorted(pos_arr, gs,          side="left"))
                    if lo < hi:
                        rel  = gs - pos_arr[lo:hi]
                        bidx = ((rel + window) // bin_size).astype(int)
                        bidx = bidx[(bidx >= 0) & (bidx < n_bins)]
                        if bidx.size:
                            bin_hits[np.unique(bidx)] += 1
                else:
                    # + strand (or unknown): TSS at gs, TES at ge
                    # Upstream:   pos in [gs-window, gs-1], rel = pos - gs (neg)
                    lo = int(np.searchsorted(pos_arr, gs - window, side="left"))
                    hi = int(np.searchsorted(pos_arr, gs,          side="left"))
                    if lo < hi:
                        rel  = pos_arr[lo:hi] - gs
                        bidx = ((rel + window) // bin_size).astype(int)
                        bidx = bidx[(bidx >= 0) & (bidx < n_bins)]
                        if bidx.size:
                            bin_hits[np.unique(bidx)] += 1

                    # Downstream: pos in [ge+1, ge+window], rel = pos - ge (pos)
                    lo = int(np.searchsorted(pos_arr, ge + 1,          side="left"))
                    hi = int(np.searchsorted(pos_arr, ge + window + 1, side="left"))
                    if lo < hi:
                        rel  = pos_arr[lo:hi] - ge
                        bidx = ((rel + window) // bin_size).astype(int)
                        bidx = bidx[(bidx >= 0) & (bidx < n_bins)]
                        if bidx.size:
                            bin_hits[np.unique(bidx)] += 1

        freq = bin_hits / max(n_genes, 1)
        for bc, f in zip(bin_centers, freq):
            results.append({"sample_id": sid, "bin_center": int(bc), "freq": float(f)})

    return pd.DataFrame(results)


def compute_context_te_composition(pos_df, genes_df, exon_intervals,
                                   window=GENE_CONTEXT_WINDOW_BP):
    """Classify each TE insertion into gene context and tabulate by superfamily/family.

    Priority: Exonic > Intronic > Upstream 2 kb > Downstream 2 kb.
    Insertions that fall in none of these windows are discarded (intergenic).

    Returns (fam_df, clade_df) — count DataFrames with
      rows = contexts, columns = superfamily or family names.
    """
    CONTEXTS = CONTEXT_ORDER
    fam_counts   = {ctx: Counter() for ctx in CONTEXTS}
    clade_counts = {ctx: Counter() for ctx in CONTEXTS}
    # Per-sample tracking for violin plots
    sample_fam_counts   = {}  # (sample_id, context, family) -> count
    sample_clade_counts = {}  # (sample_id, context, clade)  -> count

    if pos_df.empty or genes_df.empty:
        empty_long = pd.DataFrame(columns=["sample_id", "context", "category", "count"])
        return (pd.DataFrame(0, index=CONTEXTS, columns=["Unknown"]),
                pd.DataFrame(0, index=CONTEXTS, columns=["Unknown"]),
                empty_long, empty_long)

    has_te_id  = "TE_ID" in pos_df.columns
    has_sample = "sample_id" in pos_df.columns

    # Build per-chromosome sorted gene arrays
    chr_gene = {}
    for chrom, grp in genes_df.groupby("chr"):
        grp_s = grp.sort_values("start")
        chr_gene[chrom] = {
            "starts":   grp_s["start"].values.astype(np.int64),
            "ends":     grp_s["end"].values.astype(np.int64),
            "gene_ids": grp_s["gene_id"].values,
            "strands":  (grp_s["strand"].values
                         if "strand" in grp_s.columns
                         else np.full(len(grp_s), "+")),
        }

    CHUNK = 1000  # TEs per vectorised chunk
    for chrom, cdf in pos_df.groupby("Chr"):
        if chrom not in chr_gene:
            continue
        cg       = chr_gene[chrom]
        starts   = cg["starts"]
        ends     = cg["ends"]
        gene_ids = cg["gene_ids"]
        strands  = cg["strands"]

        te_pos = cdf["Start"].values.astype(np.int64)
        te_ids = (cdf["TE_ID"].values if has_te_id
                  else np.full(len(cdf), "Unknown"))
        te_sids = (cdf["sample_id"].values if has_sample
                   else np.full(len(cdf), "Unknown"))
        n_te   = len(te_pos)

        # Strand-split arrays for upstream/downstream vectorisation
        plus_m  = strands == "+"
        minus_m = ~plus_m
        p_starts = starts[plus_m];  p_ends = ends[plus_m]
        m_starts = starts[minus_m]; m_ends = ends[minus_m]

        context_arr = np.full(n_te, -1, dtype=np.int8)

        for c0 in range(0, n_te, CHUNK):
            c1     = min(c0 + CHUNK, n_te)
            tp_arr = te_pos[c0:c1]

            # Body overlap: (chunk, n_genes)
            overlap   = ((starts[np.newaxis, :] <= tp_arr[:, np.newaxis]) &
                         (tp_arr[:, np.newaxis] <= ends[np.newaxis, :]))
            body_mask = overlap.any(axis=1)

            for i in range(len(tp_arr)):
                tp = int(tp_arr[i])
                if body_mask[i]:
                    # Gene body — classify as exonic or intronic
                    is_exonic = False
                    for g_idx in np.where(overlap[i])[0]:
                        exons = exon_intervals.get(gene_ids[g_idx])
                        if exons is not None:
                            j = int(np.searchsorted(exons[:, 0], tp, side="right")) - 1
                            if j >= 0 and exons[j, 1] >= tp:
                                is_exonic = True
                                break
                    context_arr[c0 + i] = 0 if is_exonic else 1
                else:
                    # Check upstream (priority) then downstream
                    up, dn = False, False
                    if p_starts.size:
                        up = up or bool(np.any((p_starts - window <= tp) & (tp < p_starts)))
                        dn = dn or bool(np.any((p_ends < tp) & (tp <= p_ends + window)))
                    if m_ends.size:
                        up = up or bool(np.any((m_ends < tp) & (tp <= m_ends + window)))
                        dn = dn or bool(np.any((m_starts - window <= tp) & (tp < m_starts)))
                    if up:
                        context_arr[c0 + i] = 2
                    elif dn:
                        context_arr[c0 + i] = 3

        # Unclassified TEs (not in any genic/proximal window) → Intergenic
        context_arr[context_arr == -1] = 4

        # Accumulate superfamily/family counts per context (aggregate + per-sample)
        for i in range(n_te):
            ctx_idx = int(context_arr[i])
            ctx_name  = CONTEXTS[ctx_idx]
            te_id_str = str(te_ids[i])
            sid       = str(te_sids[i])
            for order, fam, clade in parse_te_levels(te_id_str):
                fam_counts[ctx_name][fam] += 1
                key_f = (sid, ctx_name, fam)
                sample_fam_counts[key_f] = sample_fam_counts.get(key_f, 0) + 1
                if order == "LTR":
                    clade_counts[ctx_name][clade] += 1
                    key_c = (sid, ctx_name, clade)
                    sample_clade_counts[key_c] = sample_clade_counts.get(key_c, 0) + 1

    # Build output DataFrames (rows=contexts, cols=superfamily/family)
    all_fams   = sorted({f for c in fam_counts.values()   for f in c}) or ["Unknown"]
    all_clades = sorted({c for cc in clade_counts.values() for c in cc}) or ["Unknown"]

    fam_df = pd.DataFrame(
        [[fam_counts[ctx].get(f, 0) for f in all_fams] for ctx in CONTEXTS],
        index=CONTEXTS, columns=all_fams,
    )
    clade_df = pd.DataFrame(
        [[clade_counts[ctx].get(c, 0) for c in all_clades] for ctx in CONTEXTS],
        index=CONTEXTS, columns=all_clades,
    )

    # Build per-sample long-form DataFrames
    fam_long = pd.DataFrame(
        [{"sample_id": k[0], "context": k[1], "category": k[2], "count": v}
         for k, v in sample_fam_counts.items()]
    ) if sample_fam_counts else pd.DataFrame(
        columns=["sample_id", "context", "category", "count"])
    clade_long = pd.DataFrame(
        [{"sample_id": k[0], "context": k[1], "category": k[2], "count": v}
         for k, v in sample_clade_counts.items()]
    ) if sample_clade_counts else pd.DataFrame(
        columns=["sample_id", "context", "category", "count"])

    return fam_df, clade_df, fam_long, clade_long


def describe_cluster_tes(chrom, pos, pos_df, window=50):
    """
    Find all BED rows within ±window bp of pos on chrom and summarise
    the TE identities (order/superfamily/family) present in the cluster.
    Returns a list of strings, one per unique TE type, sorted by count desc.
    """
    if pos_df.empty or "TE_ID" not in pos_df.columns:
        return []
    mask = (
        (pos_df["Chr"] == chrom)
        & (pos_df["Start"] >= pos - window)
        & (pos_df["End"]   <= pos + window)
    )
    sub = pos_df.loc[mask, "TE_ID"].dropna()
    if sub.empty:
        return []

    counts = Counter()
    for te_id in sub:
        for part in str(te_id).split(","):
            m = re.search(r"#([^:_\s]+)", part)
            if m:
                counts[m.group(1)] += 1   # e.g. "LTR/Copia/SIRE"

    return [f"{taxonomy}  (n={cnt})"
            for taxonomy, cnt in counts.most_common()]


def annotate_insertion(chrom, pos, genes_df, upstream_bp=1000):
    """
    Return list of dicts describing how pos overlaps genes on chrom.
    Each dict has keys: gene_name, biotype, overlap, strand, dist_bp,
    gene_start, gene_end.
      overlap = 'INTERNAL' | 'PROMOTER'
      dist_bp = bp from TSS (PROMOTER only)
    """
    if genes_df.empty:
        return []
    chr_g = genes_df[genes_df["chr"] == chrom]
    hits  = []
    for _, g in chr_g.iterrows():
        if g["start"] <= pos <= g["end"]:
            hits.append(dict(gene_name=g["gene_name"], biotype=g["biotype"],
                             overlap="INTERNAL", strand=g["strand"],
                             dist_bp=0,
                             gene_start=g["start"], gene_end=g["end"]))
        elif g["prom_start"] <= pos <= g["prom_end"]:
            dist = (g["start"] - pos if g["strand"] == "+"
                    else pos - g["end"])
            hits.append(dict(gene_name=g["gene_name"], biotype=g["biotype"],
                             overlap="PROMOTER", strand=g["strand"],
                             dist_bp=abs(dist),
                             gene_start=g["start"], gene_end=g["end"]))
    return hits


def load_all_bed_positions(merged, te_pattern, awk_filter):
    """
    Re-read BED files for all samples and return a DataFrame of raw positions:
      Chr, Start, End, sample_id
    Used for the karyotype enrichment analysis.
    """
    records = []
    for sid in merged.index:
        fp  = te_pattern.replace("{sample}", str(sid))
        df  = load_te_bed(fp, awk_filter)
        if df is None or df.empty:
            continue
        if not {"Chr", "Start", "End"}.issubset(df.columns):
            continue
        keep_cols = ["Chr", "Start", "End"]
        if "TE_ID" in df.columns:
            keep_cols.append("TE_ID")
        sub = df[keep_cols].copy()
        sub["Start"] = pd.to_numeric(sub["Start"], errors="coerce")
        sub["End"]   = pd.to_numeric(sub["End"],   errors="coerce")
        sub = sub.dropna(subset=["Start", "End"])
        sub["sample_id"] = sid
        records.append(sub)
    if not records:
        return pd.DataFrame(columns=["Chr", "Start", "End", "sample_id"])
    return pd.concat(records, ignore_index=True)


# ── Statistical helpers ─────────────────────────────────────────────────────

def drop_outliers_iqr(x, y, k=3.0):
    """Return (xc, yc, n_excluded) with IQR outliers on y removed."""
    mask = x.notna() & y.notna()
    xp, yp = x[mask], y[mask]
    q1, q3 = yp.quantile(0.25), yp.quantile(0.75)
    iqr = q3 - q1
    keep = ((yp >= q1 - k * iqr) & (yp <= q3 + k * iqr)
            if iqr > 0 else pd.Series(True, index=yp.index))
    n_excl = int((~keep).sum())
    return xp[keep], yp[keep], n_excl


def safe_corr(x, y, method="spearman"):
    mask = x.notna() & y.notna()
    xc, yc = x[mask], y[mask]
    if len(xc) < 5 or xc.std() == 0 or yc.std() == 0:
        return np.nan, np.nan
    return (stats.spearmanr(xc, yc) if method == "spearman"
            else stats.pearsonr(xc, yc))


def safe_kruskal(x, groups, min_n=3):
    vecs = [x[groups == g].dropna().values
            for g in groups.dropna().unique()
            if len(x[groups == g].dropna()) >= min_n]
    if len(vecs) < 2:
        return np.nan, np.nan
    try:
        return stats.kruskal(*vecs)
    except Exception:
        return np.nan, np.nan


def _chrom_sort_key(name):
    """Sort chromosome names (e.g. NC_057761.1) by embedded integer."""
    m = re.search(r'(\d+)', str(name))
    return int(m.group(1)) if m else 0


def find_enriched_clusters(pos_df, gly_map, window=5, min_samples=2, alpha=0.05):
    """
    Cluster TE insertions across samples within ±window bp.
    For each cluster test R vs S enrichment with Fisher exact + Bonferroni correction.

    Parameters
    ----------
    pos_df   : DataFrame with columns Chr, Start, End, sample_id
    gly_map  : dict {sample_id: True (Resistant) / False (Susceptible)}
    window   : bp to expand each insertion on each side before merging
    min_samples : minimum distinct samples in a cluster to test

    Returns
    -------
    DataFrame: Chr, pos, r_count, s_count, total, log2fc, pval, padj, enrichment
    """
    if pos_df.empty:
        return pd.DataFrame()

    n_R = sum(1 for v in gly_map.values() if v is True)
    n_S = sum(1 for v in gly_map.values() if v is False)
    if n_R == 0 or n_S == 0:
        return pd.DataFrame()

    results = []

    for chrom, grp in pos_df.groupby("Chr"):
        rows = sorted(
            zip(grp["Start"].astype(int), grp["End"].astype(int), grp["sample_id"]),
            key=lambda x: x[0],
        )
        if not rows:
            continue

        # Sweep-line merge: expand each interval by ±window then merge overlaps
        cs, ce = rows[0][0] - window, rows[0][1] + window
        csids  = {rows[0][2]}

        def _flush(cs, ce, csids):
            if len(csids) < min_samples:
                return
            known = {s for s in csids if gly_map.get(s) is not None}
            r_with    = sum(1 for s in known if gly_map[s] is True)
            s_with    = sum(1 for s in known if gly_map[s] is False)
            r_without = n_R - r_with
            s_without = n_S - s_with
            try:
                # two-sided test; direction assigned post-hoc via log2fc sign
                _, pval = stats.fisher_exact(
                    [[r_with, r_without], [s_with, s_without]],
                    alternative="two-sided",
                )
            except Exception:
                pval = 1.0
            eps    = 0.5 / max(n_R, n_S)      # pseudocount avoids log(0)
            log2fc = np.log2((r_with / n_R + eps) / (s_with / n_S + eps))
            results.append({
                "Chr":     chrom,
                "pos":     (cs + ce) / 2,      # midpoint of merged window
                "r_count": r_with,
                "s_count": s_with,
                "total":   len(csids),
                "log2fc":  log2fc,
                "pval":    pval,
            })

        for start, end, sid in rows[1:]:
            es = start - window
            ee = end   + window
            if es <= ce:          # overlapping — extend cluster
                ce = max(ce, ee)
                csids.add(sid)
            else:                 # gap — flush and start new cluster
                _flush(cs, ce, csids)
                cs, ce, csids = es, ee, {sid}
        _flush(cs, ce, csids)

    if not results:
        return pd.DataFrame()

    cdf = pd.DataFrame(results)

    # Benjamini-Hochberg FDR (step-down, monotonicity-enforced)
    n = len(cdf)
    sorted_idx = cdf["pval"].argsort().values          # ascending rank order
    ranks = np.empty(n, dtype=float)
    ranks[sorted_idx] = np.arange(1, n + 1)
    raw_q = (cdf["pval"] * n / ranks).clip(upper=1.0)
    # step-down: scan from largest p to smallest, take running minimum
    cdf["padj"] = (raw_q.iloc[sorted_idx[::-1]]
                   .cummin()
                   .reindex(cdf.index))

    # Assign direction post-hoc from log2fc sign
    cdf["enrichment"] = "shared"
    cdf.loc[(cdf["padj"] < alpha) & (cdf["log2fc"] > 0), "enrichment"] = "R"
    cdf.loc[(cdf["padj"] < alpha) & (cdf["log2fc"] < 0), "enrichment"] = "S"
    return cdf


def best_annot_corner(xc, yc):
    """Return (xy, va) placing annotation in the quadrant with fewest points."""
    xm, ym = np.median(xc), np.median(yc)
    counts = [
        ((xc < xm) & (yc > ym)).sum(),   # top-left
        ((xc > xm) & (yc > ym)).sum(),   # top-right
        ((xc < xm) & (yc < ym)).sum(),   # bottom-left
        ((xc > xm) & (yc < ym)).sum(),   # bottom-right
    ]
    corners = [
        ((0.03, 0.97), "top"),
        ((0.60, 0.97), "top"),
        ((0.03, 0.05), "bottom"),
        ((0.60, 0.05), "bottom"),
    ]
    return corners[int(np.argmin(counts))]


# ── Panel drawing functions ─────────────────────────────────────────────────

def panel_bar(ax, merged, col_prefix, title, panel_letter, top_n=12):
    """Generic horizontal bar: mean TE insertions for te_fam_* or te_clade_*.
    Pass top_n=None to show all categories."""
#    cols = [c for c in merged.columns if c.startswith(col_prefix)
#            and c != col_prefix + "mixture"]
    cols = get_te_cols(merged, col_prefix)
    y_label = "LTR-RT Family" if "clade" in col_prefix else "TE Superfamily"
    if not cols:
        ax.text(0.5, 0.5, f"No {col_prefix} data", ha="center",
                transform=ax.transAxes)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
        return

    all_means = merged[cols].mean().sort_values(ascending=False)
    means = all_means if top_n is None else all_means.head(top_n)
    sems  = (merged[means.index].std() /
             np.sqrt(merged[means.index].notna().sum())).values
    labels = [c.replace(col_prefix, "") for c in means.index]
    pal    = CLADE_PALETTE if "clade" in col_prefix else FAM_PALETTE
    colors = [pal[i % len(pal)] for i in range(len(labels))]

    ax.barh(labels[::-1], means.values[::-1], xerr=sems[::-1],
            color=colors[::-1], edgecolor="white", height=0.65,
            error_kw=dict(elinewidth=0.7, capsize=2, ecolor="#555"))
    ax.set_xlabel("Mean Insertions per Sample")
    ax.set_ylabel(y_label)
    ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right",
            clip_on=False)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0)


def panel_boxplot(ax, merged, cat_col, title, panel_letter, label_map=None,
                  ylabel=None):
    """Horizontal sorted boxplot + jitter."""
    if cat_col not in merged.columns:
        ax.text(0.5, 0.5, f"'{cat_col}' not found", ha="center",
                transform=ax.transAxes)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
        return

    sub = merged[[cat_col, "n_te_total"]].copy()
    sub[cat_col] = sub[cat_col].astype(str)
    if label_map:
        sub[cat_col] = sub[cat_col].map(
            {str(k): v for k, v in label_map.items()})
    # Drop rows where category is NaN or literal "nan" string
    sub = sub[sub[cat_col].notna() & (sub[cat_col].str.lower() != "nan")]
    sub = sub.dropna()
    if sub.empty:
        ax.text(0.5, 0.5, "No data after dropna", ha="center",
                transform=ax.transAxes)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
        return

    medians = sub.groupby(cat_col)["n_te_total"].median().sort_values(ascending=True)
    order   = medians.index.tolist()
    if len(order) < 2:
        ax.text(0.5, 0.5, "Need ≥ 2 groups", ha="center",
                transform=ax.transAxes)
        return

    data = [sub.loc[sub[cat_col] == g, "n_te_total"].values for g in order]
    bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(linewidth=0.7),
                    whiskerprops=dict(linewidth=0.7),
                    capprops=dict(linewidth=0.7))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.72)

    rng = np.random.default_rng(42)
    for i, d in enumerate(data):
        jitter = rng.normal(0, 0.08, len(d))
        ax.scatter(d, np.full(len(d), i + 1) + jitter,
                   alpha=0.32, s=7, color="black", zorder=4, linewidths=0)

    ax.set_yticks(range(1, len(order) + 1))
    fs = max(5, min(8, 120 // max(len(order), 1)))
    ax.set_yticklabels(order, fontsize=fs)
    ax.set_xlabel("Total TE Insertions")
    if ylabel:
        ax.set_ylabel(ylabel)
    H, p = safe_kruskal(sub["n_te_total"], sub[cat_col])
    if not np.isnan(p):
        ax.text(0.97, 0.03, f"Kruskal–Wallis\np = {p:.2e}",
                transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#bbbbbb", alpha=0.85, lw=0.6))
    ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right",
            clip_on=False)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0)


def panel_regression(ax, x, y, xlabel, ylabel, panel_letter, var_desc=None):
    """Scatter + OLS; outliers (IQR k=3) excluded; annotation in emptiest corner."""
    xc, yc, n_excl = drop_outliers_iqr(x, y)
    if len(xc) < 5:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888")
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
        return

    ax.scatter(xc, yc, s=16, alpha=0.55, color=PALETTE[0],
               edgecolors="none", zorder=3)

    r2 = np.nan
    try:
        m, b = np.polyfit(xc, yc, 1)
        xl = np.linspace(xc.min(), xc.max(), 200)
        ax.plot(xl, m * xl + b, color="#CC3333", lw=1.6,
                linestyle="--", zorder=4, alpha=0.85)
        yhat  = m * xc + b
        ss_res = ((yc - yhat) ** 2).sum()
        ss_tot = ((yc - yc.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    except Exception:
        pass

    rho, pval = safe_corr(xc, yc, "spearman")
    parts = []
    if not np.isnan(r2):  parts.append(f"R² = {r2:.3f}")
    if not np.isnan(rho): parts.append(f"ρ = {rho:.3f}")
    if not np.isnan(pval): parts.append(f"p = {pval:.2e}")
    parts.append(f"n = {len(xc)}")
    annot = ",  ".join(parts)

    # Place stats below the axes, clear of the x-axis label
    ax.text(0.5, -0.16, annot.strip(), transform=ax.transAxes,
            fontsize=7, color="#333", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#bbbbbb", alpha=0.93, lw=0.6))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right",
            clip_on=False)
    leg = ax.get_legend()
    if leg:
        leg.remove()

    if var_desc:
        ax.text(0.5, -0.30, var_desc, transform=ax.transAxes,
                fontsize=6.2, color="#555", ha="center", va="top",
                style="italic",
                bbox=dict(boxstyle="square,pad=0.15", fc="#f8f8f8",
                          ec="none", alpha=0.85))


def panel_gly_violin(ax, merged, gly_col, col_prefix, title, panel_letter,
                     top_n=10, y_transform="log10p1"):
    """
    Grouped violin + box overlay + jitter per TE superfamily/family,
    split by Glyphosate resistance (Susceptible vs Resistant).
    Pairs are plotted side-by-side for easy comparison.
    Mann-Whitney U significance shown above each pair.

    y_transform:
        "none"    -> raw counts
        "log10p1" -> plot log10(count + 1), stats still on raw counts
    """
    cols = get_te_cols(merged, col_prefix)
    if not cols or gly_col not in merged.columns:
        ax.text(0.5, 0.5, "Data not available", ha="center",
                transform=ax.transAxes)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
        return

    top_cols = (merged[cols].mean()
                .sort_values(ascending=False)
                .head(top_n).index.tolist())

    sub = merged[[gly_col] + top_cols].copy()
    sub[gly_col] = (sub[gly_col].astype(str).str.strip()
                    .map({"0": "Susceptible", "1": "Resistant",
                          "0.0": "Susceptible", "1.0": "Resistant"}))
    sub = sub[sub[gly_col].notna()]

    GROUPS      = ["Susceptible", "Resistant"]
    GRP_COLORS  = [PALETTE[0], PALETTE[3]]
    GAP         = 0.22
    V_WIDTH     = 0.36
    BOX_WIDTH   = 0.10

    n_cats = len(top_cols)
    rng    = np.random.default_rng(42)
    y_tops = np.zeros(n_cats)

    def _disp(vals):
        vals = np.asarray(vals, dtype=float)
        if y_transform == "log10p1":
            return np.log10(vals + 1.0)
        return vals

    for gi, (grp, color) in enumerate(zip(GROUPS, GRP_COLORS)):
        sub_g     = sub[sub[gly_col] == grp]
        n_grp     = len(sub_g)
        positions = np.arange(n_cats) + (GAP if gi == 1 else -GAP)

        raw_data  = [sub_g[fc].dropna().values for fc in top_cols]
        plot_data = [_disp(d) for d in raw_data]

        # violin
        v_data = [(d, p) for d, p in zip(plot_data, positions) if len(d) >= 5]
        if v_data:
            try:
                vd, vp_pos = zip(*v_data)
                vp = ax.violinplot(list(vd), positions=list(vp_pos),
                                   widths=V_WIDTH,
                                   showmeans=False, showmedians=False,
                                   showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.45)
                    pc.set_edgecolor("white")
                    pc.set_linewidth(0.4)
            except Exception:
                pass

        # box overlay
        ax.boxplot(plot_data, positions=positions, widths=BOX_WIDTH,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color="white", linewidth=1.8),
                   boxprops=dict(facecolor=color, linewidth=0, alpha=0.9),
                   whiskerprops=dict(linewidth=0.8, color=color),
                   capprops=dict(linewidth=0.8, color=color))

        # jitter
        for pos, d in zip(positions, plot_data):
            if len(d):
                jx = np.full(len(d), pos) + rng.normal(0, 0.06, len(d))
                ax.scatter(jx, d, alpha=0.22, s=4,
                           color=color, zorder=4, linewidths=0)

        # track tops for significance label placement
        for ci, d in enumerate(plot_data):
            if len(d):
                y_tops[ci] = max(y_tops[ci], np.percentile(d, 98))

        if gi == 0:
            n_susceptible = n_grp
        else:
            n_resistant = n_grp

    # significance testing remains on RAW counts (Bonferroni-corrected)
    y_range = y_tops.max() - y_tops.min() if y_tops.max() > 0 else 1
    raw_pvals = []
    for ci, fc in enumerate(top_cols):
        s_d = sub[sub[gly_col] == "Susceptible"][fc].dropna().values
        r_d = sub[sub[gly_col] == "Resistant"][fc].dropna().values
        if len(s_d) >= 3 and len(r_d) >= 3:
            try:
                _, pval = stats.mannwhitneyu(s_d, r_d, alternative="two-sided")
                raw_pvals.append((ci, pval))
            except Exception:
                pass

    n_tests_mw = len(raw_pvals)
    for ci, pval in raw_pvals:
        padj = min(pval * n_tests_mw, 1.0)
        sig = ("***" if padj < 0.001 else "**" if padj < 0.01
               else "*" if padj < 0.05 else "")
        if sig:
            ax.text(ci, y_tops[ci] + y_range * 0.05, sig,
                    ha="center", va="bottom", fontsize=8,
                    color="#333", fontweight="bold")

    x_label = "LTR-RT Family" if "clade" in col_prefix else "TE Superfamily"
    cat_labels = [fc.replace(col_prefix, "") for fc in top_cols]
    ax.set_xticks(np.arange(n_cats))
    ax.set_xticklabels(cat_labels, fontsize=7.5, rotation=30, ha="right")
    ax.set_xlabel(x_label)

    if y_transform == "log10p1":
        tick_vals = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)
        tick_pos  = np.log10(tick_vals + 1.0)
        ymax = ax.get_ylim()[1]
        keep = tick_pos <= ymax + 1e-9
        ax.set_yticks(tick_pos[keep])
        ax.set_yticklabels([str(int(v)) for v in tick_vals[keep]])
        ax.set_ylabel("TE Insertions per Sample")
    else:
        ax.set_ylabel("TE Insertions per Sample")

    ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right",
            clip_on=False)

    n_susceptible = n_susceptible if "n_susceptible" in dir() else "?"
    n_resistant   = n_resistant   if "n_resistant"   in dir() else "?"
    legend_patches = [
        Patch(facecolor=GRP_COLORS[0], alpha=0.72,
              label=f"Susceptible  (n={n_susceptible})"),
        Patch(facecolor=GRP_COLORS[1], alpha=0.72,
              label=f"Resistant  (n={n_resistant})"),
    ]
    ax.legend(handles=legend_patches, fontsize=7,
              loc="upper right", framealpha=0.88, edgecolor="#cccccc")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0)


def panel_bioclim_correlation(ax_bar, ax_key, merged, panel_letter="C"):
    """Horizontal Spearman ρ bars + side legend key.
    Colors: blue = significant positive, red = significant negative,
            grey = not significant after Bonferroni correction.
    """
    bio_cols = sorted([c for c in merged.columns if c.startswith("Bio")])
    if not bio_cols:
        ax_bar.text(0.5, 0.5, "No BioClim columns found", ha="center",
                    transform=ax_bar.transAxes)
        return

    y_total = pd.to_numeric(merged["n_te_total"], errors="coerce")
    corrs = []
    for col in bio_cols:
        s = pd.to_numeric(merged[col], errors="coerce")
        # Remove IQR outliers on n_te_total — consistent with page 2 regressions
        s_clean, y_clean, _ = drop_outliers_iqr(s, y_total, k=3.0)
        rho, pval = safe_corr(s_clean, y_clean)
        if not np.isnan(rho):
            corrs.append({"col": col, "rho": rho, "p": pval})
    if not corrs:
        return

    cdf     = pd.DataFrame(corrs).sort_values("rho", ascending=True).reset_index(drop=True)
    n_tests = len(cdf)

    # Bonferroni-adjusted p per row
    cdf["p_adj"] = (cdf["p"] * n_tests).clip(upper=1.0)

    # Color: sig+positive → blue, sig+negative → red, not sig → grey
    def _bar_color(row):
        if row["p_adj"] >= 0.05:
            return "#aaaaaa"
        return PALETTE[0] if row["rho"] >= 0 else PALETTE[3]

    colors = cdf.apply(_bar_color, axis=1).tolist()

    ax_bar.barh(cdf["col"], cdf["rho"], color=colors,
                edgecolor="none", height=0.72)
    ax_bar.axvline(0, color="black", lw=0.6)
    ax_bar.set_xlabel("Spearman ρ  (vs Total TE Insertions)")
    ax_bar.set_ylabel("BioClim Variable")
    ax_bar.tick_params(axis="y", labelsize=6.2)
    ax_bar.text(-0.06, 1.04, panel_letter, transform=ax_bar.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.grid(axis="y", alpha=0)

    # Significance asterisks on the bar ends (Bonferroni-corrected)
    for i, row in cdf.iterrows():
        sig = ("***" if row["p_adj"] < 0.001 else "**" if row["p_adj"] < 0.01
               else "*" if row["p_adj"] < 0.05 else "")
        if sig:
            offset = 0.004 if row["rho"] >= 0 else -0.004
            ax_bar.text(row["rho"] + offset, i, sig,
                        va="center",
                        ha="left" if row["rho"] >= 0 else "right",
                        fontsize=5.8, color="#222")

    # Key panel — colored to match bars (grey = not sig)
    ax_key.axis("off")
    sig_map = {row["col"]: (row["rho"], row["p_adj"]) for _, row in cdf.iterrows()}
    present = sorted((c for c in sig_map if c in BIO_KEY))
    if present:
        ax_key.text(0.03, 0.98, "Var.    Description",
                    transform=ax_key.transAxes, va="top", ha="left",
                    fontsize=5.8, family="monospace",
                    color="#333", fontweight="bold")
        step = 0.95 / (len(present) + 1)
        for i, c in enumerate(present):
            rho_val, p_val_adj = sig_map[c]
            if p_val_adj >= 0.05:
                col_color = "#888888"
            elif rho_val >= 0:
                col_color = PALETTE[0]
            else:
                col_color = PALETTE[3]
            y = 0.98 - (i + 1.5) * step
            ax_key.text(0.03, y, f"{c:<7}{BIO_KEY[c]}",
                        transform=ax_key.transAxes, va="top", ha="left",
                        fontsize=5.4, family="monospace", color=col_color)


def panel_sample_stacked(ax, merged, col_prefix="te_fam_", top_n=10,
                         panel_letter="", panel_letter_y=1.015):
    """
    Horizontal stacked bar per sample sorted by stacked total (ascending).
    Glyphosate-resistant samples highlighted with a light-red background band.
    col_prefix: 'te_fam_' for families, 'te_clade_' for LTR-RT clades.
    top_n: number of top categories to show individually; None = show all.
    """
#    seg_cols = [c for c in merged.columns if c.startswith(col_prefix)
#                and c != col_prefix + "mixture"]
    seg_cols = get_te_cols(merged, col_prefix)
    if not seg_cols:
        ax.text(0.5, 0.5, f"No {col_prefix} data", ha="center",
                transform=ax.transAxes)
        return

    all_means = merged[seg_cols].mean().sort_values(ascending=False)
    if top_n is None:
        top_segs   = all_means.index.tolist()
        other_cols = []
    else:
        top_segs   = all_means.head(top_n).index.tolist()
        other_cols = [c for c in seg_cols if c not in top_segs]
    all_seg = top_segs + other_cols

    # Sort by sum of ALL plotted segments so visual length matches sort order
    row_totals = merged[all_seg].fillna(0).sum(axis=1)
    sorted_idx = row_totals.sort_values(ascending=True).index

    n  = len(sorted_idx)
    y  = np.arange(n)
    bh = 1.0   # full-unit height = no gap between bars

    # Glyphosate resistance
    gly_res = pd.Series(False, index=merged.index)
    for gc in ["Glyphosate_R", "glyphosate_res"]:
        if gc in merged.columns:
            gly_res |= (merged[gc].astype(str).str.strip()
                        .isin(["1", "1.0", "R", "Resistant"]))
    gly_arr = gly_res.reindex(sorted_idx).fillna(False).values

    # Background bands for resistant samples
    for i, is_res in enumerate(gly_arr):
        if is_res:
            ax.axhspan(i - 0.5, i + 0.5,
                       color="#ffdddd", alpha=0.55, zorder=0)

    pal   = CLADE_PALETTE if "clade" in col_prefix else FAM_PALETTE
    lefts = np.zeros(n)
    for fi, fc in enumerate(top_segs):
        vals = merged.loc[sorted_idx, fc].fillna(0).values
        ax.barh(y, vals, left=lefts, height=bh,
                color=pal[fi % len(pal)],
                label=fc.replace(col_prefix, ""),
                edgecolor="none")
        lefts += vals

    if other_cols:
        ov = merged.loc[sorted_idx, other_cols].fillna(0).sum(axis=1).values
        ax.barh(y, ov, left=lefts, height=bh,
                color="#cccccc", label="Other", edgecolor="none")

    # No Y-axis tick labels — pink bands are the only per-sample indicator
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylabel("Sample")

    group_name = "Superfamily" if "fam" in col_prefix else "Family"
    te_label   = "LTR-RT" if "clade" in col_prefix else "TE"
    ax.set_xlabel(f"Total {te_label} Insertions")
    if panel_letter:
#        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
        ax.text(-0.06, panel_letter_y, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
    ax.legend(title=group_name, fontsize=6.5, title_fontsize=7,
              loc="lower right", ncol=2)
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", alpha=0)


def panel_karyotype(ax, clusters_df, pos_df, fai_lengths=None, crm_intervals=None):
    """
    Linear karyotype: one horizontal bar per chromosome (stacked vertically).
    R-enriched insertions → red ticks above bar.
    S-enriched insertions → blue ticks below bar.
    Non-significant clusters → hairline grey marks on bar.
    Tick height scales with −log10(padj), capped at 5.

    fai_lengths   : dict {chr_name: length} from a samtools FAI file.
                    When provided, used as authoritative chromosome lengths.
                    Otherwise lengths are inferred from max observed End position.
    crm_intervals : dict {chr_name: [(start, end), ...]} CRM element coordinates.
                    Drawn as semi-transparent purple bands on the chromosome bar
                    to approximate centromere locations.
    """
    # ── chromosome lengths ───────────────────────────────────────────────────
    if fai_lengths:
        chr_lens = pd.Series(fai_lengths, dtype=int)
        # Restrict to chromosomes that actually have data (or all FAI entries)
        if not pos_df.empty:
            observed = set(pos_df["Chr"].unique())
            if not clusters_df.empty:
                observed |= set(clusters_df["Chr"].unique())
            chr_lens = chr_lens[chr_lens.index.isin(observed)]
            if chr_lens.empty:       # FAI names don't match BED names — fall back
                chr_lens = pos_df.groupby("Chr")["End"].max().astype(int)
    elif not pos_df.empty:
        chr_lens = pos_df.groupby("Chr")["End"].max().astype(int)
    elif not clusters_df.empty:
        chr_lens = clusters_df.groupby("Chr")["pos"].max().astype(int)
    else:
        ax.text(0.5, 0.5, "No position data available",
                ha="center", va="center", transform=ax.transAxes)
        return

    chroms  = sorted(chr_lens.index.tolist(), key=_chrom_sort_key)
    n_chr   = len(chroms)
    max_len = int(chr_lens.max())

    BAR_H  = 0.22   # half-height of chromosome rectangle
    TICK_H = 0.38   # max tick height above/below bar (scales with −log10 p)

    ax.set_xlim(-max_len * 0.09, max_len * 1.01)
    ax.set_ylim(-1.2, n_chr + 0.3)
    ax.axis("off")

    for yi, chrom in enumerate(chroms[::-1]):   # first chrom at top
        y       = yi
        chr_len = int(chr_lens[chrom])

        # Chromosome bar
        ax.fill_betweenx([y - BAR_H, y + BAR_H], 0, chr_len,
                         color="#dde0e8", zorder=1, linewidth=0)
        for yy in [y - BAR_H, y + BAR_H]:
            ax.plot([0, chr_len], [yy, yy], color="#aaaaaa", lw=0.25, zorder=2)

        # Chromosome label — use full accession name
        ax.text(-max_len * 0.01, y, chrom,
                ha="right", va="center", fontsize=6.5, color="#333")

        # CRM density heatmap: bin midpoints → Gaussian-smooth → imshow on bar.
        # Dark purple = dense CRMs → likely centromere; near-white = sparse.
        if crm_intervals and chrom in crm_intervals:
            from scipy.ndimage import gaussian_filter1d as _gf1d
            N_BINS = 400
            midpoints = [(s + e) / 2 for s, e in crm_intervals[chrom]]
            counts, _ = np.histogram(midpoints, bins=N_BINS, range=(0, chr_len))
            smoothed = _gf1d(counts.astype(float), sigma=N_BINS * 0.02)
            if smoothed.max() > 0:
                smoothed /= smoothed.max()
            ax.imshow(
                smoothed[np.newaxis, :],
                aspect="auto",
                extent=[0, chr_len, y - BAR_H * 0.92, y + BAR_H * 0.92],
                cmap="Purples",
                vmin=0, vmax=1,
                zorder=2,
                origin="lower",
                alpha=0.45,
            )

        if clusters_df.empty:
            continue

        chr_cl = clusters_df[clusters_df["Chr"] == chrom]
        if chr_cl.empty:
            continue

        r_cl     = chr_cl[chr_cl["enrichment"] == "R"]
        s_cl     = chr_cl[chr_cl["enrichment"] == "S"]
        other_cl = chr_cl[chr_cl["enrichment"] == "shared"]

        # Shared / non-significant: very faint hairlines on the bar
        if not other_cl.empty:
            ax.vlines(other_cl["pos"].values,
                      y - BAR_H * 0.55, y + BAR_H * 0.55,
                      color="#bbbbbb", lw=0.25, alpha=0.35, zorder=2)

        # R-enriched: red ticks above (height ∝ −log10 padj, capped at 5)
        if not r_cl.empty:
            lp = np.clip(-np.log10(r_cl["padj"].clip(lower=1e-30).values), 0, 5)
            ax.vlines(r_cl["pos"].values,
                      y + BAR_H, y + BAR_H + TICK_H * lp / 5,
                      color="#CC2222", lw=1.4, alpha=0.90, zorder=3)

        # S-enriched: blue ticks below
        if not s_cl.empty:
            lp = np.clip(-np.log10(s_cl["padj"].clip(lower=1e-30).values), 0, 5)
            ax.vlines(s_cl["pos"].values,
                      y - BAR_H - TICK_H * lp / 5, y - BAR_H,
                      color="#2255CC", lw=1.4, alpha=0.9, zorder=3)

    # ── known variant marker (NC_057763.1 : 38,804,274) ─────────────────────
    KNOWN_VAR = {"NC_057763.1": 38_804_274}
    chroms_rev = chroms[::-1]   # same order as the drawing loop (yi = index here)
    for kchrom, kpos in KNOWN_VAR.items():
        if kchrom in chroms_rev:
            ky = chroms_rev.index(kchrom)   # matches yi in the draw loop
            ax.annotate(
                "",
                xy=(kpos, ky + BAR_H),
                xytext=(kpos, ky + BAR_H + TICK_H + 0.18),
                arrowprops=dict(arrowstyle="-|>", color="#FF8C00",
                                lw=1.4, mutation_scale=8),
                zorder=6,
            )

    # ── scale bar ────────────────────────────────────────────────────────────
    if max_len >= 10_000_000:
        scale, slabel = 10_000_000, "10 Mb"
    elif max_len >= 1_000_000:
        scale, slabel = 1_000_000, "1 Mb"
    else:
        scale, slabel = max_len // 10, f"{max_len // 10 // 1000} kb"
    ax.plot([0, scale], [-0.85, -0.85], color="#333", lw=1.5)
    ax.text(scale / 2, -0.95, slabel, ha="center", va="top",
            fontsize=6.5, color="#333")

    # ── summary stats and legend ─────────────────────────────────────────────
    n_R_enr = int((clusters_df["enrichment"] == "R").sum()) if not clusters_df.empty else 0
    n_S_enr = int((clusters_df["enrichment"] == "S").sum()) if not clusters_df.empty else 0
    n_shr   = int((clusters_df["enrichment"] == "shared").sum()) if not clusters_df.empty else 0

    legend_elements = [
        Line2D([0], [0], color="#CC2222", lw=1.5,
               label=f"R-enriched  (n={n_R_enr}, BH-FDR < 0.05)"),
        Line2D([0], [0], color="#2255CC", lw=1.5,
               label=f"S-enriched  (n={n_S_enr}, BH-FDR < 0.05)"),
        Line2D([0], [0], color="#bbbbbb", lw=1.0,
               label=f"Shared / not significant  (n={n_shr})"),
        Line2D([0], [0], color="#FF8C00", lw=1.4,
               marker="v", markersize=6, markerfacecolor="#FF8C00",
               label="Known R-variant (NC_057763.1:38,804,274)"),
    ]
    if crm_intervals:
        n_crm = sum(len(v) for v in crm_intervals.values())
        legend_elements.append(
            Patch(facecolor="#9B59B6", alpha=0.85, edgecolor="none",
                  label=f"CRM density  (n={n_crm})")
        )
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=7.5, framealpha=0.92, edgecolor="#cccccc",
              bbox_to_anchor=(1.0, 0.0))



def panel_gene_disruption(ax, disruption_df, merged, n_genes=1,
                          panel_letter="A", region_order=None):
    """Grouped horizontal boxplot: exonic vs intronic disruption *frequency* per region.
    Frequency = disrupted genes per sample / total genes in GFF.
    """
    if disruption_df.empty or n_genes < 1:
        ax.text(0.5, 0.5, "No disruption data", ha="center",
                transform=ax.transAxes)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
        return

    reg_col = "Region_new" if "Region_new" in merged.columns else None
    if reg_col is None:
        ax.text(0.5, 0.5, "No Region column found", ha="center",
                transform=ax.transAxes)
        return

    df = disruption_df.copy()
    df["freq_exonic"]   = df["n_exonic"]   / n_genes
    df["freq_intronic"] = df["n_intronic"] / n_genes
    df[reg_col] = df["sample_id"].map(merged[reg_col])
    df = df[df[reg_col].notna() & (df[reg_col].astype(str).str.lower() != "nan")]
    if df.empty:
        ax.text(0.5, 0.5, "No region data", ha="center", transform=ax.transAxes)
        return

    if region_order is not None:
        order = [r for r in region_order if r in df[reg_col].values]
    else:
        medians = (df.groupby(reg_col)[["freq_exonic", "freq_intronic"]]
                   .sum().sum(axis=1).sort_values(ascending=True))
        order = medians.index.tolist()

    EXON_COL   = PALETTE[0]
    INTRON_COL = PALETTE[1]
    GAP = 0.22
    rng = np.random.default_rng(42)

    for ri, region in enumerate(order):
        sub = df[df[reg_col] == region]
        for gi, (col, color) in enumerate([
            ("freq_exonic",   EXON_COL),
            ("freq_intronic", INTRON_COL),
        ]):
            vals = sub[col].dropna().values
            y_off = ri + 1 + (GAP if gi == 1 else -GAP)
            if len(vals) == 0:
                continue
            ax.boxplot([vals], positions=[y_off], widths=0.28,
                       vert=False, patch_artist=True, showfliers=False,
                       medianprops=dict(color="black", linewidth=1.4),
                       boxprops=dict(facecolor=color, linewidth=0.6, alpha=0.8),
                       whiskerprops=dict(linewidth=0.7, color=color),
                       capprops=dict(linewidth=0.7, color=color))
            jitter = rng.normal(0, 0.06, len(vals))
            ax.scatter(vals, np.full(len(vals), y_off) + jitter,
                       alpha=0.30, s=5, color=color, zorder=4, linewidths=0)

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order,
                       fontsize=max(5, min(8, 120 // max(len(order), 1))))
    ax.set_xlabel("Disrupted Gene Frequency per Sample")
    ax.set_ylabel("Region")
    ax.set_xlim(left=0)
    ax.legend(handles=[
        Patch(facecolor=EXON_COL,   alpha=0.8, label="Exonic"),
        Patch(facecolor=INTRON_COL, alpha=0.8, label="Intronic"),
    ], fontsize=7, loc="lower right", framealpha=0.88, edgecolor="#cccccc")
    ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right",
            clip_on=False)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0)


def panel_context_bar(ax, comp_df, palette, panel_letter, top_n=10, label=None):

    """Stacked proportional horizontal bar chart: gene-context × TE superfamily/family.

    comp_df rows = contexts (Exonic, Intronic, Upstream 2kb, Downstream 2kb).
    comp_df cols = superfamily or family names.
    """
    CONTEXTS = [c for c in CONTEXT_ORDER if c in comp_df.index]
    present  = [c for c in CONTEXTS if c in comp_df.index]
    if not present:
        ax.text(0.5, 0.5, "No context data", ha="center", transform=ax.transAxes)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right", clip_on=False)
        return

    df = comp_df.loc[present].copy()

    # Select top_n columns by total count; remainder → "Other"
    col_totals = df.sum(axis=0).sort_values(ascending=False)
    top_cols   = col_totals.head(top_n).index.tolist()
    other_cols = [c for c in df.columns if c not in top_cols]

    plot_df = df[top_cols].copy()
    if other_cols:
        plot_df["Other"] = df[other_cols].sum(axis=1)

    # Normalise to proportions
    row_sums = plot_df.sum(axis=1).replace(0, 1)
    prop_df  = plot_df.div(row_sums, axis=0)

    colors = list(palette[:len(top_cols)])
    if "Other" in prop_df.columns:
        colors.append("#cccccc")

    left = np.zeros(len(present))
    for j, col in enumerate(prop_df.columns):
        vals = prop_df[col].values
        ax.barh(present, vals, left=left, color=colors[j],
                height=0.65, linewidth=0)
        left += vals

    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion")
    if label:
        ax.set_ylabel(label)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    handles = [Patch(facecolor=colors[j], label=col)
               for j, col in enumerate(prop_df.columns)]
    ax.legend(handles=handles, fontsize=6, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              framealpha=0.88, edgecolor="#cccccc")

    ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right", clip_on=False)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0)

CONTEXTS = CONTEXT_ORDER

def panel_context_enrichment(ax, comp_df, panel_letter, top_n=10, label=None,
                             show_xticklabels=True, top_cats_override=None,
                             context_lengths_kb=None):
    """Log₂-enrichment heatmap: observed vs expected context proportion per category.

    comp_df rows = contexts, columns = family or clade names (raw counts).
    Cells show log₂(observed_proportion / expected_proportion).
    Expected = genome-wide average context distribution across all categories.
    """
    CONTEXTS = [c for c in CONTEXT_ORDER if c in comp_df.index]
    if not CONTEXTS or comp_df.empty:
        ax.text(0.5, 0.5, "No context data", ha="center", transform=ax.transAxes)
        if panel_letter:
            ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom", ha="right", clip_on=False)
        return

    df = comp_df.loc[CONTEXTS].copy()

    # Select top_n columns by total count
    col_totals = df.sum(axis=0).sort_values(ascending=False)
    top_cols = top_cats_override if top_cats_override else col_totals.head(top_n).index.tolist()
    # Filter to columns that actually exist in df
    top_cols = [c for c in top_cols if c in df.columns]
    df = df[top_cols]

    # Length-normalise to insertion density (per kb) so that exons/introns
    # (variable length) are directly comparable to fixed up/downstream windows.
    if context_lengths_kb:
        df = df.astype(float)
        for ctx in list(df.index):
            if ctx in context_lengths_kb:
                df.loc[ctx] = df.loc[ctx] / context_lengths_kb[ctx]

    # Expected proportions: genome-wide average across all categories
    genome_total = df.sum(axis=1)  # total density per context
    expected = genome_total / genome_total.sum()
    expected = expected.clip(lower=1e-10)

    # Observed proportions per category (column-normalised)
    cat_totals = df.sum(axis=0).replace(0, 1)
    observed = df.div(cat_totals, axis=1)

    # Log2 fold-change
    log2fc = np.log2(observed.div(expected, axis=0).clip(lower=1e-6))
    # Clamp to ±3 for visual sanity
    log2fc = log2fc.clip(-3, 3)

    # Plot heatmap
    from matplotlib.colors import TwoSlopeNorm
    vmax = max(abs(log2fc.values.min()), abs(log2fc.values.max()), 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(log2fc.values, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="nearest")

    # Annotate cells with log2FC values
    for i in range(len(CONTEXTS)):
        for j in range(len(top_cols)):
            val = log2fc.values[i, j]
            color = "white" if abs(val) > vmax * 0.65 else "#222222"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")

    ax.set_xticks(np.arange(len(top_cols)))
    if show_xticklabels:
        ax.set_xticklabels(top_cols, fontsize=6.5, rotation=35, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(CONTEXTS)))
    ax.set_yticklabels(CONTEXTS, fontsize=6)
    if label and show_xticklabels:
        ax.set_xlabel(label, fontsize=8)

    # Colorbar as inset — does not steal width from the heatmap axes,
    # so the heatmap plot area stays aligned with the violin above.
    cax = ax.inset_axes([1.02, 0.0, 0.03, 1.0])   # [x0, y0, width, height]
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("log₂(obs/exp)", fontsize=5.5, labelpad=2)
    cbar.ax.tick_params(labelsize=5)

    if panel_letter:
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right", clip_on=False)

    return top_cols


CONTEXT_COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#999999"]


def panel_context_prop_violin(ax, long_df, panel_letter, top_n=10, label=None,
                              show_xticklabels=True, top_cats_override=None,
                              context_lengths_kb=None):
    """Grouped violin plot of per-sample context *proportions* per category.

    For each sample × category, the raw counts are normalised to proportions
    (sum-to-1 within that sample–category pair), so every superfamily/family gets
    equal visual weight regardless of abundance.

    long_df columns: [sample_id, context, category, count].
    """
    if long_df.empty:
        ax.text(0.5, 0.5, "No context data", ha="center", transform=ax.transAxes)
        if panel_letter:
            ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom", ha="right", clip_on=False)
        return

    # Rank categories by total count, keep top_n
    cat_totals = long_df.groupby("category")["count"].sum().sort_values(ascending=False)
    top_cats = top_cats_override if top_cats_override else cat_totals.head(top_n).index.tolist()
    df = long_df[long_df["category"].isin(top_cats)].copy()

    # Pivot to wide: rows = (sample_id, category), cols = context
    pivot = df.pivot_table(index=["sample_id", "category"],
                           columns="context", values="count",
                           fill_value=0, aggfunc="sum")
    # Length-normalise to insertion density (per kb) before computing proportions,
    # so that fixed-length upstream/downstream regions are comparable to variable-
    # length exons and introns.
    if context_lengths_kb:
        for ctx in list(pivot.columns):
            if ctx in context_lengths_kb:
                pivot[ctx] = pivot[ctx] / context_lengths_kb[ctx]
    # Normalise each row to proportions
    row_sums = pivot.sum(axis=1).replace(0, 1)
    prop = pivot.div(row_sums, axis=0)
    prop = prop.reset_index()

    present_ctx = [c for c in CONTEXT_ORDER if c in prop.columns]
    if not present_ctx:
        ax.text(0.5, 0.5, "No context data", ha="center", transform=ax.transAxes)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right", clip_on=False)
        return

    n_cats = len(top_cats)
    n_ctx  = len(present_ctx)
    GAP    = 0.8 / n_ctx
    V_WIDTH = GAP * 0.85
    BOX_WIDTH = GAP * 0.22
    rng = np.random.default_rng(42)

    for ctx, color in zip(CONTEXT_ORDER, CONTEXT_COLORS):
        if ctx not in present_ctx:
            continue
        ctx_i = present_ctx.index(ctx)
        offset = (ctx_i - (n_ctx - 1) / 2) * GAP
        positions = np.arange(n_cats) + offset

        data = []
        for cat in top_cats:
            vals = prop.loc[prop["category"] == cat, ctx].values
            if len(vals) == 0:
                vals = np.array([0.0])
            data.append(vals.astype(float))

        # Violin
        v_data = [(d, p) for d, p in zip(data, positions) if len(d) >= 5]
        if v_data:
            vd, vp = zip(*v_data)
            try:
                vp_obj = ax.violinplot(list(vd), positions=list(vp),
                                       widths=V_WIDTH,
                                       showmeans=False, showmedians=False,
                                       showextrema=False)
                for pc in vp_obj["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.45)
                    pc.set_edgecolor("white")
                    pc.set_linewidth(0.4)
            except Exception:
                pass

        # Narrow box overlay
        ax.boxplot(data, positions=positions, widths=BOX_WIDTH,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color="white", linewidth=1.4),
                   boxprops=dict(facecolor=color, linewidth=0, alpha=0.9),
                   whiskerprops=dict(linewidth=0.7, color=color),
                   capprops=dict(linewidth=0.7, color=color))

        # Jitter
        for pos, d in zip(positions, data):
            if len(d):
                jx = np.full(len(d), pos) + rng.normal(0, 0.04, len(d))
                ax.scatter(jx, d, alpha=0.20, s=3,
                           color=color, zorder=4, linewidths=0)

    ax.set_xticks(np.arange(n_cats))
    if show_xticklabels:
        ax.set_xticklabels(top_cats, fontsize=6.5, rotation=35, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("Context Proportion\n(normalized density)" if context_lengths_kb
                  else "Context Proportion", fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    if label and show_xticklabels:
        ax.set_xlabel(label, fontsize=8)

    legend_patches = [Patch(facecolor=CONTEXT_COLORS[i], alpha=0.72,
                            label=CONTEXT_ORDER[i])
                      for i in range(len(CONTEXT_ORDER))
                      if CONTEXT_ORDER[i] in present_ctx]
#    ax.legend(handles=legend_patches, fontsize=5.5, loc="upper right",
    ax.legend(handles=legend_patches, fontsize=5.5, loc="upper left",
              framealpha=0.88, edgecolor="#cccccc")

    if panel_letter:
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right", clip_on=False)
    ax.grid(axis="y", alpha=0.20)
    ax.grid(axis="x", alpha=0)


def panel_metagene(axes, metagene_df, merged, panel_letter="D",
                   region_order=None):
    """Stacked bar+whisker metagene plots — one subplot per region.

    axes        : list of Axes, one per region (top to bottom).
    region_order: regions in bottom-to-top display order (same as boxplots);
                  the list is reversed internally so axes[0] = highest median.
    """
    reg_col = "Region_new" if "Region_new" in merged.columns else None

    if metagene_df.empty or reg_col is None or not axes:
        for ax in axes:
            ax.text(0.5, 0.5, "No metagene data", ha="center",
                    transform=ax.transAxes, fontsize=8)
        if axes:
            axes[0].text(-0.06, 1.04, panel_letter, transform=axes[0].transAxes,
                         fontsize=11, fontweight="bold", va="bottom", ha="right",
                         clip_on=False)
        return

    x_abs = int(max(abs(metagene_df["bin_center"].min()),
                    abs(metagene_df["bin_center"].max())) + 25)

    df = metagene_df.copy()
    df[reg_col] = df["sample_id"].map(merged[reg_col])
    df = df[df[reg_col].notna() & (df[reg_col].astype(str).str.lower() != "nan")]

    if region_order is not None:
        present = set(df[reg_col].dropna().unique())
        # Filter while preserving canonical order, then reverse so that
        # axes[0] (top strip) = highest-median region — matches boxplot visual order.
        regions = [r for r in region_order if r in present][::-1]
    else:
        regions = sorted(df[reg_col].dropna().unique(), reverse=True)
    bin_size = 50  # must match compute_metagene_profile bin_size

    # Fixed y-range and ticks per user spec
    Y_MAX   = 0.0009
    Y_TICKS = [0.0000, 0.0004, 0.0008]
    Y_LABELS = ["0.0000", "0.0004", "0.0008"]

    for ri, (region, ax) in enumerate(zip(regions, axes)):
        sub = df[df[reg_col] == region]
        grp = sub.groupby("bin_center")["freq"].agg(["mean", "std", "count"])
        grp["se"] = grp["std"] / np.sqrt(grp["count"].clip(lower=1))
        bc  = grp.index.values
        mn  = grp["mean"].values
        se  = grp["se"].values
        color = PALETTE[ri % len(PALETTE)]

        ax.bar(bc, mn, width=bin_size - 1, color=color, alpha=0.75,
               edgecolor="none", align="center")
        ax.errorbar(bc, mn, yerr=se, fmt="none", ecolor="#333333",
                    elinewidth=0.6, capsize=1.5, capthick=0.6)
        ax.axvline(0, color="#555555", lw=0.9, linestyle="--", zorder=5)
        ax.set_ylim(0, Y_MAX)
        ax.set_xlim(-x_abs, x_abs)
        ax.set_yticks(Y_TICKS)
        ax.set_yticklabels(Y_LABELS)

        # Region label inside plot (upper-right)
        ax.text(0.98, 0.92, region, transform=ax.transAxes,
                fontsize=7, ha="right", va="top", color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.7))

        # Upstream vs downstream stat test
        try:
            up_sums, dn_sums = [], []
            for sid, sgrp in sub.groupby("sample_id"):
                up_sums.append(sgrp.loc[sgrp["bin_center"] < 0, "freq"].sum())
                dn_sums.append(sgrp.loc[sgrp["bin_center"] > 0, "freq"].sum())
            up_arr, dn_arr = np.array(up_sums), np.array(dn_sums)
            if len(up_arr) >= 3 and len(dn_arr) >= 3:
                _, pval = stats.mannwhitneyu(up_arr, dn_arr,
                                             alternative="two-sided")
                direction = ("Up > Dn" if np.median(up_arr) > np.median(dn_arr)
                             else "Dn > Up")
                sig = ("***" if pval < 0.001 else "**" if pval < 0.01
                       else "*" if pval < 0.05 else "ns")
                ax.text(0.98, 0.72, f"{direction}  p={pval:.3g} {sig}",
                        transform=ax.transAxes, fontsize=5.5,
                        ha="right", va="top", color="#555555",
                        fontstyle="italic")
        except Exception:
            pass

        ax.tick_params(axis="y", labelsize=6)

        # X-axis: only show ticks/label on bottom subplot; no "Gene boundary" text
        if ri < len(regions) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Distance from Gene Boundary (bp)")
            ax.tick_params(axis="x", labelsize=7)

        ax.grid(axis="y", alpha=0.20)
        ax.grid(axis="x", alpha=0)

    # Single shared y-axis label on the middle strip
    if axes:
        mid = len(regions) // 2
        axes[min(mid, len(axes) - 1)].set_ylabel(
            "Frequency of TE insertion in bin (50 bp)", fontsize=7, labelpad=6)

    # Panel letter on top subplot
    if axes:
        axes[0].text(-0.06, 1.04, panel_letter, transform=axes[0].transAxes,
                     fontsize=11, fontweight="bold", va="bottom", ha="right",
                     clip_on=False)


def _canonical_region_order(merged, reg_col="Region_new"):
    """Regions sorted by median n_te_total ascending — same rule as panel_boxplot."""
    if reg_col not in merged.columns or "n_te_total" not in merged.columns:
        return None
    sub = merged[[reg_col, "n_te_total"]].copy()
    sub[reg_col] = sub[reg_col].astype(str)
    sub = sub[sub[reg_col].notna() & (sub[reg_col].str.lower() != "nan")].dropna()
    if sub.empty:
        return None
    return (sub.groupby(reg_col)["n_te_total"]
            .median().sort_values(ascending=True).index.tolist())


# ── Page builders ────────────────────────────────────────────────────────────

def build_page1(pdf, merged):
    """Page 1: family bar, superfamily bar, region boxplot, gly boxplot."""
    fig = plt.figure(figsize=(11, 8.5))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.35, wspace=0.42, # EDIT 0.40 to shift. Was 0.52
                            left=0.10, right=0.97,
                            top=0.97, bottom=0.08)

    panel_bar(fig.add_subplot(gs[0, 0]), merged,
              "te_fam_", "Mean TE Insertions by Superfamily", "A",
              top_n=None)
    panel_bar(fig.add_subplot(gs[0, 1]), merged,
              "te_clade_", "Mean LTR-RT Insertions by Family", "B",
              top_n=None)
    panel_boxplot(fig.add_subplot(gs[1, 0]), merged, "Region_new",
                  "Total TE Insertions by Region", "C",
                  ylabel="Region")
    panel_boxplot(fig.add_subplot(gs[1, 1]), merged, "Glyphosate_R",
                  "Total TE Insertions by Glyphosate Resistance", "D",
                  label_map={"0": "Susceptible", "1": "Resistant",
                             "0.0": "Susceptible", "1.0": "Resistant"},
                  ylabel="Glyphosate Resistance")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_page2(pdf, merged):
    """Page 2: three regression panels — top-3 variables by R² vs n_te_total."""
    # Candidate variables: all Bio01–Bio40 present in merged + CWD_All_Annual
    candidates = [f"Bio{i:02d}" for i in range(1, 41)] + ["CWD_All_Annual"]
    y = pd.to_numeric(merged["n_te_total"], errors="coerce")

    def _r2(col_name):
        x = pd.to_numeric(merged.get(col_name, pd.Series(dtype=float)),
                          errors="coerce")
        xc, yc, _ = drop_outliers_iqr(x, y)
        if len(xc) < 5:
            return np.nan
        try:
            m, b = np.polyfit(xc, yc, 1)
            yhat = m * xc + b
            ss_res = ((yc - yhat) ** 2).sum()
            ss_tot = ((yc - yc.mean()) ** 2).sum()
            return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except Exception:
            return np.nan

    r2_scores = [(v, _r2(v)) for v in candidates]
    r2_scores = [(v, r) for v, r in r2_scores if not np.isnan(r)]
    r2_scores.sort(key=lambda x: x[1], reverse=True)
    top6 = r2_scores[:6]

    def _col(name):
        return pd.to_numeric(merged.get(name, pd.Series(dtype=float)),
                             errors="coerce")

    fig = plt.figure(figsize=(11, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            wspace=0.42, hspace=0.52, # EDIT 0.52 to shift. was 0.72.
                            left=0.07, right=0.97,
                            top=0.92, bottom=0.18)

    for idx, (var, _) in enumerate(top6):
        row, col = divmod(idx, 3)
        label = "CWD Annual" if var == "CWD_All_Annual" else var
        panel_regression(fig.add_subplot(gs[row, col]),
                         _col(var), y, label, "Total TE Insertions",
                         chr(65 + idx),
                         var_desc=VAR_DESC.get(var, var))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_page3(pdf, merged):
    """
    Page 3:
      Row 0: [Gly × Superfamily stacked]  [Gly × Family stacked]
      Row 1: [BioClim correlation bar + key — full width]
    """
    fig = plt.figure(figsize=(11, 11))
    outer = gridspec.GridSpec(2, 1, figure=fig,
                              hspace=0.30, # EDIT 0.40 to shift. was 0.58.
                              left=0.07, right=0.97,
                              top=0.94, bottom=0.06,
                              height_ratios=[1.2, 1.0])

    # Row 0: two glyphosate stacked-bar panels
    row0 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.42)

    gly_col = next((c for c in ["Glyphosate_R", "glyphosate_res"]
                    if c in merged.columns), None) or ""

    panel_gly_violin(fig.add_subplot(row0[0, 0]), merged, gly_col,
                     "te_fam_",
                     "TE Superfamily Insertions by Glyphosate Resistance", "A",
                     y_transform="log10p1")

    panel_gly_violin(fig.add_subplot(row0[0, 1]), merged, gly_col,
                     "te_clade_",
                     "LTR-RT Family Insertions by Glyphosate Resistance", "B",
                     y_transform="log10p1")
                                      
    # Row 1: BioClim bar + key
    row1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        wspace=0.06, width_ratios=[1.6, 1.0])

    panel_bioclim_correlation(fig.add_subplot(row1[0, 0]),
                              fig.add_subplot(row1[0, 1]),
                              merged, panel_letter="C")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_page4(pdf, merged):
    """
    Page 4: per-sample stacked bars — superfamily (left) and LTR-RT family (right)
    side-by-side on a single page.  No Y-axis labels; pink bands = resistant.
    """
    n = len(merged)
    fig_h = max(7, min(n * 0.030 + 1.5, 14))

    fig = plt.figure(figsize=(11, fig_h))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03,
                        wspace=0.06)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    panel_sample_stacked(ax1, merged, col_prefix="te_fam_", top_n=10,
                         panel_letter="A", panel_letter_y=1.015)
    panel_sample_stacked(ax2, merged, col_prefix="te_clade_", top_n=None,
                         panel_letter="B", panel_letter_y=1.015)


    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_sample_presence_matrix(pos_df, window=5):
    """
    Cluster all insertions (all samples combined) within ±window bp using a
    sweep-line merge, then return a binary sample×cluster presence/absence
    DataFrame (int8).  Each column is one genomic locus (cluster).
    """
    if pos_df.empty:
        return pd.DataFrame()

    samples = sorted(pos_df["sample_id"].unique())
    sample_idx = {s: i for i, s in enumerate(samples)}
    clusters = []          # list of frozensets of sample IDs

    for chrom, grp in pos_df.groupby("Chr"):
        rows = sorted(
            zip(grp["Start"].astype(int), grp["End"].astype(int),
                grp["sample_id"]),
            key=lambda x: x[0],
        )
        if not rows:
            continue

        cs, ce = rows[0][0] - window, rows[0][1] + window
        csids = {rows[0][2]}

        for start, end, sid in rows[1:]:
            es, ee = start - window, end + window
            if es <= ce:
                ce = max(ce, ee)
                csids.add(sid)
            else:
                clusters.append(frozenset(csids))
                cs, ce, csids = es, ee, {sid}
        clusters.append(frozenset(csids))

    if not clusters:
        return pd.DataFrame()

    mat = np.zeros((len(samples), len(clusters)), dtype=np.int8)
    for j, cl in enumerate(clusters):
        for s in cl:
            if s in sample_idx:
                mat[sample_idx[s], j] = 1

    return pd.DataFrame(mat, index=samples)


def build_page_sample_sharing(pdf, pos_df, gly_map, merged=None):
    """
    New page: sample TE insertion sharing.

    Left  panel → PCoA (metric MDS on Jaccard distances) coloured by Region;
                  marker edge colour encodes R/S phenotype.
    Right panel → pairwise Jaccard similarity heatmap hierarchically ordered
                  (UPGMA) with a top dendrogram whose branches are coloured by
                  Region (grey where multiple regions merge).
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    mat = build_sample_presence_matrix(pos_df)
    if mat.empty or mat.shape[0] < 3:
        return

    samples = list(mat.index)
    n = len(samples)

    # ── Jaccard distances ──────────────────────────────────────────────────────
    X = mat.values.astype(float)
    dist_vec = pdist(X, metric="jaccard")
    dist_sq  = squareform(dist_vec)
    np.fill_diagonal(dist_sq, 0.0)
    sim_sq = 1.0 - dist_sq

    # ── UPGMA hierarchical clustering ─────────────────────────────────────────
    Z     = linkage(dist_vec, method="average")
    order = leaves_list(Z)
    ordered_samples = [samples[i] for i in order]
    sim_ordered     = sim_sq[np.ix_(order, order)]

    # ── PCoA via metric MDS ───────────────────────────────────────────────────
    try:
        from sklearn.manifold import MDS
        try:
            mds = MDS(n_components=2, dissimilarity="precomputed",
                      random_state=42, normalized_stress="auto")
        except TypeError:
            mds = MDS(n_components=2, dissimilarity="precomputed",
                      random_state=42)
        coords = mds.fit_transform(dist_sq)
        stress = mds.stress_
        pcoa_ok = True
    except ImportError:
        pcoa_ok = False

    # ── Region map & palette ──────────────────────────────────────────────────
    region_map = {}
    if merged is not None:
        for col in ["Region_new", "Region", "region"]:
            if col in merged.columns:
                for sid in samples:
                    if sid in merged.index:
                        v = merged.loc[sid, col]
                        if pd.notna(v) and str(v).lower() not in ("nan", ""):
                            region_map[sid] = str(v)
                break

    unique_regions = sorted(set(region_map.values()))
    n_reg = len(unique_regions)
    base_cmap = plt.cm.get_cmap("tab10" if n_reg <= 10 else "tab20")
    region_palette = {
        r: mcolors.to_hex(base_cmap(i % base_cmap.N))
        for i, r in enumerate(unique_regions)
    }
    MIXED_COL = "#888888"

    def rcol(sid):
        """Region colour for a sample; grey if unknown."""
        return region_palette.get(region_map.get(sid), MIXED_COL)

    # ── Dendrogram link_color_func (colour by region) ─────────────────────────
    # cluster_region_set[k] = set of region labels for all leaves under node k.
    # Leaves are indices 0..n-1; internal nodes n..2n-2.
    # link_color_func(k) colours the link FROM node k TO its parent.
    cluster_region_set = {}
    for i, s in enumerate(samples):
        r = region_map.get(s)
        cluster_region_set[i] = {r} if r is not None else {None}
    for i, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        cluster_region_set[n + i] = (cluster_region_set[left] |
                                     cluster_region_set[right])

    def link_color_func(k):
        regs = cluster_region_set.get(k, {None}) - {None}
        if len(regs) == 1:
            return region_palette.get(next(iter(regs)), MIXED_COL)
        return MIXED_COL

    # ── Phenotype edge colours for PCoA ───────────────────────────────────────
    R_COL, S_COL = "#c0392b", "#2980b9"

    def pheno_edge(sid):
        v = gly_map.get(sid)
        return R_COL if v is True else S_COL if v is False else "white"

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))

    # Reserve bottom margin for the region legend (~2 rows of 17 entries)
    LEG_BOTTOM = 0.17
    if pcoa_ok:
        outer = gridspec.GridSpec(
            1, 2, figure=fig,
            left=0.06, right=0.97, top=0.97, bottom=LEG_BOTTOM,
            wspace=0.32, width_ratios=[2, 3],
        )
        ax_pcoa = fig.add_subplot(outer[0, 0])
        heat_ss = outer[0, 1]
    else:
        outer = gridspec.GridSpec(
            1, 1, figure=fig,
            left=0.10, right=0.92, top=0.97, bottom=LEG_BOTTOM,
        )
        heat_ss = outer[0, 0]

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=heat_ss,
        height_ratios=[0.18, 0.82],
        width_ratios=[0.91, 0.04],
        hspace=0.01, wspace=0.04,
    )
    ax_dend = fig.add_subplot(inner[0, 0])
    ax_heat = fig.add_subplot(inner[1, 0])
    ax_cbar = fig.add_subplot(inner[1, 1])

    # ── PCoA panel ────────────────────────────────────────────────────────────
    if pcoa_ok:
        pt_face   = [rcol(s) for s in samples]
        pt_edge   = [pheno_edge(s) for s in samples]
        ax_pcoa.scatter(
            coords[:, 0], coords[:, 1],
            c=pt_face, edgecolors=pt_edge, linewidths=1.2,
            s=60, zorder=3, alpha=0.92,
        )
        lbl_fs = max(4, min(7, 100 // n))
        for i, sid in enumerate(samples):
            ax_pcoa.annotate(
                str(sid), (coords[i, 0], coords[i, 1]),
                textcoords="offset points", xytext=(0, 4),
                fontsize=lbl_fs, ha="center", va="bottom", alpha=0.75,
            )
        ax_pcoa.axhline(0, color="#ddd", lw=0.5, zorder=1)
        ax_pcoa.axvline(0, color="#ddd", lw=0.5, zorder=1)
        ax_pcoa.set_xlabel("PCoA 1", fontsize=9)
        ax_pcoa.set_ylabel("PCoA 2", fontsize=9)
        ax_pcoa.set_title(
            f"PCoA  (Jaccard · metric MDS · stress={stress:.3f})",
            fontsize=8.5,
        )
        ax_pcoa.tick_params(labelsize=7)

    # ── Dendrogram – drawn manually so each leg gets its own colour ───────────
    # scipy's link_color_func(k) colours the *entire* U-shape (both legs AND
    # the bar) with the colour of the merged node k.  That means when a red
    # sample and a blue sample fork, the leg going down to the red sample also
    # turns grey.  Fix: compute x/y positions ourselves (same midpoint layout
    # scipy uses) then draw each left-leg, bar, and right-leg as separate
    # Line2D objects with independent colours.
    LEAF_SCALE = 10
    _leaf_order = list(leaves_list(Z))        # leaf display order (left → right)
    _cx: dict = {}                            # cluster → x centre
    _cy: dict = {}                            # cluster → y (merge height)
    for _di, _si in enumerate(_leaf_order):
        _cx[_si] = LEAF_SCALE * _di + LEAF_SCALE / 2
        _cy[_si] = 0.0

    def _lcol(k):
        """Colour for the link/leg leading UP from cluster k."""
        if not region_map:
            return MIXED_COL
        regs = cluster_region_set.get(k, {None}) - {None}
        if len(regs) == 1:
            return region_palette.get(next(iter(regs)), MIXED_COL)
        return MIXED_COL

    LW = 1.5
    for _i, _row in enumerate(Z):
        _left, _right = int(_row[0]), int(_row[1])
        _h   = float(_row[2])
        _node = n + _i
        _xl, _xr = _cx[_left], _cx[_right]
        _yl, _yr = _cy[_left], _cy[_right]
        _cx[_node] = (_xl + _xr) / 2
        _cy[_node] = _h
        # left leg: child → merge height  (coloured by the child's region)
        ax_dend.plot([_xl, _xl], [_yl, _h], color=_lcol(_left),  lw=LW,
                     solid_capstyle="butt")
        # horizontal bar: left → right at merge height (grey when regions mix)
        ax_dend.plot([_xl, _xr], [_h,  _h], color=_lcol(_node),  lw=LW,
                     solid_capstyle="butt")
        # right leg
        ax_dend.plot([_xr, _xr], [_yr, _h], color=_lcol(_right), lw=LW,
                     solid_capstyle="butt")

    ax_dend.set_xlim(0, n * LEAF_SCALE)
    ax_dend.set_ylim(0, float(Z[:, 2].max()) * 1.1)
    ax_dend.axis("off")

    # ── Heatmap (log-normalised) ───────────────────────────────────────────────
    from matplotlib.colors import LogNorm
    off_diag = sim_ordered[~np.eye(n, dtype=bool)]
    nonzero  = off_diag[off_diag > 0]
    vmin_log = float(nonzero.min()) if len(nonzero) else 1e-4
    vmin_log = max(vmin_log, 1e-4)          # floor to avoid extreme log range
    sim_display = np.clip(sim_ordered, vmin_log, 1.0)
    np.fill_diagonal(sim_display, 1.0)      # self-similarity stays at 1

    extent = [0, n * LEAF_SCALE, 0, n]
    im = ax_heat.imshow(
        sim_display, extent=extent, aspect="auto",
        cmap="YlOrRd", norm=LogNorm(vmin=vmin_log, vmax=1.0),
        interpolation="nearest", origin="upper",
    )
    ax_heat.set_xlim(0, n * LEAF_SCALE)
    ax_heat.set_ylim(0, n)

    tick_fs = max(4, min(7, 120 // n))
    x_ticks = [LEAF_SCALE * i + LEAF_SCALE / 2 for i in range(n)]
    y_ticks = [n - i - 0.5 for i in range(n)]
    ax_heat.set_xticks(x_ticks)
    ax_heat.set_xticklabels(ordered_samples, rotation=90, fontsize=tick_fs)
    ax_heat.set_yticks(y_ticks)
    ax_heat.set_yticklabels(ordered_samples, fontsize=tick_fs)

    # Colour tick labels by glyphosate status (R=red, S=blue, unknown=grey)
    R_COL, S_COL, U_COL = "#c0392b", "#2980b9", "#95a5a6"
    def gcol(sid):
        v = gly_map.get(sid)
        return R_COL if v is True else S_COL if v is False else U_COL

    fig.canvas.draw()
    for lbl in ax_heat.get_xticklabels():
        lbl.set_color(gcol(lbl.get_text()))
    for lbl in ax_heat.get_yticklabels():
        lbl.set_color(gcol(lbl.get_text()))

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Jaccard\nsimilarity\n(log scale)", fontsize=7,
                   rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize=6)

    # ── Region legend (always shown on dendrogram axes) ───────────────────────
    if region_map and unique_regions:
        leg_patches = [mpatches.Patch(color=region_palette[r], label=r)
                       for r in unique_regions]
        if pcoa_ok:
            # R/S edge indicators only meaningful when PCoA dots are visible
            R_COL, S_COL = "#c0392b", "#2980b9"
            if any(gly_map.get(s) is True  for s in samples):
                leg_patches.append(Line2D([0], [0], marker="o", color="w",
                                          markerfacecolor="w",
                                          markeredgecolor=R_COL,
                                          markeredgewidth=1.5, markersize=6,
                                          label="Resistant (dot edge)"))
            if any(gly_map.get(s) is False for s in samples):
                leg_patches.append(Line2D([0], [0], marker="o", color="w",
                                          markerfacecolor="w",
                                          markeredgecolor=S_COL,
                                          markeredgewidth=1.5, markersize=6,
                                          label="Susceptible (dot edge)"))
        # ncol: aim for ~2 rows
        ncols = max(1, -(-len(leg_patches) // 2))   # ceiling division → 2 rows
        fig.legend(
            handles=leg_patches,
            fontsize=6.5, loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncols,
            framealpha=0.85,
            title="Region",
            title_fontsize=7,
        )

    # ── Subtitle ──────────────────────────────────────────────────────────────
    n_loci = mat.shape[1]
    pcoa_note = "  ·  PCoA dot edge = R (red) / S (blue)" if pcoa_ok else ""
    fig.text(
        0.5, LEG_BOTTOM - 0.01,
        f"{n} samples  ·  {n_loci:,} insertion loci (±5 bp merge window)"
        f"  ·  heatmap log-normalised"
        f"  ·  sample label colour = Resistant (red) / Susceptible (blue){pcoa_note}",
        ha="center", va="top", fontsize=6.5, color="#555",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_page_insertion_spectrum(pdf, pos_df):
    """
    TE insertion cluster frequency spectrum — 3 panels.
    A (top, full-width): overall cluster size histogram, log y.
    B (bottom-left):     per-superfamily singleton enrichment vs. global baseline,
                         Fisher's exact + BH FDR, visualised as a lollipop plot.
    C (bottom-right):    same for LTR-RT families (clades).
    """
    from collections import Counter, defaultdict
    import matplotlib.ticker as mticker
    import matplotlib.gridspec as mgridspec
    import math as _math
    from matplotlib.lines import Line2D

    if pos_df.empty:
        return

    has_te_id = "TE_ID" in pos_df.columns
    window    = 5

    # ── Sweep-line clustering with TE label capture ──────────────────────────
    def _register(te_id, sup_cnt, fam_cnt):
        for order, sup, clade in parse_te_levels(te_id):
            sup_cnt[sup] += 1
            if order == "LTR":
                fam_cnt[clade] += 1

    all_clusters = []   # list of (n_samples, dom_superfamily_or_None, dom_ltr_family_or_None)

    for chrom, grp in pos_df.groupby("Chr"):
        te_ids = grp["TE_ID"].tolist() if has_te_id else [""] * len(grp)
        rows   = sorted(
            zip(grp["Start"].astype(int), grp["End"].astype(int),
                grp["sample_id"], te_ids),
            key=lambda x: x[0],
        )
        if not rows:
            continue

        cs, ce   = rows[0][0] - window, rows[0][1] + window
        csids    = {rows[0][2]}
        csup_cnt = Counter()
        cfam_cnt = Counter()
        _register(rows[0][3], csup_cnt, cfam_cnt)

        def _flush(sids, sup_cnt, fam_cnt):
            dom_sup = sup_cnt.most_common(1)[0][0] if sup_cnt else None
            dom_fam = fam_cnt.most_common(1)[0][0] if fam_cnt else None
            all_clusters.append((len(sids), dom_sup, dom_fam))

        for start, end, sid, te_id in rows[1:]:
            es, ee = start - window, end + window
            if es <= ce:
                ce = max(ce, ee)
                csids.add(sid)
                _register(te_id, csup_cnt, cfam_cnt)
            else:
                _flush(csids, csup_cnt, cfam_cnt)
                cs, ce, csids = es, ee, {sid}
                csup_cnt      = Counter()
                cfam_cnt      = Counter()
                _register(te_id, csup_cnt, cfam_cnt)
        _flush(csids, csup_cnt, cfam_cnt)

    if not all_clusters:
        return

    # ── Global spectrum ──────────────────────────────────────────────────────
    sup_sizes     = defaultdict(list)
    fam_sizes     = defaultdict(list)
    overall_counts = [n for n, _, _ in all_clusters]

    for n, dom_sup, dom_fam in all_clusters:
        if dom_sup:
            sup_sizes[dom_sup].append(n)
        if dom_fam:
            fam_sizes[dom_fam].append(n)

    overall_spec  = Counter(overall_counts)
    max_count     = max(overall_spec.keys())
    xs_all        = list(range(1, max_count + 1))
    ys_all        = [overall_spec.get(k, 0) for k in xs_all]

    n_total       = len(overall_counts)
    n_sing        = overall_spec.get(1, 0)
    n_shared      = n_total - n_sing
    p_global_sing = n_sing / n_total        # global singleton rate

    # ── Statistical testing (mirrors existing find_enriched_clusters BH) ─────
    MIN_CLUSTERS = 20    # minimum per group to be included
    TOP_N        = 50

    def _bh_correct(pvals_arr):
        """BH step-down FDR — identical approach to find_enriched_clusters."""
        n          = len(pvals_arr)
        sorted_idx = np.argsort(pvals_arr)
        ranks      = np.empty(n, dtype=float)
        ranks[sorted_idx] = np.arange(1, n + 1)
        raw_q      = np.clip(pvals_arr * n / ranks, 0, 1)
        # monotonicity: scan from largest p down, take running minimum
        fdr        = (pd.Series(raw_q).iloc[sorted_idx[::-1]]
                      .cummin()
                      .reindex(pd.RangeIndex(n))
                      .values)
        return fdr

    def _build_stats_df(group_sizes):
        """
        For each group with >= MIN_CLUSTERS clusters:
          - 2×2 Fisher's exact: (group_singletons | group_shared)
                             vs (rest_singletons  | rest_shared)
          - log2FC = log2(group_singleton_rate / global_singleton_rate)
          - BH FDR across all tested groups
        Returns a DataFrame sorted by log2FC.
        """
        top_groups = sorted(
            [k for k in group_sizes if len(group_sizes[k]) >= MIN_CLUSTERS],
            key=lambda k: -len(group_sizes[k]),
        )[:TOP_N]

        rows = []
        for gname in top_groups:
            sz        = group_sizes[gname]
            n_g       = len(sz)
            n_g_sing  = sum(1 for x in sz if x == 1)
            n_g_share = n_g - n_g_sing
            rest_sing  = n_sing  - n_g_sing
            rest_share = n_shared - n_g_share
            try:
                _, pval = stats.fisher_exact(
                    [[n_g_sing,  n_g_share],
                     [max(0, rest_sing), max(0, rest_share)]],
                    alternative="two-sided",
                )
            except Exception:
                pval = 1.0
            p_obs  = n_g_sing / n_g
            eps    = 0.5 / n_total      # pseudocount matching existing script style
            log2fc = np.log2((p_obs + eps) / (p_global_sing + eps))
            label  = gname.split("/")[-1] if "/" in gname else gname
            rows.append({"name": gname, "label": label,
                         "n": n_g, "log2fc": log2fc, "pval": pval})

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        fdr = _bh_correct(df["pval"].values.astype(float))
        df["fdr"]   = fdr
        df["ratio"] = np.exp2(df["log2fc"])   # observed / expected, centred at 1.0
        return df.sort_values("ratio").reset_index(drop=True)

    # ── Tick helpers ─────────────────────────────────────────────────────────
    def _setup_log_yticks(ax, ymax):
        pow10 = [10**i for i in range(0, _math.ceil(_math.log10(max(ymax, 2))) + 1)]
        ax.set_yticks(pow10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    def _setup_xticks_bar(ax, xmax):
        locator = mticker.MaxNLocator(integer=True, nbins=10, prune="both")
        locator.set_params(min_n_ticks=5)
        ax.xaxis.set_major_locator(locator)
        fig.canvas.draw()
        ticks = [t for t in ax.get_xticks() if 1 <= t <= xmax]
        if ticks and ticks[0] != 1:
            ticks = [1] + ticks
        ax.set_xticks(ticks)

    # ── Lollipop drawing ─────────────────────────────────────────────────────
    COL_MORE = "#d62728"   # red  = more singletons than expected (recent/private)
    COL_LESS = "#1f77b4"   # blue = fewer singletons than expected (old/widespread)
    SIG_FDR  = 0.05

    def _draw_lollipop(ax, df, panel_letter):
        ax.text(-0.14, 1.06, panel_letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        if df.empty:
            ax.text(0.5, 0.5,
                    f"No groups with ≥ {MIN_CLUSTERS} clusters",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888", fontsize=8)
            return

        for i, row in df.iterrows():
            r     = row["ratio"]
            sig   = row["fdr"] < SIG_FDR
            color = COL_MORE if r > 1 else COL_LESS
            ax.plot([1.0, r], [i, i],
                    color=color, lw=1.5, solid_capstyle="round", zorder=1)
            ax.plot(r, i, "o",
                    color=color, ms=7 if sig else 5,
                    mfc=color if sig else "white", mew=1.5, zorder=2)

        ax.axvline(1.0, color="#555", lw=0.8, ls="--", zorder=0)
        ax.set_ylim(-0.6, len(df) - 0.4)
        ax.set_yticks(range(len(df)))
        ylabels = [
            f"{row['label']}  (n={int(row['n']):,})"
            for _, row in df.iterrows()
        ]
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlabel("Singleton rate  (observed / expected)", fontsize=8)

        xabs = (df["ratio"] - 1.0).abs().max()
        ax.set_xlim(1 - xabs * 1.3 - 0.02, 1 + xabs * 1.3 + 0.02)

        # Inline legend
        leg_handles = [
            Line2D([0], [0], marker="o", color=COL_MORE, ms=7,
                   mfc=COL_MORE, mew=1.5, ls="none", label="more singletons"),
            Line2D([0], [0], marker="o", color=COL_LESS, ms=7,
                   mfc=COL_LESS, mew=1.5, ls="none", label="fewer singletons"),
            Line2D([0], [0], marker="o", color="#888", ms=7,
                   mfc="#888", mew=1.5, ls="none", label=f"FDR < {SIG_FDR}"),
            Line2D([0], [0], marker="o", color="#888", ms=5,
                   mfc="white", mew=1.5, ls="none", label="n.s."),
        ]
        ax.legend(handles=leg_handles, fontsize=6, framealpha=0.85,
                  loc="lower right", borderpad=0.5, labelspacing=0.3)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 9))
    gs  = mgridspec.GridSpec(2, 2, figure=fig,
                             left=0.14, right=0.97, top=0.94, bottom=0.07,
                             hspace=0.28, wspace=0.38,
                             height_ratios=[1, 1.5])
    ax_A = fig.add_subplot(gs[0, :])
    ax_B = fig.add_subplot(gs[1, 0])
    ax_C = fig.add_subplot(gs[1, 1])

    # ── Panel A ───────────────────────────────────────────────────────────────
    ax_A.bar(xs_all, ys_all, color=PALETTE[0], edgecolor="none", width=0.85)
    ax_A.set_yscale("log")
    _setup_log_yticks(ax_A, max(ys_all))
    ax_A.set_xlim(0.3, max_count + 0.7)
    ax_A.set_xlabel("Number of samples in TE insertion cluster")
    ax_A.set_ylabel("Clusters (log scale)")
    _setup_xticks_bar(ax_A, max_count)
    ax_A.text(-0.08, 1.06, "A", transform=ax_A.transAxes,
              fontsize=12, fontweight="bold", va="top")
    fig.text(
        0.5, 0.965,
        f"Total clusters: {n_total:,}  ·  "
        f"Singletons: {n_sing:,} ({100*n_sing/n_total:.1f}%)  ·  "
        f"Shared (≥2 samples): {n_shared:,} ({100*n_shared/n_total:.1f}%)",
        ha="center", va="top", fontsize=7.5, color="#555",
    )

    # ── Panels B & C ──────────────────────────────────────────────────────────
    if has_te_id:
        df_sup = _build_stats_df(sup_sizes)
        df_fam = _build_stats_df(fam_sizes)
    else:
        df_sup = pd.DataFrame()
        df_fam = pd.DataFrame()

    _draw_lollipop(ax_B, df_sup, "B")
    _draw_lollipop(ax_C, df_fam, "C")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_page_kary_sharing(pdf, clusters_df, pos_df,
                            fai_lengths, crm_intervals,
                            gly_map, merged=None):
    """
    Combined page: Panel A (top) = karyotype enrichment,
                   Panel B (bottom) = UPGMA dendrogram + Jaccard heatmap.
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, leaves_list
    from matplotlib.colors import LogNorm
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    # ── Sample sharing data ───────────────────────────────────────────────────
    mat = build_sample_presence_matrix(pos_df)
    has_sharing = not mat.empty and mat.shape[0] >= 3

    LEG_BOTTOM = 0.11
    fig = plt.figure(figsize=(11, 16))

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        left=0.06, right=0.95, top=0.97, bottom=LEG_BOTTOM,
        hspace=0.10, height_ratios=[0.40, 0.60],
    )

    # ── Panel A: karyotype ────────────────────────────────────────────────────
    ax_kary = fig.add_subplot(outer[0, 0])
    panel_karyotype(ax_kary, clusters_df, pos_df,
                    fai_lengths=fai_lengths, crm_intervals=crm_intervals)
    ax_kary.text(-0.005, 1.02, "A", transform=ax_kary.transAxes,
                 fontsize=13, fontweight="bold", va="bottom")

    # ── Panel B: dendrogram + heatmap ─────────────────────────────────────────
    if not has_sharing:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    samples = list(mat.index)
    n = len(samples)

    X        = mat.values.astype(float)
    dist_vec = pdist(X, metric="jaccard")
    dist_sq  = squareform(dist_vec)
    np.fill_diagonal(dist_sq, 0.0)
    sim_sq   = 1.0 - dist_sq

    Z               = linkage(dist_vec, method="average")
    order           = leaves_list(Z)
    ordered_samples = [samples[i] for i in order]
    sim_ordered     = sim_sq[np.ix_(order, order)]

    # Region palette
    region_map = {}
    if merged is not None:
        for col in ["Region_new", "Region", "region"]:
            if col in merged.columns:
                for sid in samples:
                    if sid in merged.index:
                        v = merged.loc[sid, col]
                        if pd.notna(v) and str(v).lower() not in ("nan", ""):
                            region_map[sid] = str(v)
                break

    unique_regions = sorted(set(region_map.values()))
    n_reg = len(unique_regions)
    base_cmap = plt.cm.get_cmap("tab10" if n_reg <= 10 else "tab20")
    region_palette = {
        r: mcolors.to_hex(base_cmap(i % base_cmap.N))
        for i, r in enumerate(unique_regions)
    }
    MIXED_COL = "#888888"

    def rcol(sid):
        return region_palette.get(region_map.get(sid), MIXED_COL)

    # cluster_region_set for branch coloring
    cluster_region_set = {}
    for i, s in enumerate(samples):
        r = region_map.get(s)
        cluster_region_set[i] = {r} if r is not None else {None}
    for i, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        cluster_region_set[n + i] = (cluster_region_set[left] |
                                     cluster_region_set[right])

    def _lcol(k):
        if not region_map:
            return MIXED_COL
        regs = cluster_region_set.get(k, {None}) - {None}
        if len(regs) == 1:
            return region_palette.get(next(iter(regs)), MIXED_COL)
        return MIXED_COL

    # Glyphosate colours for tick labels
    R_COL, S_COL, U_COL = "#c0392b", "#2980b9", "#95a5a6"
    def gcol(sid):
        v = gly_map.get(sid)
        return R_COL if v is True else S_COL if v is False else U_COL

    # Sub-grid for panel B
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[1, 0],
        height_ratios=[0.15, 0.85], width_ratios=[0.92, 0.04],
        hspace=0.01, wspace=0.04,
    )
    ax_dend = fig.add_subplot(inner[0, 0])
    ax_heat = fig.add_subplot(inner[1, 0])
    ax_cbar = fig.add_subplot(inner[1, 1])

    ax_dend.text(-0.005, 1.10, "B", transform=ax_dend.transAxes,
                 fontsize=13, fontweight="bold", va="bottom")

    # Dendrogram (manual per-leg coloring)
    LEAF_SCALE = 10
    _leaf_order = list(leaves_list(Z))
    _cx: dict = {}
    _cy: dict = {}
    for _di, _si in enumerate(_leaf_order):
        _cx[_si] = LEAF_SCALE * _di + LEAF_SCALE / 2
        _cy[_si] = 0.0

    LW = 1.5
    for _i, _row in enumerate(Z):
        _left, _right = int(_row[0]), int(_row[1])
        _h    = float(_row[2])
        _node = n + _i
        _xl, _xr = _cx[_left], _cx[_right]
        _yl, _yr = _cy[_left], _cy[_right]
        _cx[_node] = (_xl + _xr) / 2
        _cy[_node] = _h
        ax_dend.plot([_xl, _xl], [_yl, _h], color=_lcol(_left),  lw=LW,
                     solid_capstyle="butt")
        ax_dend.plot([_xl, _xr], [_h,  _h], color=_lcol(_node),  lw=LW,
                     solid_capstyle="butt")
        ax_dend.plot([_xr, _xr], [_yr, _h], color=_lcol(_right), lw=LW,
                     solid_capstyle="butt")

    ax_dend.set_xlim(0, n * LEAF_SCALE)
    ax_dend.set_ylim(0, float(Z[:, 2].max()) * 1.1)
    ax_dend.axis("off")

    # Heatmap (log-normalised)
    off_diag = sim_ordered[~np.eye(n, dtype=bool)]
    nonzero  = off_diag[off_diag > 0]
    vmin_log = float(nonzero.min()) if len(nonzero) else 1e-4
    vmin_log = max(vmin_log, 1e-4)
    sim_display = np.clip(sim_ordered, vmin_log, 1.0)
    np.fill_diagonal(sim_display, 1.0)

    extent = [0, n * LEAF_SCALE, 0, n]
    im = ax_heat.imshow(
        sim_display, extent=extent, aspect="auto",
        cmap="YlOrRd", norm=LogNorm(vmin=vmin_log, vmax=1.0),
        interpolation="nearest", origin="upper",
    )
    ax_heat.set_xlim(0, n * LEAF_SCALE)
    ax_heat.set_ylim(0, n)

    tick_fs = max(4, min(7, 120 // n))
    x_ticks = [LEAF_SCALE * i + LEAF_SCALE / 2 for i in range(n)]
    y_ticks = [n - i - 0.5 for i in range(n)]
    ax_heat.set_xticks(x_ticks)
    ax_heat.set_xticklabels(ordered_samples, rotation=90, fontsize=tick_fs)
    ax_heat.set_yticks(y_ticks)
    ax_heat.set_yticklabels(ordered_samples, fontsize=tick_fs)

    fig.canvas.draw()
    for lbl in ax_heat.get_xticklabels():
        lbl.set_color(gcol(lbl.get_text()))
    for lbl in ax_heat.get_yticklabels():
        lbl.set_color(gcol(lbl.get_text()))

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Jaccard\nsimilarity\n(log scale)", fontsize=7,
                   rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize=6)

    # Region legend at bottom
    if region_map and unique_regions:
        leg_patches = [mpatches.Patch(color=region_palette[r], label=r)
                       for r in unique_regions]
        ncols = max(1, -(-len(leg_patches) // 2))
        fig.legend(
            handles=leg_patches,
            fontsize=6.5, loc="lower center",
            bbox_to_anchor=(0.5, 0.045),
            ncol=ncols, framealpha=0.85,
            title="Region", title_fontsize=7,
        )

    n_loci = mat.shape[1]
    fig.text(
        0.5, LEG_BOTTOM - 0.005,
        f"{n} samples  ·  {n_loci:,} insertion loci (±5 bp merge window)"
        f"  ·  heatmap log-normalised"
        f"  ·  branch/label colour = Region"
        f"  ·  sample label colour = Resistant (red) / Susceptible (blue)",
        ha="center", va="top", fontsize=6.5, color="#555",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_page6(pdf, clusters_df, pos_df, fai_lengths=None, crm_intervals=None):
    """Page 6: linear karyotype with R/S-enriched insertion positions."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.06)
    ax = fig.add_subplot(111)
    panel_karyotype(ax, clusters_df, pos_df, fai_lengths=fai_lengths,
                    crm_intervals=crm_intervals)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def compute_intergenic_log2fc(clade_comp_df, top_clades, ctx_lengths_kb=None):
    """Return a Series {family: intergenic log2(obs/exp)}.

    Replicates the exact normalisation used by panel_context_enrichment so the
    values match those shown in the heatmap (length-normalised, clamped ±3).
    """
    CONTEXTS = [c for c in CONTEXT_ORDER if c in clade_comp_df.index]
    if not CONTEXTS:
        return pd.Series(dtype=float)

    df = clade_comp_df.loc[CONTEXTS].copy()
    top_cols = [c for c in top_clades if c in df.columns]
    if not top_cols:
        return pd.Series(dtype=float)
    df = df[top_cols].astype(float)

    if ctx_lengths_kb:
        for ctx in list(df.index):
            kb = ctx_lengths_kb.get(ctx, None)
            if kb and kb > 0:
                df.loc[ctx] = df.loc[ctx] / kb

    genome_total = df.sum(axis=1)
    total = genome_total.sum()
    if total == 0:
        return pd.Series(dtype=float)

    expected = (genome_total / total).clip(lower=1e-10)
    cat_totals = df.sum(axis=0).replace(0, 1)
    observed = df.div(cat_totals, axis=1)
    log2fc = np.log2(observed.div(expected, axis=0).clip(lower=1e-6)).clip(-3, 3)

    if "Intergenic" not in log2fc.index:
        return pd.Series(dtype=float)
    return log2fc.loc["Intergenic"]


def panel_intergenic_regression(ax, clade_comp_df, merged, top_clades,
                                ctx_lengths_kb=None, panel_letter="A"):
    """Scatter + OLS: intergenic log2(obs/exp) vs mean insertions per sample.

    One data point per LTR-RT family (LTR-RT families only, matching heatmap).
    Aesthetics mirror the page-2 regressions.  Stats annotation is placed
    inside the axes (bottom-centre) to avoid overlap with the x-axis label.
    """
    intergenic_fc = compute_intergenic_log2fc(clade_comp_df, top_clades,
                                              ctx_lengths_kb)
    if intergenic_fc.empty:
        ax.text(0.5, 0.5, "No intergenic data", ha="center",
                transform=ax.transAxes)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
        return

    clade_map = {c.replace("te_clade_", ""): c
                 for c in merged.columns if c.startswith("te_clade_")}

    families, x_vals, y_vals = [], [], []
    for fam in intergenic_fc.index:
        if fam in clade_map:
            families.append(fam)
            x_vals.append(float(merged[clade_map[fam]].mean()))
            y_vals.append(float(intergenic_fc[fam]))

    if len(families) < 3:
        ax.text(0.5, 0.5, "Insufficient family data", ha="center",
                transform=ax.transAxes)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
        return

    xc = np.array(x_vals, dtype=float)
    yc = np.array(y_vals, dtype=float)

    ax.scatter(xc, yc, s=16, alpha=0.55, color=PALETTE[0],
               edgecolors="none", zorder=3)

    r2 = np.nan
    try:
        m, b = np.polyfit(xc, yc, 1)
        xl = np.linspace(xc.min(), xc.max(), 200)
        ax.plot(xl, m * xl + b, color="#CC3333", lw=1.6,
                linestyle="--", zorder=4, alpha=0.85)
        yhat   = m * xc + b
        ss_res = ((yc - yhat) ** 2).sum()
        ss_tot = ((yc - yc.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    except Exception:
        pass

    rho, pval = safe_corr(pd.Series(xc), pd.Series(yc), "spearman")
    parts = []
    if not np.isnan(r2):   parts.append(f"R² = {r2:.3f}")
    if not np.isnan(rho):  parts.append(f"ρ = {rho:.3f}")
    if not np.isnan(pval): parts.append(f"p = {pval:.2e}")
    parts.append(f"n = {len(xc)}")
    annot = ",  ".join(parts)

    # Place stats *inside* the axes (top-left) to avoid overlapping the xlabel
    ax.text(0.04, 0.97, annot, transform=ax.transAxes,
            fontsize=7, color="#333", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#bbbbbb", alpha=0.93, lw=0.6))

    ax.set_xlabel("Mean Insertions per Sample")
    ax.set_ylabel("Intergenic log₂(obs/exp)")
    ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right",
            clip_on=False)

    # Label each family point
    for xi, yi, fam in zip(x_vals, y_vals, families):
        ax.annotate(fam, (xi, yi), fontsize=5.5, color="#333333",
                    xytext=(3, 3), textcoords="offset points")


def panel_ref_count_regression(ax, merged, ltr_age, top_clades, panel_letter=None):
    """Scatter + OLS: Mean Insertions per Sample vs LTR-RT count in the reference.

    Reference count per family = number of elements in the LTR age file for
    that family (len(ltr_age[family])).  One point per LTR-RT family.
    Stats annotation placed inside the axes (top-left) to avoid label overlap.
    """
    if not ltr_age:
        ax.text(0.5, 0.5, "No LTR age data", ha="center",
                transform=ax.transAxes)
        if panel_letter:
            ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom", ha="right",
                    clip_on=False)
        return

    clade_map = {c.replace("te_clade_", ""): c
                 for c in merged.columns if c.startswith("te_clade_")}

    families, x_vals, y_vals = [], [], []
    for fam in top_clades:
        if fam in clade_map and fam in ltr_age:
            families.append(fam)
            x_vals.append(float(merged[clade_map[fam]].mean()))
            y_vals.append(float(len(ltr_age[fam])))

    if len(families) < 3:
        ax.text(0.5, 0.5, "Insufficient family data", ha="center",
                transform=ax.transAxes)
        if panel_letter:
            ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom", ha="right",
                    clip_on=False)
        return

    xc = np.array(x_vals, dtype=float)
    yc = np.array(y_vals, dtype=float)

    ax.scatter(xc, yc, s=16, alpha=0.55, color=PALETTE[0],
               edgecolors="none", zorder=3)

    r2 = np.nan
    try:
        m, b = np.polyfit(xc, yc, 1)
        xl = np.linspace(xc.min(), xc.max(), 200)
        ax.plot(xl, m * xl + b, color="#CC3333", lw=1.6,
                linestyle="--", zorder=4, alpha=0.85)
        yhat   = m * xc + b
        ss_res = ((yc - yhat) ** 2).sum()
        ss_tot = ((yc - yc.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    except Exception:
        pass

    rho, pval = safe_corr(pd.Series(xc), pd.Series(yc), "spearman")
    parts = []
    if not np.isnan(r2):   parts.append(f"R² = {r2:.3f}")
    if not np.isnan(rho):  parts.append(f"ρ = {rho:.3f}")
    if not np.isnan(pval): parts.append(f"p = {pval:.2e}")
    parts.append(f"n = {len(xc)}")
    annot = ",  ".join(parts)

    ax.text(0.04, 0.97, annot, transform=ax.transAxes,
            fontsize=7, color="#333", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#bbbbbb", alpha=0.93, lw=0.6))

    ax.set_xlabel("Mean Insertions per Sample")
    ax.set_ylabel("Reference LTR-RT Count")

    if panel_letter:
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)

    for xi, yi, fam in zip(x_vals, y_vals, families):
        ax.annotate(fam, (xi, yi), fontsize=5.5, color="#333333",
                    xytext=(3, 3), textcoords="offset points")


def panel_metagene_single(ax, metagene_df, panel_letter="D"):
    """Single collapsed metagene bar plot — all samples pooled across regions."""
    if metagene_df.empty:
        ax.text(0.5, 0.5, "No metagene data", ha="center",
                transform=ax.transAxes, fontsize=8)
        ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="right",
                clip_on=False)
        return

    x_abs = int(max(abs(metagene_df["bin_center"].min()),
                    abs(metagene_df["bin_center"].max())) + 25)
    bin_size = 50

    grp = metagene_df.groupby("bin_center")["freq"].agg(["mean", "std", "count"])
    grp["se"] = grp["std"] / np.sqrt(grp["count"].clip(lower=1))
    bc = grp.index.values
    mn = grp["mean"].values
    se = grp["se"].values

    Y_MAX    = 0.0009
    Y_TICKS  = [0.0000, 0.0004, 0.0008]
    Y_LABELS = ["0.0000", "0.0004", "0.0008"]

    ax.bar(bc, mn, width=bin_size - 1, color=PALETTE[0], alpha=0.75,
           edgecolor="none", align="center")
    ax.errorbar(bc, mn, yerr=se, fmt="none", ecolor="#333333",
                elinewidth=0.6, capsize=1.5, capthick=0.6)
    ax.axvline(0, color="#555555", lw=0.9, linestyle="--", zorder=5)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlim(-x_abs, x_abs)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels(Y_LABELS)
    ax.set_ylabel("Frequency of TE insertion\nin bin (50 bp)", fontsize=7, labelpad=4)
    ax.set_xlabel("Distance from Gene Boundary (bp)")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", alpha=0.20)
    ax.grid(axis="x", alpha=0)

    # Upstream vs downstream significance test (Mann-Whitney U, per sample)
    try:
        up_sums, dn_sums = [], []
        for sid, sgrp in metagene_df.groupby("sample_id"):
            up_sums.append(sgrp.loc[sgrp["bin_center"] < 0, "freq"].sum())
            dn_sums.append(sgrp.loc[sgrp["bin_center"] > 0, "freq"].sum())
        up_arr, dn_arr = np.array(up_sums), np.array(dn_sums)
        if len(up_arr) >= 3 and len(dn_arr) >= 3:
            _, pval = stats.mannwhitneyu(up_arr, dn_arr, alternative="two-sided")
            direction = ("Up > Dn" if np.median(up_arr) > np.median(dn_arr)
                         else "Dn > Up")
            sig = ("***" if pval < 0.001 else "**" if pval < 0.01
                   else "*" if pval < 0.05 else "ns")
            ax.text(0.98, 0.92, f"{direction}  p={pval:.3g} {sig}",
                    transform=ax.transAxes, fontsize=5.5,
                    ha="right", va="top", color="#555555", fontstyle="italic")
    except Exception:
        pass

    ax.text(-0.06, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right",
            clip_on=False)


def panel_ltr_age_density(axes, ltr_age_data, panel_letter="E"):
    """Stacked K2P density histograms, one strip per LTR-RT family.

    Family order follows LTR_AGE_FAMILY_ORDER (matching Panel C top-to-bottom).
    Only families present in ltr_age_data are plotted; families in the fixed
    order that are absent from the data are silently skipped.
    Bins: 0–0.15 in steps of 0.003.  Values above 0.15 are excluded.
    The distribution peak is identified via KDE (avoids local histogram noise)
    and marked with a dotted vertical line + label.
    """
    from scipy.stats import gaussian_kde

    K2P_MAX  = 0.15
    BIN_SIZE = 0.003
    bins = np.arange(0, K2P_MAX + BIN_SIZE, BIN_SIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Use fixed Panel-C order; skip families absent from the data
    families = [(fam, ltr_age_data[fam])
                for fam in LTR_AGE_FAMILY_ORDER
                if fam in ltr_age_data]

    if not families or not axes:
        for ax in axes:
            ax.text(0.5, 0.5, "No LTR age data", ha="center",
                    transform=ax.transAxes, fontsize=8)
        if axes:
            axes[0].text(-0.06, 1.04, panel_letter, transform=axes[0].transAxes,
                         fontsize=11, fontweight="bold", va="bottom", ha="right",
                         clip_on=False)
        return

    for fi, (ax, (family, values)) in enumerate(zip(axes, families)):
        vals = np.array(values, dtype=float)
        vals_clipped = vals[vals <= K2P_MAX]
        counts, _ = np.histogram(vals_clipped, bins=bins)
        density = counts / max(len(values), 1)

        color = CLADE_PALETTE[fi % len(CLADE_PALETTE)]
        ax.bar(bin_centers, density, width=BIN_SIZE * 0.85,
               color=color, alpha=0.78, edgecolor="none")

        # KDE peak — evaluate on a fine grid to find the smooth mode
        peak_x = np.nan
        if len(vals_clipped) >= 5:
            try:
                kde_fn   = gaussian_kde(vals_clipped, bw_method="scott")
                eval_pts = np.linspace(0, K2P_MAX, 1000)
                kde_vals = kde_fn(eval_pts)
                peak_x   = eval_pts[np.argmax(kde_vals)]
            except Exception:
                pass

        if not np.isnan(peak_x):
            ax.axvline(peak_x, color=color, lw=0.9, linestyle=":",
                       alpha=0.95, zorder=5)
            ax.text(peak_x + K2P_MAX * 0.015,
                    ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0
                    else density.max() * 0.92,
                    f"{peak_x:.3f}",
                    fontsize=5.5, color="#222222", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.75))

        ax.text(0.98, 0.97, f"{family}  (n={len(values):,})",
                transform=ax.transAxes, fontsize=6,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.80))

        ax.set_xlim(0, K2P_MAX)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.grid(axis="y", alpha=0.20)
        ax.grid(axis="x", alpha=0)

        if fi < len(families) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("K2P Divergence", fontsize=7)
            ax.tick_params(axis="x", labelsize=7)

    if axes:
        mid = len(axes) // 2
        axes[min(mid, len(axes) - 1)].set_ylabel(
            "Proportion", fontsize=7, labelpad=4)
        axes[0].text(-0.06, 1.04, panel_letter, transform=axes[0].transAxes,
                     fontsize=11, fontweight="bold", va="bottom", ha="right",
                     clip_on=False)


def build_page_gene_context(pdf, merged, pos_df, gff_path, fai_lengths=None,
                            ltr_age=None):
    """Page 6: panels A (intergenic regression), B (LTR-RT superfamily context),
    C (LTR-RT family context), D (collapsed metagene), E (K2P age distributions)."""
    print("    Loading GFF for gene-context page …")
    genes_df, exon_intervals = load_gff_full(gff_path)
    if genes_df.empty:
        print("    WARNING: no genes loaded — skipping gene-context page")
        return
    n_genes = len(genes_df)
    print(f"    {n_genes:,} genes, {len(exon_intervals):,} with exon data")

    print("    Computing gene disruptions …")
    disruption_df = compute_gene_disruptions(pos_df, genes_df, exon_intervals)

    print("    Computing metagene profile …")
    metagene_df = compute_metagene_profile(
        pos_df, genes_df, window=GENE_CONTEXT_WINDOW_BP
    )


    print("    Computing TE context composition …")
    fam_comp_df, clade_comp_df, fam_long, clade_long = \
        compute_context_te_composition(
            pos_df, genes_df, exon_intervals,
            window=GENE_CONTEXT_WINDOW_BP
        )

    print("    Computing context region lengths for density normalisation …")
    ctx_lengths_kb = compute_context_lengths(genes_df, exon_intervals,
                                             window=GENE_CONTEXT_WINDOW_BP,
                                             fai_lengths=fai_lengths)
    print("    Context lengths (kb): " +
          ", ".join(f"{k}: {v:,.1f}" for k, v in ctx_lengths_kb.items()))

    # ── Determine age families (needed for layout) ───────────────────────
    # Count families that appear in both the fixed Panel-C order and the data.
    n_age = (sum(1 for fam in LTR_AGE_FAMILY_ORDER if fam in (ltr_age or {}))
             if ltr_age else 0)

    # ── Figure layout ────────────────────────────────────────────────────
    # Right column height: 1 metagene strip + n_age K2P strips.
    # Use enough height so strips are readable (~0.55 in each).
    fig_h = max(11.0, (1 + n_age) * 0.55 + 5.0)
    fig = plt.figure(figsize=(11, fig_h))
    outer = gridspec.GridSpec(1, 2, figure=fig,
                              wspace=0.45,
                              left=0.09, right=0.97,
                              top=0.97, bottom=0.04)

    # Left column: 3 logical panels (A, B, C) with moderate spacing.
    # A is split into two stacked regressions; B and C each split into a
    # tight violin+heatmap pair internally.
    left_gs = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[0, 0],
        hspace=0.30,
        height_ratios=[1.70, 1.5, 1.5])

    # Compute top clades here so Panel A regressions and Panel C can use them
    clade_col_totals = clade_comp_df.sum(axis=0).sort_values(ascending=False)
    top_clades = clade_col_totals.head(13).index.tolist()
    n_clades = len(top_clades)

    # Panel A: two stacked regressions (both x = Mean Insertions per Sample)
    #   top:    vs Intergenic log2(obs/exp)  (matching heatmap on Panel C)
    #   bottom: vs LTR-RT count in reference genome
    A_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=left_gs[0, 0], hspace=0.55)
    ax_A1 = fig.add_subplot(A_gs[0, 0])
    ax_A2 = fig.add_subplot(A_gs[1, 0])
    panel_intergenic_regression(ax_A1, clade_comp_df, merged, top_clades,
                                ctx_lengths_kb=ctx_lengths_kb,
                                panel_letter="A")
    panel_ref_count_regression(ax_A2, merged, ltr_age, top_clades,
                               panel_letter="B")

    # Panel C: TE superfamily — proportion violin (top) + enrichment heatmap (bottom)
    C_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=left_gs[1, 0],
        hspace=0.05,
        height_ratios=[1.0, 0.55])
    ax_C_violin = fig.add_subplot(C_gs[0, 0])
    ax_C_heat   = fig.add_subplot(C_gs[1, 0])

    fam_comp_df = fam_comp_df.drop(columns=["mixture"], errors="ignore")
    fam_long = fam_long[fam_long["category"] != "mixture"].copy()
    fam_col_totals = fam_comp_df.sum(axis=0).sort_values(ascending=False)
    top_fams = fam_col_totals.head(10).index.tolist()
    n_fams = len(top_fams)
    panel_context_prop_violin(ax_C_violin, fam_long,
                              panel_letter="C", top_n=10,
                              show_xticklabels=False,
                              top_cats_override=top_fams,
                              context_lengths_kb=ctx_lengths_kb)
    panel_context_enrichment(ax_C_heat, fam_comp_df,
                             panel_letter=None, top_n=10, label="TE Superfamily",
                             show_xticklabels=True,
                             top_cats_override=top_fams,
                             context_lengths_kb=ctx_lengths_kb)
    ax_C_violin.set_xlim(-0.5, n_fams - 0.5)
    ax_C_heat.set_xlim(-0.5, n_fams - 0.5)

    # Panel D: LTR-RT family — proportion violin (top) + enrichment heatmap (bottom)
    D_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=left_gs[2, 0],
        hspace=0.05,
        height_ratios=[1.0, 0.55])
    ax_D_violin = fig.add_subplot(D_gs[0, 0])
    ax_D_heat   = fig.add_subplot(D_gs[1, 0])

    panel_context_prop_violin(ax_D_violin, clade_long,
                              panel_letter="D", top_n=13,
                              show_xticklabels=False,
                              top_cats_override=top_clades,
                              context_lengths_kb=ctx_lengths_kb)
    panel_context_enrichment(ax_D_heat, clade_comp_df,
                             panel_letter=None, top_n=13, label="LTR-RT Family",
                             show_xticklabels=True,
                             top_cats_override=top_clades,
                             context_lengths_kb=ctx_lengths_kb)
    ax_D_violin.set_xlim(-0.5, n_clades - 0.5)
    ax_D_heat.set_xlim(-0.5, n_clades - 0.5)

    # ── Right column: Panel E (collapsed metagene) + Panel F (K2P age) ──
    if n_age > 0:
        # Split right column into metagene (top) and age stack (bottom).
        # Give each age strip ~60 % of a metagene strip's height.
        right_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0, 1],
            hspace=0.12,
            height_ratios=[1.0, n_age * 0.60])
        ax_E = fig.add_subplot(right_gs[0, 0])
        panel_metagene_single(ax_E, metagene_df, panel_letter="E")

        age_gs = gridspec.GridSpecFromSubplotSpec(
            n_age, 1, subplot_spec=right_gs[1, 0],
            hspace=0.08)
        axes_F = [fig.add_subplot(age_gs[i, 0]) for i in range(n_age)]
        panel_ltr_age_density(axes_F, ltr_age, panel_letter="F")
    else:
        # No age data: single collapsed metagene fills the right column
        right_gs = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=outer[0, 1])
        ax_E = fig.add_subplot(right_gs[0, 0])
        panel_metagene_single(ax_E, metagene_df, panel_letter="E")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_page_command(pdf):
    """Final page: record the exact command line for reproducibility."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    import shlex
    cmd_str = " ".join(shlex.quote(a) for a in sys.argv)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Wrap long command lines for display
    wrapped = []
    line = ""
    for token in shlex.split(cmd_str):
        q = shlex.quote(token)
        candidate = (line + " " + q).strip() if line else q
        if len(candidate) > 80 and line:
            wrapped.append(line + " \\")
            line = "    " + q
        else:
            line = candidate
    if line:
        wrapped.append(line)
    cmd_display = "\n".join(wrapped)

    ax.text(0.05, 0.95, "Reproducibility", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")
    ax.text(0.05, 0.88, f"Date:  {timestamp}", transform=ax.transAxes,
            fontsize=9, fontfamily="monospace", va="top", ha="left")
    ax.text(0.05, 0.82, "Command:", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")
    ax.text(0.07, 0.77, cmd_display.replace("$", r"\$"),
            transform=ax.transAxes,
            fontsize=7.5, fontfamily="monospace",
            va="top", ha="left", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7",
                      ec="#cccccc", alpha=0.95))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 65)
    print("TE Figure Generator")
    print("=" * 65)

    print(f"\n[1] Master: {args.master}")
    master = load_master(args.master, args.sample_col)
    print(f"    {len(master)} samples, {len(master.columns)} features")

    print(f"\n[2] Loading TE files …")
    summaries, missing = {}, 0
    for sid in master.index:
        s = summarise_sample(sid, args.te_pattern, args.awk_filter)
        if s:
            summaries[sid] = s
        else:
            missing += 1
    print(f"    Loaded: {len(summaries)}  Missing/empty: {missing}")

    if len(summaries) < 5:
        sys.exit("ERROR: Too few samples. Check --te-pattern.")

    te_df = pd.DataFrame.from_dict(summaries, orient="index")
    te_df.index.name = args.sample_col

    # Merge; convert only clearly numeric master columns (leaves strings intact)
    merged = master.join(te_df, how="inner")
    for col in master.columns:
        if col in te_df.columns:
            continue
        trial = pd.to_numeric(merged[col], errors="coerce")
        # Only overwrite if >70 % of non-null values converted cleanly
        non_null = merged[col].notna().sum()
        if non_null > 0 and trial.notna().sum() / non_null >= 0.7:
            merged[col] = trial

    print(f"    Merged: {len(merged)} samples")

    n_clade = sum(1 for c in merged.columns if c.startswith("te_clade_"))
    n_fam   = sum(1 for c in merged.columns if c.startswith("te_fam_"))
    print(f"    te_fam_* (superfamily) cols: {n_fam}  te_clade_* (family) cols: {n_clade}")

    # ── optional LTR-RT age file ─────────────────────────────────────────────
    ltr_age = None
    if args.ltr_age:
        try:
            ltr_age = load_ltr_age(args.ltr_age)
            n_qualifying = sum(1 for v in ltr_age.values() if len(v) > 100)
            total_elems  = sum(len(v) for v in ltr_age.values())
            print(f"\n[2c] LTR age file: {total_elems:,} elements across "
                  f"{len(ltr_age)} families "
                  f"({n_qualifying} with >100 elements) — {args.ltr_age}")
        except Exception as e:
            print(f"\n[2c] WARNING: could not read LTR age file ({e}); skipping")

    # ── optional CRM file for centromere approximation ───────────────────────
    crm_intervals = None
    if args.crm:
        try:
            crm_intervals = parse_crm(args.crm)
            n_crm = sum(len(v) for v in crm_intervals.values())
            print(f"\n[2b] CRM file loaded: {n_crm} elements across "
                  f"{len(crm_intervals)} chromosomes from {args.crm}")
        except Exception as e:
            print(f"\n[2b] WARNING: could not read CRM file ({e}); skipping")

    # ── optional FAI for authoritative chromosome lengths ─────────────────────
    fai_lengths = None
    if args.fai:
        try:
            fai_df = pd.read_csv(args.fai, sep="\t", header=None,
                                 names=["chr", "length", "offset", "bases", "bytes"])
            fai_lengths = dict(zip(fai_df["chr"], fai_df["length"].astype(int)))
            print(f"\n[3a] FAI loaded: {len(fai_lengths)} sequences from {args.fai}")
        except Exception as e:
            print(f"\n[3a] WARNING: could not read FAI ({e}); inferring lengths from data")

    # ── karyotype enrichment analysis ────────────────────────────────────────
    print(f"\n[3] Loading raw positions for karyotype analysis …")
    pos_df = load_all_bed_positions(merged, args.te_pattern, args.awk_filter)
    print(f"    {len(pos_df):,} total insertion records")

    # Build glyphosate resistance map {sample_id: True/False/None}
    # None = unknown (NA); only True/False samples are used in R/S analyses.
    gly_map = {}
    for sid in merged.index:
        is_r = None
        for gcol in ["Glyphosate_R", "glyphosate_res"]:
            if gcol in merged.columns:
                val = str(merged.loc[sid, gcol]).strip()
                if val in ("1", "1.0", "R", "Resistant"):
                    is_r = True
                    break
                elif val in ("0", "0.0", "S", "Susceptible"):
                    is_r = False
                    break
        gly_map[sid] = is_r

    n_r_samp = sum(1 for v in gly_map.values() if v is True)
    n_s_samp = sum(1 for v in gly_map.values() if v is False)
    print(f"    Resistant: {n_r_samp} samples  |  Susceptible: {n_s_samp} samples")

    clusters_df = find_enriched_clusters(pos_df, gly_map)
    if not clusters_df.empty:
        n_sig = int((clusters_df["padj"] < 0.05).sum())
        n_R_e = int((clusters_df["enrichment"] == "R").sum())
        n_S_e = int((clusters_df["enrichment"] == "S").sum())
        print(f"    Clusters tested: {len(clusters_df):,}  "
              f"significant: {n_sig}  "
              f"(R-enriched: {n_R_e}, S-enriched: {n_S_e})")

        sig = clusters_df[clusters_df["enrichment"].isin(["R", "S"])].copy()
        if not sig.empty:
            sig = sig.sort_values("padj")
            print(f"\n    {'─'*70}")
            print(f"    {'#':<4} {'Chr':<18} {'Position':>12}  "
                  f"{'R':>5} {'S':>5} {'Total':>6}  "
                  f"{'log2FC':>7}  {'p (raw)':>10}  {'q (BH)':>10}  Enr")
            print(f"    {'─'*70}")
            for i, (_, row) in enumerate(sig.iterrows(), 1):
                print(f"    {i:<4} {row['Chr']:<18} {int(row['pos']):>12,}  "
                      f"{int(row['r_count']):>5} {int(row['s_count']):>5} "
                      f"{int(row['total']):>6}  "
                      f"{row['log2fc']:>+7.3f}  "
                      f"{row['pval']:>10.3e}  "
                      f"{row['padj']:>10.3e}  "
                      f"{row['enrichment']}")
            print(f"    {'─'*70}")
            print(f"    R/S sample totals: {n_r_samp} resistant, {n_s_samp} susceptible\n")

        # ── GFF gene / promoter annotation ───────────────────────────────────
        if args.gff and not sig.empty:
            print(f"    Loading GFF: {args.gff} …")
            genes_df = load_gff_genes(args.gff)
            print(f"    {len(genes_df):,} gene records loaded")
            print()

            W = 80
            for _, row in sig.sort_values("padj").iterrows():
                chrom, pos = row["Chr"], int(row["pos"])
                enr = row["enrichment"]
                print(f"    {'═'*W}")
                print(f"    {chrom}:{pos:,}  [{enr}-enriched]  "
                      f"log2FC={row['log2fc']:+.3f}  "
                      f"q(BH)={row['padj']:.3e}  "
                      f"(R={int(row['r_count'])}/{n_r_samp}, "
                      f"S={int(row['s_count'])}/{n_s_samp})")
                print(f"    {'─'*W}")

                te_desc = describe_cluster_tes(chrom, pos, pos_df)
                print(f"    TE identity:")
                if te_desc:
                    for td in te_desc:
                        print(f"      {td}")
                else:
                    print(f"      (TE_ID not available)")
                print()

                hits = annotate_insertion(chrom, pos, genes_df)
                if not hits:
                    print(f"    No gene body or promoter overlap within 1 kb")
                else:
                    print(f"    {'Overlap':<10} {'Gene':<22} {'Biotype':<22} "
                          f"{'Strand':<7} {'Gene coords':<28} {'Dist to TSS':>12}")
                    print(f"    {'─'*W}")
                    for h in sorted(hits, key=lambda x: (x["overlap"], x["dist_bp"])):
                        coords = f"{h['gene_start']:,}–{h['gene_end']:,}"
                        dist   = (f"{h['dist_bp']:,} bp" if h["overlap"] == "PROMOTER"
                                  else "—")
                        print(f"    {h['overlap']:<10} {h['gene_name']:<22} "
                              f"{h['biotype']:<22} {h['strand']:<7} "
                              f"{coords:<28} {dist:>12}")
                print()
            print(f"    {'═'*W}\n")
    else:
        print("    No multi-sample clusters found.")

    print(f"\n[4] Writing {args.output} …")
    with PdfPages(args.output) as pdf:
        print("    Page 1: composition & comparisons")
        build_page1(pdf, merged)
        print("    Page 2: regressions (top-3 by R²)")
        build_page2(pdf, merged)
        print("    Page 3: glyphosate stacked bars & BioClim")
        build_page3(pdf, merged)
        print("    Page 4: per-sample stacked bars (superfamily + family, side-by-side)")
        build_page4(pdf, merged)
        print("    Page 5: karyotype (A) + sample TE sharing heatmap (B)")
        build_page_kary_sharing(pdf, clusters_df, pos_df,
                                fai_lengths=fai_lengths,
                                crm_intervals=crm_intervals,
                                gly_map=gly_map, merged=merged)
        print("    Page 5b: insertion cluster frequency spectrum")
        build_page_insertion_spectrum(pdf, pos_df)
        if args.gff:
            print("    Page 6: gene-context enrichment, metagene & K2P age")
            build_page_gene_context(pdf, merged, pos_df, args.gff,
                                    fai_lengths=fai_lengths,
                                    ltr_age=ltr_age)

        print("    Final page: command reproducibility")
        build_page_command(pdf)

        meta = pdf.infodict()
        meta["Title"]   = "TE Insertion Analysis"
        meta["Subject"] = "TE insertions vs phenotype/environment"

    print(f"\n{'=' * 65}")
    print(f"Saved: {args.output}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
