# uv pip install "paddlepaddle==2.6.*" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF → clean text (+ inline tables) + separate images/equations + JSON index.

Outputs per PDF:
out/<stem>/
  clean_text.md                # main human-readable file (or .txt)
  manifest.json                # links ALL assets + anchors used in text
  pages/page_0000.json         # page-level debug (optional)
  tables/page0003_table000.csv # machine tables (also inlined)
  images/page_0002/img000.png  # embedded images (xobjects)
  figures/page_0004/fig000.png # detected figure crops
  equations/page_0005/eq000.tex# externalized equations (if not inlined)
  <stem>.zip

Stack:
- PyMuPDF for text/blocks/words/images/links
- layoutparser (PaddleDetection / PubLayNet) for regions + 2-column reading order
- Camelot + Tabula for tables (region-based)
- Nougat for equations (per-crop); inline if 'clean', else externalized
- OCRmyPDF automatically if pages lack selectable text
- GROBID (optional) to enrich manifest (metadata/refs)
"""

import os, re, json, shutil, zipfile, tempfile, subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import pandas as pd
import layoutparser as lp
from lxml import etree
import requests

# IMPORTANT: PyMuPDF pip name is "pymupdf" but import is "fitz"
import fitz as pymupdf  # alias for readability with existing code

# Optional deps
try:
    import camelot  # type: ignore
except Exception:
    camelot = None
try:
    import tabula   # type: ignore
except Exception:
    tabula = None

# =========================
# CONFIG — EDIT THESE
# =========================
INPUT_DIR  = Path(os.environ.get("PDF_INPUT_DIR", "downloaded_pdfs/"))  # <— set me
OUTPUT_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "parser/"))          # <— set me
FORMAT     = os.environ.get("TEXT_FORMAT", "md")   # "md" or "txt"
GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")

DPI_LAYOUT = int(os.environ.get("DPI_LAYOUT", "220"))
RUN_OCR    = True      # auto run OCRmyPDF when needed
USE_CAMELOT= True
USE_TABULA = True

# Equation heuristics (decide inline vs external)
EQ_MIN_LEN             = 8
EQ_MAX_INLINE_CHARS    = 180      # inline if short-ish
EQ_MAX_INLINE_NEWLINES = 1
EQ_INLINE_SYM_SHARE    = 0.10     # math symbol density threshold for inline confidence
EQ_PAD_PX              = 8
N_PAGE_EQUATION_LIMIT  = 200

# =========================
# Helpers
# =========================

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def save_json(x, p: Path):
    ensure_dir(p.parent)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")

def sanitize(s:str)->str:
    return re.sub(r"[^a-zA-Z0-9._-]+","_",s).strip("_")

def run(cmd: List[str], check=True):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{err}")
    return p.returncode, out, err

# OCR
def need_ocr(pdf: Path, sample_pages=6)->bool:
    doc = pymupdf.open(pdf.as_posix())
    n = len(doc); idxs = list(range(n))
    if n>sample_pages:
        step = max(1, n//sample_pages)
        idxs = list(range(0,n,step))[:sample_pages]
    empty = sum(1 for i in idxs if len(doc[i].get_text("words"))==0)
    doc.close()
    return empty >= max(1, len(idxs)//2)

def ocr_if_needed(pdf: Path, tmp: Path)->Path:
    if not RUN_OCR or not shutil.which("ocrmypdf") or not need_ocr(pdf):
        return pdf
    out = tmp / (pdf.stem + ".ocred.pdf")
    print(f"[OCR] ocrmypdf → {out.name}")
    run([
        "ocrmypdf","--optimize","3","--deskew","--rotate-pages","--clean",
        "--jobs", str(os.cpu_count() or 4),
        pdf.as_posix(), out.as_posix()
    ])
    return out

# ============== Layout model (Detectron-free; Paddle backend; safe fallback) ==============
class LayoutModel:
    def __init__(self):
        self.backend = "none"
        self.model = None
        try:
            # PaddleDetection backend on PubLayNet (no Detectron needed)
            # Note: keep args minimal for compatibility across LP versions.
            self.model = lp.PaddleDetectionLayoutModel(
                config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                label_map={0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"},
                enforce_cpu=True  # set False if you have GPU Paddle installed
            )
            self.backend = "paddle"
        except Exception as e:
            print("[WARN] Paddle backend unavailable -> running in NO-ML fallback:", e)
            self.backend = "none"

    def detect(self, bgr: np.ndarray):
        if self.backend != "paddle":
            return []
        # Paddle expects RGB
        layout = self.model.detect(bgr[..., ::-1])
        # Manual score threshold
        MIN_SCORE = 0.5
        out=[]
        for e in layout:
            if float(e.score) < MIN_SCORE:
                continue
            x1,y1,x2,y2 = map(float, e.block.points.flatten())
            out.append({"label": e.type, "score": float(e.score), "bbox_img":[x1,y1,x2,y2]})
        return out

def img_bbox_to_pdf(b, iw, ih, rect)->List[float]:
    x1,y1,x2,y2 = b
    W,H = rect.width, rect.height
    X1 = x1/iw*W; X2 = x2/iw*W
    Y1 = H - (y1/ih*H); Y2 = H - (y2/ih*H)
    return [X1,Y1,X2,Y2]

def bbox_to_rect(b): return pymupdf.Rect(float(b[0]),float(b[1]),float(b[2]),float(b[3]))

# Reading order (1–2 columns)
def reading_text_from_boxes(page: pymupdf.Page, boxes: List[List[float]])->str:
    if not boxes: return page.get_text("text")
    W = float(page.rect.width)
    try:
        from sklearn.cluster import KMeans
        centers = np.array([((x0+x2)/2)/W for x0,y0,x2,y2 in boxes]).reshape(-1,1)
        if len(boxes)>=6:
            km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(centers)
            labels = km.labels_; cols={0:[],1:[]}
            for i,l in enumerate(labels): cols[int(l)].append(i)
            left_lbl = np.argmin([np.mean([centers[i][0] for i in cols[k]]) for k in (0,1)])
            left = sorted(cols[left_lbl], key=lambda i:(boxes[i][1], boxes[i][0]))
            right= sorted(cols[1-left_lbl], key=lambda i:(boxes[i][1], boxes[i][0]))
            lmax = max(boxes[i][2] for i in left) if left else 0
            rmin = min(boxes[i][0] for i in right) if right else W
            gutter = (rmin - lmax)/W
            if gutter<0.04:
                order = sorted(range(len(boxes)), key=lambda i:(boxes[i][1],boxes[i][0]))
            else:
                order=[]; i=j=0
                while i<len(left) or j<len(right):
                    if j>=len(right) or (i<len(left) and boxes[left[i]][1]<=boxes[right[j]][1]):
                        order.append(left[i]); i+=1
                    else:
                        order.append(right[j]); j+=1
        else:
            order = sorted(range(len(boxes)), key=lambda i:(boxes[i][1], boxes[i][0]))
    except Exception:
        order = sorted(range(len(boxes)), key=lambda i:(boxes[i][1], boxes[i][0]))

    out=[]
    for i in order:
        t = page.get_textbox(bbox_to_rect(boxes[i])).strip()
        if t: out.append(t)
    return "\n".join(out)

# Lines (for equation candidates)
def lines_from_page(page: pymupdf.Page)->List[Dict[str,Any]]:
    d = page.get_text("dict")
    lines=[]
    for b in d.get("blocks", []):
        if b.get("type",0)!=0: continue
        for l in b.get("lines", []):
            bbox = l.get("bbox")
            text = "".join(s.get("text","") for s in l.get("spans",[]))
            if text.strip():
                lines.append({"text":text, "bbox":[float(x) for x in bbox]})
    return lines

# Images
def extract_xobjects(doc: pymupdf.Document, pno:int, outdir:Path)->List[Dict[str,Any]]:
    ensure_dir(outdir); page=doc[pno]; infos=[]
    for idx, im in enumerate(page.get_images(full=True)):
        xref = im[0]
        try:
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n >= 5: pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            path = outdir / f"img{idx:03d}.png"
            pix.save(path.as_posix())
            infos.append({"file":path.relative_to(OUTPUT_DIR).as_posix(),"width":pix.width,"height":pix.height,"xref":xref})
        except Exception as e:
            infos.append({"error":str(e),"xref":xref})
    return infos

def save_crop(page: pymupdf.Page, rect: pymupdf.Rect, out_png: Path, dpi=DPI_LAYOUT):
    ensure_dir(out_png.parent)
    zoom = dpi/72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom,zoom), clip=rect, alpha=False)
    pix.save(out_png.as_posix())

# Tables (region-based → CSV + Markdown)
def extract_tables_region(pdf: Path, page_onebased:int, bbox: List[float])->List[pd.DataFrame]:
    area=",".join(f"{v:.2f}" for v in bbox)
    tables=[]
    if USE_CAMELOT and camelot is not None:
        try:
            for flavor in ("lattice","stream"):
                ts = camelot.read_pdf(pdf.as_posix(), pages=str(page_onebased), flavor=flavor, table_areas=[area])
                for t in ts:
                    if getattr(t, "df", None) is not None and t.df.size>0:
                        tables.append(t.df)
                if tables: return tables
        except Exception:
            pass
    if USE_TABULA and tabula is not None:
        try:
            dfs = tabula.read_pdf(
                pdf.as_posix(), pages=page_onebased,
                area=[[bbox[1],bbox[0],bbox[3],bbox[2]]], guess=False, stream=True,
                pandas_options={"dtype":str}
            )
            for df in dfs or []:
                if isinstance(df,pd.DataFrame) and df.size>0:
                    tables.append(df)
        except Exception:
            pass
    return tables

def df_to_markdown(df: pd.DataFrame)->str:
    df2 = df.copy().fillna("")
    df2 = df2.applymap(lambda x: str(x).replace("\n"," ").strip())
    has_header = (df2.iloc[0].apply(lambda s: bool(re.search(r"[A-Za-z]", s))).sum() >= max(2, df2.shape[1]//2))
    if has_header:
        header = "| " + " | ".join(df2.iloc[0].tolist()) + " |\n"
        sep    = "| " + " | ".join(["---"]*df2.shape[1]) + " |\n"
        body   = "\n".join("| " + " | ".join(row) + " |" for row in df2.iloc[1:].astype(str).values.tolist())
    else:
        header = "| " + " | ".join([f"col{i}" for i in range(df2.shape[1])]) + " |\n"
        sep    = "| " + " | ".join(["---"]*df2.shape[1]) + " |\n"
        body   = "\n".join("| " + " | ".join(row) + " |" for row in df2.astype(str).values.tolist())
    return header + sep + (body + ("\n" if body else ""))

# Equations — crop → mini-PDF → Nougat; decide inline vs external
MATH_SYMS = set(r"=+−-–—*/×·•^_<>≤≥≈≃≅≡≠∞∑∏∫∇∂∆√∈∉∪∩⊆⊂⊇⊃⇒⇔→←↔αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ’′″‴…°%±ληκϕφψχθμ")
def sym_share(s:str)->float:
    s2 = "".join(ch for ch in s if not ch.isspace())
    if not s2: return 0.0
    return sum(ch in MATH_SYMS for ch in s2) / len(s2)

def make_region_pdf(page: pymupdf.Page, rect: pymupdf.Rect, out_pdf: Path, dpi=DPI_LAYOUT):
    zoom = dpi/72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom,zoom), clip=rect, alpha=False)
    with tempfile.TemporaryDirectory() as td:
        png = Path(td)/"crop.png"; pix.save(png.as_posix())
        doc = pymupdf.open(); w_pt = pix.width*72.0/dpi; h_pt = pix.height*72.0/dpi
        pg = doc.new_page(width=w_pt, height=h_pt)
        pg.insert_image(pymupdf.Rect(0,0,w_pt,h_pt), filename=png.as_posix())
        ensure_dir(out_pdf.parent); doc.save(out_pdf.as_posix()); doc.close()

def run_nougat(pdf_crop: Path)->Optional[str]:
    if not shutil.which("nougat"): return None
    with tempfile.TemporaryDirectory() as td:
        out_md = Path(td)/"o.md"
        run(["nougat", pdf_crop.as_posix(), "--out", out_md.as_posix()], check=False)
        if not out_md.exists(): return None
        txt = out_md.read_text(encoding="utf-8", errors="ignore").strip()
        m = re.findall(r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])", txt)
        if m:
            raw = m[0]
            if raw.startswith("$$") and raw.endswith("$" * 2): return raw[2:-2].strip()
            if raw.startswith(r"\[") and raw.endswith(r"\]"):   return raw[2:-2].strip()
        return txt or None

def pad_rect(rect: pymupdf.Rect, px=EQ_PAD_PX, dpi=DPI_LAYOUT, page=None):
    pad_pt = px*72.0/dpi
    r = pymupdf.Rect(rect.x0-pad_pt, rect.y0-pad_pt, rect.x1+pad_pt, rect.y1+pad_pt)
    return (r & page.rect) if page is not None else r

# GROBID (optional)
def grobid_process(pdf: Path, url=GROBID_URL)->Optional[str]:
    try:
        with open(pdf,"rb") as fh:
            r = requests.post(
                f"{url}/api/processFulltextDocument",
                files={"input": fh},
                data={
                    "consolidateCitations":"1",
                    "includeRawCitations":"1",
                    "includeRawAffiliations":"1"
                },
                timeout=180
            )
        if r.status_code==200 and r.text.strip():
            return r.text
    except Exception:
        pass
    return None

def parse_grobid(tei_xml:str)->Dict[str,Any]:
    ns={"tei":"http://www.tei-c.org/ns/1.0"}
    root=etree.fromstring(tei_xml.encode("utf-8"))
    out={"title":None,"abstract":None,"authors":[], "bibliography":[]}
    t=root.xpath("//tei:fileDesc/tei:titleStmt/tei:title/text()",namespaces=ns)
    out["title"]=t[0].strip() if t else None
    a=root.xpath("//tei:profileDesc/tei:abstract//text()",namespaces=ns)
    out["abstract"]=" ".join(s.strip() for s in a if s.strip()) or None
    for pers in root.xpath("//tei:sourceDesc//tei:author",namespaces=ns):
        name="".join(pers.xpath(".//tei:persName//text()",namespaces=ns)).strip()
        aff =" ".join(pers.xpath(".//tei:affiliation//text()",namespaces=ns)).strip() or None
        out["authors"].append({"name":name or None,"affiliation":aff})
    for b in root.xpath("//tei:listBibl/tei:biblStruct",namespaces=ns):
        title=" ".join(b.xpath(".//tei:title//text()",namespaces=ns)).strip() or None
        year =(b.xpath(".//tei:date/@when",namespaces=ns) or [""])[0][:4] or None
        doi  =(b.xpath(".//tei:idno[@type='DOI']/text()",namespaces=ns) or [None])[0]
        out["bibliography"].append({"title":title,"year":year,"doi":doi})
    return out

# ============== Core per-PDF extraction that builds ONE clean text file ==========
def process_pdf(pdf_path: Path, out_root: Path)->Dict[str,Any]:
    print(f"\n== {pdf_path.name} ==")
    stem=sanitize(pdf_path.stem)
    workdir=out_root/stem
    if workdir.exists(): shutil.rmtree(workdir)
    ensure_dir(workdir)

    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)
        parse_pdf = ocr_if_needed(pdf_path, tmp)
        doc = pymupdf.open(parse_pdf.as_posix())
        model = LayoutModel()

        # Optional metadata
        grobid = None
        tei = grobid_process(parse_pdf)
        if tei:
            (workdir/"metadata_grobid.xml").write_text(tei, encoding="utf-8")
            grobid = parse_grobid(tei)

        text_lines = []   # will join to single file
        manifest = {
            "file": pdf_path.name,
            "output_dir": workdir.as_posix(),
            "text_file": None,
            "ocr_applied": parse_pdf.name.endswith(".ocred.pdf"),
            "metadata_pdf": doc.metadata or {},
            "grobid": grobid,
            "assets": {
                "images": [],      # [{id, kind, page, path}]
                "figures": [],     # [{id, page, path}]
                "tables": [],      # [{id, page, csv, md_snippet}]
                "equations": []    # [{id, page, bbox, path_tex?, path_png?, inline:bool, latex?}]
            },
            "anchors": []          # [{type, id, page, char_start, char_end}]
        }

        for pno in range(len(doc)):
            page = doc[pno]
            text_lines.append(f"\n\n# [PAGE {pno+1}]\n")
            page_char_start = sum(len(x) for x in text_lines)

            # Raster for layout
            zoom = DPI_LAYOUT/72.0
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom,zoom), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if img.shape[2]==4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            dets = model.detect(img)

            # Reading-order text from Text/Title/List regions
            text_boxes=[]
            for d in dets:
                if d["label"] in ("Title","Text","List"):
                    text_boxes.append(img_bbox_to_pdf(d["bbox_img"], img.shape[1], img.shape[0], page.rect))
            ro_text = reading_text_from_boxes(page, text_boxes)
            if ro_text.strip():
                text_lines.append(ro_text.strip()+"\n")

            # Embedded images (XObjects)
            ximgs = extract_xobjects(doc, pno, workdir/"images"/f"page_{pno:04d}")
            for info in ximgs:
                if "file" in info:
                    asset_id = f"img_p{pno}_{len(manifest['assets']['images']):03d}"
                    manifest["assets"]["images"].append({"id":asset_id,"kind":"xobject","page":pno,"path":info["file"]})
                    text_lines.append(f"[FIGURE {asset_id}]\n")
                    manifest["anchors"].append({"type":"image","id":asset_id,"page":pno})

            # Detected figures (visual crops)
            fig_idx=0
            for d in dets:
                if d["label"]=="Figure":
                    bbox = img_bbox_to_pdf(d["bbox_img"], img.shape[1], img.shape[0], page.rect)
                    rect = bbox_to_rect(bbox)
                    outp = workdir/"figures"/f"page_{pno:04d}"/f"fig{fig_idx:03d}.png"
                    save_crop(page, rect, outp, dpi=DPI_LAYOUT)
                    asset_id = f"fig_p{pno}_{fig_idx:03d}"
                    manifest["assets"]["figures"].append({"id":asset_id,"page":pno,"path":outp.relative_to(OUTPUT_DIR).as_posix()})
                    text_lines.append(f"[FIGURE {asset_id}]\n")
                    manifest["anchors"].append({"type":"figure","id":asset_id,"page":pno})
                    fig_idx+=1

            # Tables (region-based → CSV + inline Markdown)
            tcount=0
            for d in dets:
                if d["label"]=="Table":
                    bbox = img_bbox_to_pdf(d["bbox_img"], img.shape[1], img.shape[0], page.rect)
                    dfs = extract_tables_region(parse_pdf, pno+1, bbox)
                    for df in dfs:
                        csv_name = f"page{pno:04d}_table{tcount:03d}.csv"
                        csv_path = workdir/"tables"/csv_name
                        ensure_dir(csv_path.parent)
                        df.to_csv(csv_path, index=False, header=False)
                        md = df_to_markdown(df)
                        asset_id = f"tbl_p{pno}_{tcount:03d}"
                        text_lines.append(f"\n[TABLE {asset_id}]\n")
                        text_lines.append(md+"\n")
                        manifest["assets"]["tables"].append({
                            "id": asset_id, "page": pno,
                            "csv": csv_path.relative_to(OUTPUT_DIR).as_posix(),
                            "md_snippet": md
                        })
                        manifest["anchors"].append({"type":"table","id":asset_id,"page":pno})
                        tcount+=1

            # Equations (per-line crops -> Nougat; decide inline vs external)
            lines = lines_from_page(page)
            eq_added=0
            for L in lines:
                if eq_added >= N_PAGE_EQUATION_LIMIT: break
                raw = L["text"].strip()
                if len(raw) < EQ_MIN_LEN: continue
                density = sym_share(raw)
                candidate = (density >= EQ_INLINE_SYM_SHARE) or bool(re.search(r"(\$\$|\\\[|\\\(|\\begin\{equation\})", raw))
                if not candidate: continue
                rect = bbox_to_rect(L["bbox"])
                rect = pad_rect(rect, px=EQ_PAD_PX, dpi=DPI_LAYOUT, page=page)
                with tempfile.TemporaryDirectory() as td2:
                    mini_pdf = Path(td2)/"crop.pdf"
                    make_region_pdf(page, rect, mini_pdf)
                    latex = run_nougat(mini_pdf) or ""
                inline_ok = (latex and len(latex) <= EQ_MAX_INLINE_CHARS and latex.count("\n") <= EQ_MAX_INLINE_NEWLINES)
                eq_id = f"eq_p{pno}_{eq_added:03d}"
                if inline_ok:
                    text_lines.append(f"\n[EQ {eq_id}] $${latex}$$\n")
                    manifest["assets"]["equations"].append({
                        "id":eq_id,"page":pno,"bbox":[rect.x0,rect.y0,rect.x1,rect.y1],
                        "inline": True, "latex": latex
                    })
                else:
                    png_path = workdir/"equations"/f"page_{pno:04d}"/f"{eq_id}.png"
                    save_crop(page, rect, png_path, dpi=DPI_LAYOUT)
                    tex_path = png_path.with_suffix(".tex")
                    tex_path.write_text(latex or raw, encoding="utf-8")
                    text_lines.append(f"\n[EQ {eq_id}] -> see {tex_path.name}\n")
                    manifest["assets"]["equations"].append({
                        "id":eq_id,"page":pno,"bbox":[rect.x0,rect.y0,rect.x1,rect.y1],
                        "inline": False,
                        "path_png": png_path.relative_to(OUTPUT_DIR).as_posix(),
                        "path_tex": tex_path.relative_to(OUTPUT_DIR).as_posix(),
                        "latex": latex or None,
                        "text_hint": raw
                    })
                manifest["anchors"].append({"type":"equation","id":eq_id,"page":pno})
                eq_added+=1

            page_char_end = sum(len(x) for x in text_lines)
            page_dbg = {"page": pno, "char_start": page_char_start, "char_end": page_char_end}
            ensure_dir(workdir/"pages")
            (workdir/"pages"/f"page_{pno:04d}.json").write_text(json.dumps(page_dbg), encoding="utf-8")

        # write the clean text file
        ext = ".md" if FORMAT=="md" else ".txt"
        text_path = workdir/("clean_text"+ext)
        text_path.write_text("".join(text_lines), encoding="utf-8")
        manifest["text_file"] = text_path.relative_to(OUTPUT_DIR).as_posix()

        # save manifest
        save_json(manifest, workdir/"manifest.json")

        # zip artifact
        zip_path = workdir.with_suffix(".zip")
        if zip_path.exists(): zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(workdir):
                for f in files:
                    fp = Path(root)/f
                    z.write(fp, arcname=fp.relative_to(workdir.parent))

        return {"dir": workdir.as_posix(), "zip": zip_path.as_posix(), "manifest": manifest}

# =========================
# Batch over INPUT_DIR
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    pdfs = sorted([p for p in INPUT_DIR.glob("**/*.pdf") if p.is_file()])
    if not pdfs:
        print(f"No PDFs under {INPUT_DIR}")
        return
    rows=[]
    for p in pdfs:
        try:
            res = process_pdf(p, OUTPUT_DIR)
            m = res["manifest"]
            rows.append({"file": m["file"], "text": m["text_file"], "zip": res["zip"]})
            print(f"OK: {p.name} -> {res['zip']}")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")
    # batch index
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/"batch_index.csv", index=False)
    print("Wrote:", (OUTPUT_DIR/"batch_index.csv").as_posix())

if __name__ == "__main__":
    main()

