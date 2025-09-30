#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-stack research PDF extractor with per-page equation crops → Nougat (LaTeX).

Edit the CONFIG block below and run this file.
"""

import os, re, io, math, json, shutil, zipfile, tempfile, subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import cv2
import requests
from lxml import etree
import pandas as pd
import layoutparser as lp

# Optional libs (tables)
try:
    import camelot
except Exception:
    camelot = None
try:
    import tabula  # needs Java
except Exception:
    tabula = None

# ===========================
# CONFIG — EDIT THIS SECTION
# ===========================
INPUT_DIR  = Path(os.environ.get("PDF_INPUT_DIR", "downloaded_pdfs/"))   # <— set me
OUTPUT_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "parser/"))      # <— change
GROBID_URL  = os.environ.get("GROBID_URL", "http://localhost:8070")
DPI_LAYOUT  = int(os.environ.get("DPI_LAYOUT", "220"))  # raster DPI for layout & crops
USE_CAMELOT = True   # set False to skip camelot
USE_TABULA  = True   # requires Java; used as fallback/extra
RUN_OCR     = True   # auto OCR if needed via OCRmyPDF
N_OCR_SAMPLE_PAGES = 6

# Equation heuristics
EQ_MIN_SYMBOL_SHARE = 0.15   # line is equation-like if share of math symbols >= this
EQ_MIN_LENGTH       = 10     # min chars on the line to consider for equation
EQ_BBOX_PAD_PX      = 8      # inflate crop bbox by a few pixels
N_PAGE_EQUATION_LIMIT = 200  # safety cap per page

# ===========================
# Utilities
# ===========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")

def run(cmd: List[str], check=True) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{err}")
    return p.returncode, out, err

# ===========================
# OCR preflight
# ===========================
def need_ocr(pdf_path: Path, sample_pages: int = N_OCR_SAMPLE_PAGES) -> bool:
    doc = fitz.open(pdf_path.as_posix())
    n = len(doc)
    idxs = list(range(n))
    if n > sample_pages:
        step = max(1, n // sample_pages)
        idxs = list(range(0, n, step))[:sample_pages]
    empty = 0
    for pno in idxs:
        if len(doc[pno].get_text("words")) == 0:
            empty += 1
    doc.close()
    return empty >= max(1, len(idxs)//2)

def ocr_if_needed(pdf_path: Path, tmpdir: Path) -> Path:
    if not RUN_OCR or not shutil.which("ocrmypdf"):
        return pdf_path
    if not need_ocr(pdf_path):
        return pdf_path
    out_pdf = tmpdir / (pdf_path.stem + ".ocred.pdf")
    print(f"[OCR] ocrmypdf → {out_pdf.name}")
    run(["ocrmypdf", "--optimize", "3", "--deskew", "--rotate-pages", "--clean",
         "--jobs", str(os.cpu_count() or 4), pdf_path.as_posix(), out_pdf.as_posix()])
    return out_pdf

# ===========================
# Layout model (PubLayNet)
# ===========================
class LayoutModel:
    def __init__(self, config="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"):
        self.model = lp.Detectron2LayoutModel(
            config_path=config,
            label_map={0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
        )

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        layout = self.model.detect(image_bgr)
        out = []
        for e in layout:
            x1, y1, x2, y2 = map(float, e.block.points.flatten())
            out.append({"label": e.type, "score": float(e.score),
                        "bbox_img": [x1, y1, x2, y2]})
        return out

def img_bbox_to_pdf_bbox(img_bbox, img_w, img_h, page_rect: fitz.Rect) -> List[float]:
    x1, y1, x2, y2 = img_bbox
    W, H = page_rect.width, page_rect.height
    X1 = x1 / img_w * W
    X2 = x2 / img_w * W
    Y1 = H - (y1 / img_h * H)  # top in PDF coords
    Y2 = H - (y2 / img_h * H)  # bottom in PDF coords
    return [X1, Y1, X2, Y2]

def bbox_to_rect(b):
    return fitz.Rect(float(b[0]), float(b[1]), float(b[2]), float(b[3]))

# ===========================
# Reading order (2-column)
# ===========================
def reading_order_from_boxes(page: fitz.Page, text_boxes_pdf: List[List[float]]) -> str:
    if not text_boxes_pdf:
        return page.get_text("text")
    W = float(page.rect.width)
    try:
        from sklearn.cluster import KMeans
        centers = np.array([((x0+x2)/2)/W for x0,y0,x2,y2 in text_boxes_pdf]).reshape(-1,1)
        if len(text_boxes_pdf) >= 6:
            km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(centers)
            labels = km.labels_
            cols = {0:[], 1:[]}
            for i, lab in enumerate(labels):
                cols[int(lab)].append(i)
            left_label = np.argmin([np.mean([centers[i][0] for i in cols[k]]) for k in (0,1)])
            left = sorted(cols[left_label], key=lambda i: (text_boxes_pdf[i][1], text_boxes_pdf[i][0]))
            right = sorted(cols[1-left_label], key=lambda i: (text_boxes_pdf[i][1], text_boxes_pdf[i][0]))
            lmax = max(text_boxes_pdf[i][2] for i in left) if left else 0
            rmin = min(text_boxes_pdf[i][0] for i in right) if right else W
            gutter = (rmin - lmax)/W
            indices = []
            if gutter < 0.04:
                indices = sorted(range(len(text_boxes_pdf)), key=lambda i: (text_boxes_pdf[i][1], text_boxes_pdf[i][0]))
            else:
                i=j=0
                while i < len(left) or j < len(right):
                    if j >= len(right) or (i < len(left) and text_boxes_pdf[left[i]][1] <= text_boxes_pdf[right[j]][1]):
                        indices.append(left[i]); i+=1
                    else:
                        indices.append(right[j]); j+=1
        else:
            indices = sorted(range(len(text_boxes_pdf)), key=lambda i: (text_boxes_pdf[i][1], text_boxes_pdf[i][0]))
    except Exception:
        indices = sorted(range(len(text_boxes_pdf)), key=lambda i: (text_boxes_pdf[i][1], text_boxes_pdf[i][0]))

    chunks = []
    for i in indices:
        rect = bbox_to_rect(text_boxes_pdf[i])
        t = page.get_textbox(rect).strip()
        if t:
            chunks.append(t)
    return "\n".join(chunks)

# ===========================
# Embedded images & figures
# ===========================
def extract_embedded_images(doc: fitz.Document, pno: int, outdir: Path) -> List[Dict[str, Any]]:
    ensure_dir(outdir)
    page = doc[pno]
    infos = []
    for idx, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            path = outdir / f"page{pno:04d}_img{idx:03d}.png"
            pix.save(path.as_posix())
            infos.append({"file": path.name, "width": pix.width, "height": pix.height, "xref": xref})
        except Exception as e:
            infos.append({"error": str(e), "xref": xref})
    return infos

def save_crop(page: fitz.Page, rect: fitz.Rect, out_path: Path, dpi: int = DPI_LAYOUT):
    ensure_dir(out_path.parent)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    pix.save(out_path.as_posix())

# ===========================
# Tables (region-based)
# ===========================
def extract_tables_region(pdf_path: Path, page_no_1based: int, pdf_bbox: List[float]) -> List[pd.DataFrame]:
    tables = []
    area = ",".join(f"{v:.2f}" for v in pdf_bbox)
    # Camelot first
    if USE_CAMELOT and camelot is not None:
        try:
            for flavor in ("lattice", "stream"):
                ts = camelot.read_pdf(pdf_path.as_posix(), pages=str(page_no_1based),
                                      flavor=flavor, table_areas=[area])
                for t in ts:
                    if t.df is not None and t.df.size > 0:
                        tables.append(t.df)
                if tables:
                    return tables
        except Exception:
            pass
    # Tabula fallback
    if USE_TABULA and tabula is not None:
        try:
            dfs = tabula.read_pdf(pdf_path.as_posix(),
                                  pages=page_no_1based,
                                  area=[[pdf_bbox[1], pdf_bbox[0], pdf_bbox[3], pdf_bbox[2]]],
                                  guess=False, stream=True,
                                  pandas_options={"dtype": str})
            for df in dfs or []:
                if isinstance(df, pd.DataFrame) and df.size > 0:
                    tables.append(df)
        except Exception:
            pass
    return tables

# ===========================
# Equations — per page (crop) → Nougat
# ===========================
MATH_CHARS = set(r"=+−-–—*/×·•^_<>≤≥≈≃≅≡≠∞∑∏∫∇∂∆√∈∉∪∩⊆⊂⊇⊃⇒⇔→←↔αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ’′″‴…°%±ληκϕφψχθ μ")
MATH_DELIMS = ("$$", r"\[", r"\]", r"\begin{equation}", r"\end{equation}", r"\(", r"\)")

def is_equation_like(text: str) -> bool:
    t = text.strip()
    if not t or len(t) < EQ_MIN_LENGTH:
        return False
    if any(d in t for d in MATH_DELIMS):
        return True
    if re.search(r"^\(?\s*Eq\.?\s*\d+", t, re.IGNORECASE):
        return True
    # symbol density
    if t:
        share = sum(ch in MATH_CHARS for ch in t) / max(1, len(t))
        if share >= EQ_MIN_SYMBOL_SHARE:
            return True
    # centered-looking (common for display equations)
    if re.match(r"^\s{0,5}[^a-zA-Z0-9]*[a-zA-Z0-9].{0,5}$", t) and len(t) <= 80:
        return True
    return False

def lines_from_page(page: fitz.Page) -> List[Dict[str, Any]]:
    """Return lines with text and bbox using PyMuPDF 'dict' structure."""
    d = page.get_text("dict")
    lines = []
    for b in d.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            bbox = l.get("bbox")
            text = "".join(span.get("text","") for span in l.get("spans", []))
            if text.strip():
                lines.append({"text": text, "bbox": [float(b) for b in bbox]})
    return lines

def crop_bbox_pad(rect: fitz.Rect, pad_px: int, page: fitz.Page, dpi=DPI_LAYOUT) -> fitz.Rect:
    # Convert pixel pad to points pad approximately via DPI
    pad_pt = pad_px * 72.0 / dpi
    r = fitz.Rect(rect.x0 - pad_pt, rect.y0 - pad_pt, rect.x1 + pad_pt, rect.y1 + pad_pt)
    return r & page.rect  # clip to page

def save_rect_as_pdf(page: fitz.Page, rect: fitz.Rect, out_pdf: Path, dpi=DPI_LAYOUT):
    """Render a rectangle region to PNG, then embed into a single-page PDF for Nougat."""
    ensure_dir(out_pdf.parent)
    # render crop to PNG
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    # save temp PNG
    with tempfile.TemporaryDirectory() as td:
        png_path = Path(td) / "crop.png"
        pix.save(png_path.as_posix())
        # write a single-page PDF with the image filling the page
        doc = fitz.open()
        # set page size equal to image size in points
        w_pt = pix.width * 72.0 / dpi
        h_pt = pix.height * 72.0 / dpi
        pg = doc.new_page(width=w_pt, height=h_pt)
        pg.insert_image(fitz.Rect(0,0,w_pt,h_pt), filename=png_path.as_posix())
        doc.save(out_pdf.as_posix())
        doc.close()

def nougat_on_pdf(pdf_region: Path) -> Optional[str]:
    """Run Nougat CLI on a tiny single-page PDF crop; return LaTeX (best-effort)."""
    if not shutil.which("nougat"):
        return None
    with tempfile.TemporaryDirectory() as td:
        out_md = Path(td) / "o.md"
        run(["nougat", pdf_region.as_posix(), "--out", out_md.as_posix()], check=False)
        if not out_md.exists():
            return None
        txt = out_md.read_text(encoding="utf-8", errors="ignore")
        # pull $$...$$ or \[...\] blocks; if none, return whole (trimmed) line
        m = re.findall(r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])", txt)
        if m:
            # strip $$ or \[ \]
            raw = m[0].strip()
            if raw.startswith("$$") and raw.endswith("$$"):
                return raw[2:-2].strip()
            if raw.startswith(r"\[") and raw.endswith(r"\]"):
                return raw[2:-2].strip()
            return raw
        return txt.strip() or None

# ===========================
# GROBID
# ===========================
def grobid_process(pdf_path: Path, grobid_url=GROBID_URL) -> Optional[str]:
    endpoint = f"{grobid_url}/api/processFulltextDocument"
    try:
        with open(pdf_path, "rb") as fh:
            r = requests.post(endpoint, files={"input": fh},
                              data={"consolidateCitations":"1","includeRawCitations":"1","includeRawAffiliations":"1"},
                              timeout=180)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except Exception:
        pass
    return None

def parse_grobid_tei(tei_xml: str) -> Dict[str, Any]:
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = etree.fromstring(tei_xml.encode("utf-8"))
    out = {"title": None, "abstract": None, "authors": [], "sections": [], "bibliography": []}
    title = root.xpath("//tei:fileDesc/tei:titleStmt/tei:title/text()", namespaces=ns)
    out["title"] = title[0].strip() if title else None
    abs_txt = root.xpath("//tei:profileDesc/tei:abstract//text()", namespaces=ns)
    out["abstract"] = " ".join(t.strip() for t in abs_txt if t.strip()) or None
    for pers in root.xpath("//tei:sourceDesc//tei:author", namespaces=ns):
        name = "".join(pers.xpath(".//tei:persName//text()", namespaces=ns)).strip()
        aff = " ".join(pers.xpath(".//tei:affiliation//text()", namespaces=ns)).strip() or None
        email = (pers.xpath(".//tei:email/text()", namespaces=ns) or [None])[0]
        out["authors"].append({"name": name or None, "affiliation": aff, "email": email})
    for div in root.xpath("//tei:body//tei:div", namespaces=ns):
        head = " ".join(div.xpath("./tei:head//text()", namespaces=ns)).strip() or None
        txt  = " ".join(div.xpath(".//tei:p//text()", namespaces=ns)).strip() or None
        if head or txt:
            out["sections"].append({"heading": head, "text": txt})
    for bibl in root.xpath("//tei:listBibl/tei:biblStruct", namespaces=ns):
        title = " ".join(bibl.xpath(".//tei:title//text()", namespaces=ns)).strip() or None
        year  = (bibl.xpath(".//tei:date/@when", namespaces=ns) or [""])[0][:4] or None
        doi   = (bibl.xpath(".//tei:idno[@type='DOI']/text()", namespaces=ns) or [None])[0]
        authors = [" ".join(a.xpath(".//text()", namespaces=ns)).strip()
                   for a in bibl.xpath(".//tei:author", namespaces=ns)]
        out["bibliography"].append({"title": title, "authors": authors, "year": year, "doi": doi})
    return out

# ===========================
# Core extraction per PDF
# ===========================
class Extractor:
    def __init__(self, pdf_path: Path, out_root: Path):
        self.pdf_path = pdf_path
        self.out_root = out_root / sanitize(pdf_path.stem)
        if self.out_root.exists():
            shutil.rmtree(self.out_root)
        ensure_dir(self.out_root)
        self.layout_model = LayoutModel()

    def extract(self) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            pdf_parse = ocr_if_needed(self.pdf_path, tmpdir)
            doc = fitz.open(pdf_parse.as_posix())
            n_pages = len(doc)

            # GROBID
            tei = grobid_process(pdf_parse)
            grobid = parse_grobid_tei(tei) if tei else None
            if tei:
                (self.out_root / "metadata_grobid.xml").write_text(tei, encoding="utf-8")

            pages_manifest = []
            total_chars = total_images = total_tables = 0

            for pno in range(n_pages):
                page = doc[pno]
                page_out = {
                    "page_index": pno,
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "blocks": [],
                    "layout_regions": [],
                    "text_reading_order": "",
                    "tables_csv": [],
                    "images": [],
                    "figures": [],
                    "equations": [],   # list of {bbox, latex, crop_path}
                    "links": [],
                }

                # Blocks
                for b in page.get_text("blocks"):
                    x0,y0,x1,y1,txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), b[4] or ""
                    btype = int(b[6]) if len(b) > 6 else 0
                    page_out["blocks"].append({"bbox":[x0,y0,x1,y1], "type":btype, "text":txt.strip()})

                # Links
                for l in page.get_links():
                    r = l.get("from", fitz.Rect())
                    page_out["links"].append({"from":[float(r.x0), float(r.y0), float(r.x1), float(r.y1)],
                                              "uri": l.get("uri"), "kind": l.get("kind")})

                # Embedded images
                img_dir = self.out_root / "images" / f"page_{pno:04d}"
                imgs = extract_embedded_images(doc, pno, img_dir)
                total_images += len([i for i in imgs if "file" in i])
                page_out["images"] = imgs

                # Raster for layout
                zoom = DPI_LAYOUT / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                dets = self.layout_model.detect(img)
                page_out["layout_regions"] = dets

                # Reading order boxes (Title/Text/List)
                text_boxes_pdf = []
                for d in dets:
                    if d["label"] in ("Title", "Text", "List"):
                        text_boxes_pdf.append(img_bbox_to_pdf_bbox(d["bbox_img"], img.shape[1], img.shape[0], page.rect))
                text_ro = reading_order_from_boxes(page, text_boxes_pdf)
                page_out["text_reading_order"] = text_ro
                total_chars += len(text_ro)

                # Tables by region
                table_dir = self.out_root / "tables"
                ensure_dir(table_dir)
                tcount = 0
                for d in dets:
                    if d["label"] == "Table":
                        pdf_bbox = img_bbox_to_pdf_bbox(d["bbox_img"], img.shape[1], img.shape[0], page.rect)
                        dfs = extract_tables_region(pdf_parse, pno+1, pdf_bbox)
                        for df in dfs:
                            csv_name = f"page{pno:04d}_table{tcount:03d}.csv"
                            df.to_csv((table_dir / csv_name), index=False, header=False)
                            page_out["tables_csv"].append(csv_name)
                            tcount += 1
                total_tables += len(page_out["tables_csv"])

                # Figures crops
                fig_dir = self.out_root / "figures" / f"page_{pno:04d}"
                fcount = 0
                for d in dets:
                    if d["label"] == "Figure":
                        pdf_bbox = img_bbox_to_pdf_bbox(d["bbox_img"], img.shape[1], img.shape[0], page.rect)
                        outp = fig_dir / f"figure{fcount:03d}.png"
                        save_crop(page, bbox_to_rect(pdf_bbox), outp, dpi=DPI_LAYOUT)
                        page_out["figures"].append(outp.relative_to(self.out_root).as_posix())
                        fcount += 1

                # -------- Equations per page --------
                # Get per-line bboxes
                lines = lines_from_page(page)
                eq_dir = self.out_root / "equations" / f"page_{pno:04d}"
                ecount = 0
                for L in lines:
                    if ecount >= N_PAGE_EQUATION_LIMIT:
                        break
                    if not is_equation_like(L["text"]):
                        continue
                    rect = bbox_to_rect(L["bbox"])
                    rect = crop_bbox_pad(rect, EQ_BBOX_PAD_PX, page)
                    # Save region as a tiny PDF and run Nougat
                    with tempfile.TemporaryDirectory() as td2:
                        mini_pdf = Path(td2) / "crop.pdf"
                        save_rect_as_pdf(page, rect, mini_pdf, dpi=DPI_LAYOUT)
                        latex = nougat_on_pdf(mini_pdf) or ""
                    crop_png = eq_dir / f"eq{ecount:03d}.png"
                    save_crop(page, rect, crop_png, dpi=DPI_LAYOUT)
                    page_out["equations"].append({
                        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "latex": latex,
                        "crop": crop_png.relative_to(self.out_root).as_posix(),
                        "text_hint": L["text"].strip()
                    })
                    ecount += 1

                # Save per-page JSON
                save_json(page_out, self.out_root / "pages" / f"page_{pno:04d}.json")
                pages_manifest.append(page_out)

            # Top-level outputs
            summary = {
                "file": self.pdf_path.name,
                "pages": len(pages_manifest),
                "total_characters": total_chars,
                "total_images": total_images,
                "total_tables": total_tables,
                "dpi": DPI_LAYOUT,
                "ocr_applied": pdf_parse.name.endswith(".ocred.pdf"),
                "metadata_pdf": fitz.open(pdf_parse.as_posix()).metadata or {},
                "grobid": grobid
            }
            save_json(summary, self.out_root / "summary.json")
            save_json({"pages": pages_manifest}, self.out_root / "all_pages.json")

            # Zip
            zip_path = self.out_root.with_suffix(".zip")
            if zip_path.exists():
                zip_path.unlink()
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for root, _, files in os.walk(self.out_root):
                    for f in files:
                        fp = Path(root) / f
                        z.write(fp, arcname=fp.relative_to(self.out_root.parent))
            return {"out_dir": self.out_root.as_posix(), "zip": zip_path.as_posix(), "summary": summary}

# ===========================
# Batch over a directory
# ===========================
def main():
    ensure_dir(OUTPUT_DIR)
    pdfs = sorted([p for p in INPUT_DIR.glob("**/*.pdf") if p.is_file()])
    if not pdfs:
        print(f"No PDFs found under {INPUT_DIR}")
        return
    rows = []
    for p in pdfs:
        print(f"\n=== Extracting: {p.name} ===")
        try:
            res = Extractor(p, OUTPUT_DIR).extract()
            s = res["summary"]
            rows.append({
                "file": s["file"],
                "pages": s["pages"],
                "chars": s["total_characters"],
                "images": s["total_images"],
                "tables": s["total_tables"],
                "zip": res["zip"]
            })
        except Exception as e:
            rows.append({"file": p.name, "error": str(e)})
            print(f"[ERROR] {p.name}: {e}")

    # Write a small CSV manifest for the batch
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "batch_manifest.csv", index=False)
    print("\nBatch complete. Manifest:", (OUTPUT_DIR / "batch_manifest.csv").as_posix())

if __name__ == "__main__":
    main()
