# embed_campbell.py
# ─────────────────────────────────────────────────────────────────────────────
# ONE-TIME setup script. Run this ONCE to:
#   1. Extract text chunks from Campbell Biology 12e PDF
#   2. Extract and save meaningful figures to backend/images/
#   3. Build images/manifest.json mapping Figure IDs → saved file paths
#   4. Embed chunks and upsert to Pinecone (Cohere namespace and/or Llama namespace)
#
# Run again only if you:
#   - Add new pages (use --pages-from N to resume)
#   - Switch embedding models (use --mode to target a specific namespace)
#   - Want to add a second book (change --namespace and --doc-id)
#
# ─── USAGE ───────────────────────────────────────────────────────────────────
# Full pipeline (both Cohere and Llama namespaces):
#   python embed_campbell.py --pdf Campbell_Biology_12e.pdf --mode both
#
# Cohere only:
#   python embed_campbell.py --pdf Campbell_Biology_12e.pdf --mode cohere
#
# Llama/Pinecone Inference only:
#   python embed_campbell.py --pdf Campbell_Biology_12e.pdf --mode llama
#
# Resume from page 500 (skip already-processed pages):
#   python embed_campbell.py --pdf Campbell_Biology_12e.pdf --mode both --pages-from 500
#
# Extract images only (no embedding):
#   python embed_campbell.py --pdf Campbell_Biology_12e.pdf --images-only
#
# ─── PREREQUISITES ───────────────────────────────────────────────────────────
# pip install pypdf pdfplumber cohere pinecone-client Pillow python-dotenv
# Set PINECONE_API_KEY, COHERE_API_KEY in your .env file.
# Create the Pinecone index first — see WORKFLOW.md.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── CAMPBELL BIOLOGY 12e STRUCTURE ───────────────────────────────────────────
# Hard-coded chapter/unit map derived from the Table of Contents (page 2).
# Used to tag every chunk with its unit and chapter title without
# relying solely on regex detection.
#
# Key: chapter number → {unit, title, start_page (approximate)}
CAMPBELL_12E_STRUCTURE = {
    1:  {"unit": "Unit 1 The Chemistry of Life",    "title": "Evolution, the Themes of Biology, and Scientific Inquiry", "start_pg": 52},
    2:  {"unit": "Unit 1 The Chemistry of Life",    "title": "The Chemical Context of Life",                  "start_pg": 78},
    3:  {"unit": "Unit 1 The Chemistry of Life",    "title": "Water and Life",                                 "start_pg": 94},
    4:  {"unit": "Unit 1 The Chemistry of Life",    "title": "Carbon and the Molecular Diversity of Life",     "start_pg": 106},
    5:  {"unit": "Unit 1 The Chemistry of Life",    "title": "The Structure and Function of Large Biological Molecules", "start_pg": 116},
    6:  {"unit": "Unit 2 The Cell",                 "title": "A Tour of the Cell",                             "start_pg": 143},
    7:  {"unit": "Unit 2 The Cell",                 "title": "Membrane Structure and Function",                "start_pg": 176},
    8:  {"unit": "Unit 2 The Cell",                 "title": "An Introduction to Metabolism",                  "start_pg": 193},
    9:  {"unit": "Unit 2 The Cell",                 "title": "Cellular Respiration and Fermentation",          "start_pg": 214},
    10: {"unit": "Unit 2 The Cell",                 "title": "Photosynthesis",                                 "start_pg": 237},
    11: {"unit": "Unit 2 The Cell",                 "title": "Cell Communication",                             "start_pg": 262},
    12: {"unit": "Unit 2 The Cell",                 "title": "The Cell Cycle",                                 "start_pg": 284},
    13: {"unit": "Unit 3 Genetics",                 "title": "Meiosis and Sexual Life Cycles",                 "start_pg": 304},
    14: {"unit": "Unit 3 Genetics",                 "title": "Mendel and the Gene Idea",                       "start_pg": 319},
    15: {"unit": "Unit 3 Genetics",                 "title": "The Chromosomal Basis of Inheritance",           "start_pg": 344},
    16: {"unit": "Unit 3 Genetics",                 "title": "The Molecular Basis of Inheritance",             "start_pg": 364},
    17: {"unit": "Unit 3 Genetics",                 "title": "Gene Expression: From Gene to Protein",          "start_pg": 385},
    18: {"unit": "Unit 3 Genetics",                 "title": "Regulation of Gene Expression",                  "start_pg": 415},
    19: {"unit": "Unit 3 Genetics",                 "title": "Viruses",                                        "start_pg": 448},
    20: {"unit": "Unit 3 Genetics",                 "title": "DNA Tools and Biotechnology",                    "start_pg": 465},
    21: {"unit": "Unit 3 Genetics",                 "title": "Genomes and Their Evolution",                    "start_pg": 492},
    22: {"unit": "Unit 4 Mechanisms of Evolution",  "title": "Descent with Modification: A Darwinian View",   "start_pg": 518},
    23: {"unit": "Unit 4 Mechanisms of Evolution",  "title": "The Evolution of Populations",                  "start_pg": 536},
    24: {"unit": "Unit 4 Mechanisms of Evolution",  "title": "The Origin of Species",                         "start_pg": 556},
    25: {"unit": "Unit 4 Mechanisms of Evolution",  "title": "The History of Life on Earth",                  "start_pg": 575},
    26: {"unit": "Unit 5 Evolutionary History of Biological Diversity", "title": "Phylogeny and the Tree of Life", "start_pg": 603},
    27: {"unit": "Unit 5 Evolutionary History of Biological Diversity", "title": "Bacteria and Archaea",      "start_pg": 623},
    28: {"unit": "Unit 5 Evolutionary History of Biological Diversity", "title": "Protists",                  "start_pg": 643},
    29: {"unit": "Unit 5 Evolutionary History of Biological Diversity", "title": "Plant Diversity I",         "start_pg": 668},
    30: {"unit": "Unit 5 Evolutionary History of Biological Diversity", "title": "Plant Diversity II",        "start_pg": 686},
    31: {"unit": "Unit 5 Evolutionary History of Biological Diversity", "title": "Fungi",                     "start_pg": 704},
    32: {"unit": "Unit 5 Evolutionary History of Biological Diversity", "title": "An Overview of Animal Diversity", "start_pg": 723},
    33: {"unit": "Unit 5 Evolutionary History of Biological Diversity", "title": "An Introduction to Invertebrates", "start_pg": 736},
    34: {"unit": "Unit 5 Evolutionary History of Biological Diversity", "title": "The Origin and Evolution of Vertebrates", "start_pg": 768},
    35: {"unit": "Unit 6 Plant Form and Function",  "title": "Vascular Plant Structure, Growth, and Development", "start_pg": 808},
    36: {"unit": "Unit 6 Plant Form and Function",  "title": "Resource Acquisition and Transport in Vascular Plants", "start_pg": 834},
    37: {"unit": "Unit 6 Plant Form and Function",  "title": "Soil and Plant Nutrition",                      "start_pg": 855},
    38: {"unit": "Unit 6 Plant Form and Function",  "title": "Angiosperm Reproduction and Biotechnology",     "start_pg": 872},
    39: {"unit": "Unit 6 Plant Form and Function",  "title": "Plant Responses to Internal and External Signals", "start_pg": 892},
    40: {"unit": "Unit 7 Animal Form and Function", "title": "Basic Principles of Animal Form and Function",  "start_pg": 923},
    41: {"unit": "Unit 7 Animal Form and Function", "title": "Animal Nutrition",                               "start_pg": 948},
    42: {"unit": "Unit 7 Animal Form and Function", "title": "Circulation and Gas Exchange",                   "start_pg": 971},
    43: {"unit": "Unit 7 Animal Form and Function", "title": "The Immune System",                              "start_pg": 1002},
    44: {"unit": "Unit 7 Animal Form and Function", "title": "Osmoregulation and Excretion",                   "start_pg": 1027},
    45: {"unit": "Unit 7 Animal Form and Function", "title": "Hormones and the Endocrine System",              "start_pg": 1049},
    46: {"unit": "Unit 7 Animal Form and Function", "title": "Animal Reproduction",                            "start_pg": 1069},
    47: {"unit": "Unit 7 Animal Form and Function", "title": "Animal Development",                             "start_pg": 1093},
    48: {"unit": "Unit 7 Animal Form and Function", "title": "Neurons, Synapses, and Signaling",               "start_pg": 1117},
    49: {"unit": "Unit 7 Animal Form and Function", "title": "Nervous Systems",                                "start_pg": 1135},
    50: {"unit": "Unit 7 Animal Form and Function", "title": "Sensory and Motor Mechanisms",                   "start_pg": 1157},
    51: {"unit": "Unit 7 Animal Form and Function", "title": "Animal Behavior",                                "start_pg": 1189},
    52: {"unit": "Unit 8 Ecology",                  "title": "An Introduction to Ecology and the Biosphere",  "start_pg": 1214},
    53: {"unit": "Unit 8 Ecology",                  "title": "Population Ecology",                             "start_pg": 1240},
    54: {"unit": "Unit 8 Ecology",                  "title": "Community Ecology",                              "start_pg": 1264},
    55: {"unit": "Unit 8 Ecology",                  "title": "Ecosystems and Restoration Ecology",             "start_pg": 1288},
    56: {"unit": "Unit 8 Ecology",                  "title": "Conservation Biology and Global Change",         "start_pg": 1310},
}

# Build a reverse lookup: approximate_page → chapter_number
# Used to assign chapter metadata to pages without a visible chapter heading
_PAGE_TO_CHAPTER_APPROX: dict[int, int] = {}
for _ch, _info in CAMPBELL_12E_STRUCTURE.items():
    _PAGE_TO_CHAPTER_APPROX[_info["start_pg"]] = _ch

def _page_to_chapter(page_num: int) -> int:
    """Return the most likely chapter for a given PDF page number."""
    best_ch = 1
    for start_pg, ch in sorted(_PAGE_TO_CHAPTER_APPROX.items()):
        if page_num >= start_pg:
            best_ch = ch
    return best_ch


# ── REGEX PATTERNS ────────────────────────────────────────────────────────────
_CHAPTER_RE = re.compile(r'\bCHAPTER\s+(\d{1,2})\b', re.IGNORECASE)
_CONCEPT_RE = re.compile(r'CONCEPT\s+(\d{1,2}\.\d{1,2})\s+(.{5,100})', re.IGNORECASE)
_SECTION_RE = re.compile(r'^(\d{1,2}\.\d{1,2})\s+(.{5,80})', re.MULTILINE)
_FIGURE_RE  = re.compile(r'(?:▲\s*)?Figure\s+(\d{1,2}\.\d{1,2}[a-z]?)\s*[-–]?\s*(.{0,120})', re.IGNORECASE)


# ── IMAGE EXTRACTION ──────────────────────────────────────────────────────────

def extract_and_save_images(pdf_path: str, images_dir: Path, min_pixels: int, min_dim: int) -> dict:
    """
    Extract all meaningful embedded images from the PDF and save them to disk.
    Organises images into per-chapter subdirectories.

    Returns:
        manifest: dict mapping figure_id → list of saved relative file paths
                  e.g. {"1.1": ["images/ch01/img_p52_i0.jpg"], ...}

    Also saves manifest.json to images_dir.

    Strategy:
    - Use pdfplumber to get page images with position data
    - Use pypdf to extract raw image bytes (XObject images)
    - Filter out tiny images (decorative bullets, icons, backgrounds)
    - Match saved images to figure references found in page text
    - Name files: img_p{page}_{xref_or_index}.jpg
    """
    try:
        from pypdf import PdfReader
        from pypdf.generic import NameObject
    except ImportError:
        logger.error("pypdf not installed: pip install pypdf")
        sys.exit(1)

    try:
        from PIL import Image as PILImage
        import io
    except ImportError:
        logger.error("Pillow not installed: pip install Pillow")
        sys.exit(1)

    manifest: dict[str, list[str]] = {}
    figure_to_page: dict[str, int] = {}   # fig_id → page number

    logger.info(f"Opening PDF for image extraction: {pdf_path}")
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # First pass: map figure references to page numbers from text
    import pdfplumber
    logger.info("Pass 1: scanning text for Figure references...")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if i % 100 == 0:
                logger.info(f"  Text scan: page {i}/{total_pages}")
            text = page.extract_text() or ""
            for m in _FIGURE_RE.finditer(text):
                fig_id = m.group(1)
                if fig_id not in figure_to_page:
                    figure_to_page[fig_id] = i

    logger.info(f"Found {len(figure_to_page)} unique Figure references in text")

    # Second pass: extract images page by page
    logger.info("Pass 2: extracting and saving images...")
    saved_count = 0
    page_img_map: dict[int, list[str]] = {}   # page_num → list of saved paths

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if page_num % 100 == 0:
            logger.info(f"  Image extraction: page {page_num}/{total_pages} | saved so far: {saved_count}")

        page = reader.pages[page_idx]
        ch_num  = _page_to_chapter(page_num)
        ch_dir  = images_dir / f"ch{ch_num:02d}"
        ch_dir.mkdir(exist_ok=True)

        # Extract XObject images from the page
        try:
            resources = page.get("/Resources", {})
            xobjects  = resources.get("/XObject", {})
        except Exception:
            continue

        img_index = 0
        for obj_name, obj_ref in xobjects.items():
            try:
                obj = obj_ref.get_object() if hasattr(obj_ref, "get_object") else obj_ref
                if obj.get("/Subtype") != "/Image":
                    continue

                width  = int(obj.get("/Width",  0))
                height = int(obj.get("/Height", 0))
                area   = width * height

                # Skip tiny images (icons, bullets, decorations)
                if area < min_pixels or width < min_dim or height < min_dim:
                    continue

                # Extract raw bytes
                data = obj.get_data() if hasattr(obj, "get_data") else None
                if data is None:
                    continue

                # Determine original format
                color_space = str(obj.get("/ColorSpace", ""))
                if "/DCTDecode" in str(obj.get("/Filter", "")):
                    ext = "jpg"
                    raw_bytes = data
                else:
                    # Convert raw bytes to JPEG via PIL for consistent storage
                    try:
                        if "Gray" in color_space:
                            mode = "L"
                        elif "CMYK" in color_space:
                            mode = "CMYK"
                        else:
                            mode  = "RGB"
                            n_comp = 3
                        # Try to construct PIL image from raw pixel data
                        img = PILImage.frombytes(mode, (width, height), data)
                        buf = io.BytesIO()
                        img.convert("RGB").save(buf, format="JPEG", quality=85)
                        raw_bytes = buf.getvalue()
                        ext = "jpg"
                    except Exception:
                        continue

                filename = f"img_p{page_num}_i{img_index}.{ext}"
                filepath = ch_dir / filename
                filepath.write_bytes(raw_bytes)

                rel_path = f"ch{ch_num:02d}/{filename}"
                page_img_map.setdefault(page_num, []).append(rel_path)

                img_index += 1
                saved_count += 1

            except Exception as e:
                logger.debug(f"  Skip image on p.{page_num}: {e}")
                continue

    logger.info(f"Saved {saved_count} images across {len(page_img_map)} pages")

    # Third pass: build manifest — map figure IDs to their images
    # Strategy: the page where a Figure reference appears should have the image
    for fig_id, fig_page in figure_to_page.items():
        imgs = page_img_map.get(fig_page, [])
        # Also check adjacent pages (figure may span to next page)
        if not imgs:
            imgs = page_img_map.get(fig_page + 1, [])
        if imgs:
            manifest[fig_id] = imgs

    # Save manifest to disk
    manifest_path = images_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved: {manifest_path} ({len(manifest)} figure entries)")
    return manifest


# ── TEXT CHUNKING ─────────────────────────────────────────────────────────────

def extract_and_chunk(pdf_path: str, chunk_size: int, overlap: int,
                      doc_id: str, pages_from: int, pages_to: int,
                      manifest: dict) -> list[dict]:
    """
    Extract text from the PDF, chunk it with overlap, and attach rich metadata.

    Each chunk dict contains:
        id          : unique Pinecone vector ID
        text        : the chunk content (800 chars ≈ 200 tokens)
        title       : "Campbell Biology"
        doc_id      : "campbell-biology-12e"
        unit        : e.g. "Unit 2 The Cell"
        chapter     : e.g. "Chapter 7"
        chapter_title: e.g. "Membrane Structure and Function"
        concept     : e.g. "Concept 7.2"
        concept_title: the concept's heading text
        section     : e.g. "7.2 Membrane proteins..."
        page        : PDF page number (string)
        figure_refs : list of figure IDs mentioned in this chunk ["7.1", "7.2"]
        image_files : list of relative image paths matching those figure refs
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed: pip install pdfplumber")
        sys.exit(1)

    chunks     = []
    chunk_idx  = 0
    current_chapter = 1
    current_concept = ""
    current_concept_title = ""
    current_section = ""

    logger.info(f"Chunking pages {pages_from}–{pages_to}...")

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        end   = min(pages_to, total)

        for page_num in range(pages_from, end + 1):
            if page_num % 100 == 0:
                logger.info(f"  Chunking: page {page_num}/{end}")

            page = pdf.pages[page_num - 1]
            text = page.extract_text() or ""
            if not text.strip():
                continue

            # ── Update running context from page text ──────────────────────

            ch_m = _CHAPTER_RE.search(text)
            if ch_m:
                detected = int(ch_m.group(1))
                if 1 <= detected <= 56:
                    current_chapter = detected

            # Use the hard-coded structure map as the ground truth
            ch_num = _page_to_chapter(page_num)
            if ch_m and 1 <= int(ch_m.group(1)) <= 56:
                ch_num = int(ch_m.group(1))

            ch_info = CAMPBELL_12E_STRUCTURE.get(ch_num, {})
            unit          = ch_info.get("unit",  "")
            chapter_title = ch_info.get("title", "")
            chapter_label = f"Chapter {ch_num}"

            concept_m = _CONCEPT_RE.search(text)
            if concept_m:
                current_concept       = f"Concept {concept_m.group(1)}"
                current_concept_title = concept_m.group(2).strip()[:80]

            section_m = _SECTION_RE.search(text)
            if section_m:
                current_section = f"{section_m.group(1)} {section_m.group(2).strip()[:60]}"

            # ── Extract figure references from this page ────────────────────
            page_fig_refs = []
            for m in _FIGURE_RE.finditer(text):
                fig_id = m.group(1)
                if fig_id not in page_fig_refs:
                    page_fig_refs.append(fig_id)

            # ── Resolve image files for figure refs ────────────────────────
            page_img_files = []
            for fig_id in page_fig_refs:
                for img_path in manifest.get(fig_id, []):
                    if img_path not in page_img_files:
                        page_img_files.append(img_path)

            # ── Slide chunking window across page text ─────────────────────
            start = 0
            while start < len(text):
                end_char = start + chunk_size
                chunk    = text[start:end_char].strip()

                if len(chunk) > 60:   # skip tiny fragments
                    # Identify which figures this specific chunk mentions
                    chunk_figs = []
                    for m in _FIGURE_RE.finditer(chunk):
                        fig_id = m.group(1)
                        if fig_id not in chunk_figs:
                            chunk_figs.append(fig_id)

                    chunk_imgs = []
                    for fig_id in chunk_figs:
                        for img_path in manifest.get(fig_id, []):
                            if img_path not in chunk_imgs:
                                chunk_imgs.append(img_path)

                    chunks.append({
                        "id"            : f"campbell12-p{page_num}-c{chunk_idx}",
                        "text"          : chunk,
                        "title"         : "Campbell Biology",
                        "doc_id"        : doc_id,
                        "unit"          : unit,
                        "chapter"       : chapter_label,
                        "chapter_title" : chapter_title,
                        "concept"       : current_concept,
                        "concept_title" : current_concept_title,
                        "section"       : current_section,
                        "page"          : str(page_num),
                        "figure_refs"   : chunk_figs,
                        "image_files"   : chunk_imgs,
                    })
                    chunk_idx += 1

                start += chunk_size - overlap

    logger.info(f"Created {len(chunks)} chunks from pages {pages_from}–{end}")
    return chunks


# ── COHERE UPSERT ─────────────────────────────────────────────────────────────

def embed_and_upsert_cohere(chunks: list[dict], namespace: str, batch_size: int):
    """
    Embed chunks with Cohere embed-english-v3.0 and upsert to Pinecone.
    Uses batched embedding (up to 96 per Cohere call) and batched upsert.
    """
    from dotenv import load_dotenv
    load_dotenv()
    from config import COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
    import cohere
    from pinecone import Pinecone

    co    = cohere.Client(api_key=COHERE_API_KEY)
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    total = len(chunks)
    done  = 0

    logger.info(f"[COHERE] Embedding {total} chunks → '{PINECONE_INDEX_NAME}' / '{namespace}'")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        # input_type="search_document" tells Cohere these are documents, not queries
        resp    = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")
        vectors = resp.embeddings

        pinecone_vecs = []
        for chunk, vec in zip(batch, vectors):
            pinecone_vecs.append({
                "id"      : chunk["id"],
                "values"  : vec,
                "metadata": {
                    k: v for k, v in chunk.items()
                    if k not in ("id",)   # id is the Pinecone vector ID, not metadata
                },
            })

        index.upsert(vectors=pinecone_vecs, namespace=namespace)
        done += len(batch)
        logger.info(f"  [COHERE] {done}/{total} ({100*done//total}%)")

        # Rate-limit safety (free tier: 100 API calls/min for Cohere)
        time.sleep(0.6)

    logger.info(f"[COHERE] Done — {done} chunks stored in namespace '{namespace}'")


# ── LLAMA / PINECONE INFERENCE UPSERT ────────────────────────────────────────

def upsert_llama_inference(chunks: list[dict], namespace: str, batch_size: int):
    """
    Upsert text records for Pinecone Inference embedding (Llama Text Embed v2).

    Pinecone generates the embedding automatically from the 'text' field.
    We use index.upsert_records() instead of index.upsert() — no vectors
    need to be generated locally.

    Requirements:
    - The Pinecone index must have been created with an integrated embedding model.
    - pinecone-client >= 4.0 (for upsert_records support).
    """
    from dotenv import load_dotenv
    load_dotenv()
    from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
    from pinecone import Pinecone

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    total = len(chunks)
    done  = 0

    logger.info(f"[LLAMA-PINECONE] Upserting {total} records → '{PINECONE_INDEX_NAME}' / '{namespace}'")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]

        # For Pinecone Inference, each record is a flat dict.
        # '_id' is the required primary key. All other fields become
        # metadata AND the 'text' field is what Pinecone embeds.
        records = []
        for chunk in batch:
            record = {"_id": chunk["id"]}
            # Copy all metadata fields; Pinecone embeds the 'text' field
            for k, v in chunk.items():
                if k == "id":
                    continue
                # Pinecone metadata: lists must be JSON-serialised strings
                if isinstance(v, list):
                    record[k] = json.dumps(v)
                else:
                    record[k] = v
            records.append(record)

        try:
            index.upsert_records(records=records, namespace=namespace)
        except AttributeError:
            logger.error(
                "index.upsert_records() not available. "
                "Upgrade pinecone-client: pip install --upgrade pinecone-client"
            )
            sys.exit(1)

        done += len(batch)
        logger.info(f"  [LLAMA] {done}/{total} ({100*done//total}%)")

        # Brief pause to respect Pinecone Inference rate limits
        time.sleep(0.3)

    logger.info(f"[LLAMA] Done — {done} records stored in namespace '{namespace}'")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract, chunk, and embed Campbell Biology 12e into Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pdf",        required=True, help="Path to Campbell_Biology_12e.pdf")
    parser.add_argument("--mode",       default="both",
                        choices=["cohere", "llama", "both"],
                        help="Which namespace to populate (default: both)")
    parser.add_argument("--doc-id",     default="campbell-biology-12e",
                        help="Unique document ID stored in metadata")
    parser.add_argument("--chunk-size", type=int, default=800,
                        help="Characters per chunk (default: 800 ≈ 200 tokens)")
    parser.add_argument("--overlap",    type=int, default=120,
                        help="Overlap characters between adjacent chunks (default: 120)")
    parser.add_argument("--batch",      type=int, default=50,
                        help="Vectors per Pinecone upsert call (default: 50)")
    parser.add_argument("--pages-from", type=int, default=1,
                        help="Start from this PDF page (for resuming, default: 1)")
    parser.add_argument("--pages-to",   type=int, default=1493,
                        help="End at this PDF page (default: 1493)")
    parser.add_argument("--images-only", action="store_true",
                        help="Extract images only — skip embedding")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image extraction — embed text only (faster)")
    args = parser.parse_args()

    # ── Validate PDF path
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    # ── Load environment
    from dotenv import load_dotenv
    load_dotenv()
    from config import (
        IMAGES_DIR, IMAGE_MANIFEST,
        PINECONE_NS_COHERE, PINECONE_NS_LLAMA,
        RETRIEVAL_CONFIG,
    )

    min_pixels = RETRIEVAL_CONFIG["min_image_pixels"]
    min_dim    = RETRIEVAL_CONFIG["min_image_dim"]

    # ── Step 1: Image extraction
    manifest = {}
    if not args.skip_images:
        logger.info("=" * 60)
        logger.info("STEP 1: Extracting and saving figures")
        logger.info("=" * 60)

        # Check for existing manifest to resume
        if IMAGE_MANIFEST.exists():
            logger.info(f"Found existing manifest at {IMAGE_MANIFEST} — loading...")
            with open(IMAGE_MANIFEST) as f:
                manifest = json.load(f)
            logger.info(f"Loaded {len(manifest)} existing figure entries")
        else:
            manifest = extract_and_save_images(
                pdf_path  = str(pdf_path),
                images_dir= IMAGES_DIR,
                min_pixels= min_pixels,
                min_dim   = min_dim,
            )

    if args.images_only:
        logger.info("Images-only mode complete.")
        sys.exit(0)

    # ── Step 2: Text chunking
    logger.info("=" * 60)
    logger.info("STEP 2: Chunking text")
    logger.info("=" * 60)

    chunks = extract_and_chunk(
        pdf_path  = str(pdf_path),
        chunk_size= args.chunk_size,
        overlap   = args.overlap,
        doc_id    = args.doc_id,
        pages_from= args.pages_from,
        pages_to  = args.pages_to,
        manifest  = manifest,
    )
    logger.info(f"Total chunks to embed: {len(chunks)}")

    # Count chunks with images
    with_imgs = sum(1 for c in chunks if c["image_files"])
    logger.info(f"Chunks with image references: {with_imgs} ({100*with_imgs//max(1,len(chunks))}%)")

    # ── Step 3: Embed and upsert
    if args.mode in ("cohere", "both"):
        logger.info("=" * 60)
        logger.info("STEP 3a: Cohere embedding → Pinecone")
        logger.info("=" * 60)
        embed_and_upsert_cohere(chunks, namespace=PINECONE_NS_COHERE, batch_size=args.batch)

    if args.mode in ("llama", "both"):
        logger.info("=" * 60)
        logger.info("STEP 3b: Pinecone Inference (Llama) → Pinecone")
        logger.info("=" * 60)
        upsert_llama_inference(chunks, namespace=PINECONE_NS_LLAMA, batch_size=args.batch)

    logger.info("=" * 60)
    logger.info("EMBEDDING COMPLETE")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  Images saved: see backend/images/")
    logger.info(f"  Manifest: {IMAGE_MANIFEST}")
    logger.info(f"  Namespaces populated: {args.mode}")
    logger.info("Next step: python api.py — then open frontend/index.html")
    logger.info("=" * 60)
