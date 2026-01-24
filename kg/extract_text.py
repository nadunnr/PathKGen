import pdfplumber
import json
from collections import defaultdict

PDF_PATH = "docs/who-breast.pdf"

def extract_lines(page):
    words = page.extract_words(
        extra_attrs=["fontname", "size", "top"]
    )
    
    if not words:
        return []
    
    # Detect column boundaries by clustering x0 positions
    # Collect all unique starting x-positions
    x_starts = [w["x0"] for w in words]
    
    # Simple clustering: find gaps in x-positions to identify columns
    # Sort unique x positions and find large gaps
    unique_x = sorted(set(x_starts))
    
    # Calculate gaps between consecutive x positions
    gaps = []
    for i in range(len(unique_x) - 1):
        gap = unique_x[i + 1] - unique_x[i]
        gaps.append((gap, unique_x[i], unique_x[i + 1]))
    
    # Find the 2 largest gaps (for 3 columns, there should be 2 major gaps)
    gaps.sort(reverse=True)
    
    # If we have enough gaps, use them to define column boundaries
    if len(gaps) >= 2:
        # Get the positions of the two largest gaps
        gap1_mid = (gaps[0][1] + gaps[0][2]) / 2
        gap2_mid = (gaps[1][1] + gaps[1][2]) / 2
        
        # Sort boundaries left to right
        boundaries = sorted([gap1_mid, gap2_mid])
        col1_max = boundaries[0]
        col2_max = boundaries[1]
    else:
        # Fallback to equal division if gap detection fails
        page_width = page.width
        col1_max = page_width / 3
        col2_max = 2 * page_width / 3
    
    # Assign each word to a column based on its x0 position
    for w in words:
        if w["x0"] < col1_max:
            w["column"] = 0
        elif w["x0"] < col2_max:
            w["column"] = 1
        else:
            w["column"] = 2
    
    # Group words by column and y-position
    column_lines = defaultdict(lambda: defaultdict(list))
    for w in words:
        col = w["column"]
        y_key = round(w["top"], 1)
        column_lines[col][y_key].append(w)
    
    # Process each column from top to bottom, then columns left to right
    merged = []
    for col in sorted(column_lines.keys()):
        for y in sorted(column_lines[col].keys()):
            row = sorted(column_lines[col][y], key=lambda x: x["x0"])
            text = " ".join(w["text"] for w in row)
            avg_size = sum(w["size"] for w in row) / len(row)
            fonts = {w["fontname"] for w in row}
            merged.append({
                "text": text.strip(),
                "font_size": avg_size,
                "fonts": fonts,
                "column": col
            })
    
    return merged


def is_heading(line):
    return (
        line["font_size"] >= 13 and
        len(line["text"]) < 120 and
        not line["text"].endswith(".")
    )


chapter = None
cancer = None
subsection = None
buffer = []

records = []

def flush():
    global buffer
    if chapter and cancer and subsection and buffer:
        records.append({
            "chapter": chapter,
            "cancer": cancer,
            "subsection": subsection,
            "content": " ".join(buffer)
        })
    buffer = []


with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        lines = extract_lines(page)

        for line in lines:
            text = line["text"]

            if text.startswith("CHAPTER"):
                flush()
                chapter = text
                cancer = None
                subsection = None

            elif is_heading(line) and chapter and not cancer:
                flush()
                cancer = text

            elif is_heading(line) and cancer:
                flush()
                subsection = text

            else:
                if subsection:
                    buffer.append(text)

flush()

print(f"Extracted {len(records)} subsections")

with open("docs/extracted_text/who-breast.json", "w") as f:
    json.dump(records, f, indent=2)
