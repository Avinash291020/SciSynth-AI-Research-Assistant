import json
import re
from pathlib import Path

def extract_title(insights):
    for line in insights.split('\n'):
        if line.strip().startswith('Title:'):
            return line.replace('Title:', '').strip()
    for line in insights.split('\n'):
        if line.strip():
            return line.strip()
    return ""

def extract_authors(insights):
    # Look for lines after Title or lines with multiple commas (author list)
    lines = insights.split('\n')
    found_title = False
    for line in lines:
        if found_title and line.strip():
            # Heuristic: line with multiple commas and no numbers
            if line.count(',') >= 1 and not any(char.isdigit() for char in line):
                # Remove affiliations if present
                authors = re.split(r'\d|\(|\[', line)[0].strip()
                # Split by comma and strip
                return [a.strip() for a in authors.split(',') if a.strip()]
        if line.strip().startswith('Title:'):
            found_title = True
    # Fallback: look for 'by' line
    for line in lines:
        if line.lower().startswith('by '):
            return [a.strip() for a in line[3:].split(',') if a.strip()]
    return []

def extract_date(insights, processed_date):
    # Look for a date pattern in the text (YYYY or YYYY-MM-DD)
    date_pattern = r'(\d{4}-\d{2}-\d{2}|\d{4})'
    match = re.search(date_pattern, insights)
    if match:
        return match.group(1)
    # Fallback: use processed_date
    return processed_date[:10] if processed_date else ""

def extract_keywords(insights):
    if "Keywords:" in insights:
        kw_section = insights.split("Keywords:")[1].split("\n")[0]
        return [k.strip() for k in kw_section.split(",") if k.strip()]
    # Fallback: look for common keyword patterns
    patterns = [
        r'\bneural networks?\b', r'\bdeep learning\b', r'\bmachine learning\b', r'\bartificial intelligence\b',
        r'\bAI\b', r'\blanguage models?\b', r'\btransformers?\b', r'\breinforcement learning\b',
        r'\bevolutionary algorithms?\b', r'\bsymbolic AI\b', r'\bneuro-symbolic\b', r'\bNLP\b'
    ]
    found = set()
    for pat in patterns:
        for m in re.finditer(pat, insights, re.IGNORECASE):
            found.add(m.group(0))
    return list(found)

def extract_technical_terms(insights, hypotheses):
    text = insights + " " + hypotheses
    terms = set()
    abbrev_pattern = r'\b([A-Z]{2,})\b(?:\s*\(([^)]+)\))?'
    for match in re.finditer(abbrev_pattern, text):
        abbr = match.group(1)
        definition = match.group(2)
        if definition:
            terms.add(f"{abbr} ({definition})")
        else:
            terms.add(abbr)
    # Terms in parentheses with key words
    tech_pattern = r'\(([^)]+(?:algorithm|model|framework|method|technique|system|approach|architecture|protocol|standard)[^)]*)\)'
    for match in re.finditer(tech_pattern, text):
        terms.add(match.group(1))
    return list(terms)

def extract_sections(insights):
    # Look for lines that look like section headers
    section_headers = []
    for line in insights.split('\n'):
        line = line.strip()
        if re.match(r'^(\d+\.|[A-Z][a-z]+)\s+[A-Z]', line):
            section_headers.append(line)
        # Common section names
        elif line.lower() in ["abstract", "introduction", "conclusion", "references"]:
            section_headers.append(line)
    return section_headers

def extract_references(insights, hypotheses):
    # Look for a References section
    refs = []
    text = insights + "\n" + hypotheses
    if "references" in text.lower():
        ref_start = text.lower().find("references")
        refs_text = text[ref_start:]
        for line in refs_text.split('\n')[1:]:
            line = line.strip()
            if not line or line.lower().startswith("abstract") or len(line) < 5:
                continue
            if line.lower().startswith("keywords") or line.lower().startswith("introduction"):
                break
            refs.append(line)
    return refs

def enrich_paper(paper):
    insights = paper.get("insights", "")
    hypotheses = paper.get("hypotheses", "")
    processed_date = paper.get("processed_date", "")
    metadata = {
        "title": extract_title(insights),
        "authors": extract_authors(insights),
        "date": extract_date(insights, processed_date),
        "keywords": extract_keywords(insights),
        "technical_terms": extract_technical_terms(insights, hypotheses),
        "sections": extract_sections(insights),
        "references": extract_references(insights, hypotheses)
    }
    paper["metadata"] = metadata
    return paper

def main():
    input_path = Path("results/all_papers_results.json")
    output_path = Path("results/all_papers_results_enriched.json")
    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    enriched = [enrich_paper(p) for p in papers]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"Enriched file saved to {output_path}")

if __name__ == "__main__":
    main() 