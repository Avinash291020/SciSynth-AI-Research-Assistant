import json
import re
from pathlib import Path


def extract_title(insights):
    """Extract title more accurately from insights."""
    lines = insights.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Title:"):
            # Get the title from the next line or the same line
            title = line.replace("Title:", "").strip()
            if not title and i + 1 < len(lines):
                title = lines[i + 1].strip()
            if title:
                # Clean up the title
                title = re.sub(r"\s+", " ", title)  # Remove extra spaces
                title = title[:200]  # Limit length
                return title
    return ""


def extract_authors(insights):
    """Extract authors more accurately."""
    lines = insights.split("\n")
    authors = []

    # Look for author patterns in the first few lines after title
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Title:"):
            # Look for author line after title
            for j in range(i + 1, min(i + 5, len(lines))):
                author_line = lines[j].strip()
                if author_line and len(author_line) > 10:
                    # Check if this looks like an author line (contains names and affiliations)
                    if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", author_line):
                        # Extract names before affiliations
                        names_part = re.split(
                            r"\d|\(|\[|Department|University|Institute", author_line
                        )[0]
                        # Split by comma and clean
                        names = [
                            name.strip()
                            for name in names_part.split(",")
                            if name.strip()
                        ]
                        # Filter out very short or very long names
                        names = [name for name in names if 3 <= len(name) <= 50]
                        if names:
                            authors.extend(names)
                        break
            break

    # Remove duplicates and limit
    return list(dict.fromkeys(authors))[:10]


def extract_date(insights, processed_date):
    """Extract date more accurately."""
    # Look for year patterns in the text
    year_patterns = [
        r"(\d{4})",  # Any 4-digit year
        r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
        r"(\d{2}/\d{2}/\d{4})",  # MM/DD/YYYY
    ]

    for pattern in year_patterns:
        matches = re.findall(pattern, insights)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            # Validate year is reasonable
            if match.isdigit() and 1900 <= int(match) <= 2030:
                return match

    # Fallback to processed_date
    if processed_date:
        # Extract year from processed_date
        year_match = re.search(r"(\d{4})", processed_date)
        if year_match:
            return year_match.group(1)

    return ""


def extract_keywords(insights):
    """Extract keywords more accurately."""
    keywords = []

    # Look for explicit keywords section
    if "Keywords:" in insights:
        kw_section = insights.split("Keywords:")[1].split("\n")[0]
        # Clean up and split keywords
        kw_section = re.sub(r"\s+", " ", kw_section)
        keywords = [kw.strip() for kw in kw_section.split(",") if kw.strip()]
        # Filter out very long keywords
        keywords = [kw for kw in keywords if len(kw) <= 50]

    # If no explicit keywords, extract from common patterns
    if not keywords:
        # Common AI/ML keywords
        ai_keywords = [
            "neural networks",
            "deep learning",
            "machine learning",
            "artificial intelligence",
            "AI",
            "language models",
            "transformers",
            "reinforcement learning",
            "evolutionary algorithms",
            "symbolic AI",
            "neuro-symbolic",
            "NLP",
            "computer vision",
            "optimization",
            "inference",
            "training",
        ]

        for keyword in ai_keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", insights, re.IGNORECASE):
                keywords.append(keyword)

    return keywords[:10]  # Limit to 10 keywords


def extract_technical_terms(insights, hypotheses):
    """Extract technical terms more comprehensively."""
    text = insights + " " + hypotheses
    terms = set()

    # Abbreviations with definitions
    abbrev_pattern = r"\b([A-Z]{2,})\b(?:\s*\(([^)]+)\))?"
    for match in re.finditer(abbrev_pattern, text):
        abbr = match.group(1)
        definition = match.group(2)
        if definition:
            terms.add(f"{abbr} ({definition})")
        else:
            terms.add(abbr)

    # Technical terms in parentheses
    tech_pattern = r"\(([^)]+(?:algorithm|model|framework|method|technique|system|approach|architecture|protocol|standard|network|transformer|encoder|decoder)[^)]*)\)"
    for match in re.finditer(tech_pattern, text):
        terms.add(match.group(1))

    # Algorithm names
    algo_patterns = [
        r"Algorithm\s+\d+[:\s]*([A-Za-z\s]+)",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Algorithm\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Method\b",
    ]
    for pattern in algo_patterns:
        for match in re.finditer(pattern, text):
            terms.add(match.group(1).strip())

    # Model names
    model_patterns = [
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Model\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Network\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Transformer\b",
    ]
    for pattern in model_patterns:
        for match in re.finditer(pattern, text):
            terms.add(match.group(1))

    # Framework names
    framework_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Framework\b"
    for match in re.finditer(framework_pattern, text):
        terms.add(match.group(1))

    # Scientific terms with numbers
    scientific_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\d*[A-Za-z]*)\b"
    for match in re.finditer(scientific_pattern, text):
        term = match.group(1).strip()
        if len(term) > 3 and any(c.isdigit() for c in term):
            terms.add(term)

    # Filter out common words and very short terms
    common_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "been",
        "they",
        "their",
    }
    filtered_terms = [
        term for term in terms if len(term) > 2 and term.lower() not in common_words
    ]

    return filtered_terms[:15]  # Limit to 15 terms


def extract_sections(insights):
    """Extract section headers more accurately."""
    sections = []
    lines = insights.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered sections
        if re.match(r"^\d+\.\s+[A-Z]", line):
            sections.append(line)
        # Look for common section names
        elif line.lower() in [
            "abstract",
            "introduction",
            "conclusion",
            "references",
            "bibliography",
            "methodology",
            "results",
            "discussion",
        ]:
            sections.append(line)
        # Look for all-caps section headers
        elif line.isupper() and len(line) > 3 and len(line) < 50:
            sections.append(line)

    return sections[:10]  # Limit to 10 sections


def extract_references(insights, hypotheses):
    """Extract references more accurately."""
    refs = []
    text = insights + "\n" + hypotheses

    # Look for references section
    if "references" in text.lower():
        ref_start = text.lower().find("references")
        refs_text = text[ref_start:]

        # Split into lines and look for reference patterns
        for line in refs_text.split("\n")[1:]:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            # Stop if we hit another major section
            if line.lower().startswith(
                ("abstract", "introduction", "conclusion", "keywords")
            ):
                break
            # Look for reference patterns (author names, years, etc.)
            if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+.*\d{4}", line):
                refs.append(line[:200])  # Limit length

    return refs[:20]  # Limit to 20 references


def enrich_paper(paper):
    """Enrich a single paper with metadata."""
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
        "references": extract_references(insights, hypotheses),
    }

    paper["metadata"] = metadata
    return paper


def main():
    """Main function to enrich all papers."""
    input_path = Path("results/all_papers_results.json")
    output_path = Path("results/all_papers_results_enhanced.json")

    print("🔧 Loading papers...")
    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"📚 Enriching {len(papers)} papers...")
    enriched = []
    for i, paper in enumerate(papers):
        print(
            f"  Processing paper {i+1}/{len(papers)}: {paper.get('paper_name', 'Unknown')}"
        )
        enriched.append(enrich_paper(paper))

    print("💾 Saving enriched papers...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f"✅ Enhanced file saved to {output_path}")

    # Print some statistics
    titles_extracted = sum(1 for p in enriched if p["metadata"]["title"])
    authors_extracted = sum(1 for p in enriched if p["metadata"]["authors"])
    keywords_extracted = sum(1 for p in enriched if p["metadata"]["keywords"])
    terms_extracted = sum(1 for p in enriched if p["metadata"]["technical_terms"])

    print(f"\n📊 Extraction Statistics:")
    print(f"  Papers with titles: {titles_extracted}/{len(papers)}")
    print(f"  Papers with authors: {authors_extracted}/{len(papers)}")
    print(f"  Papers with keywords: {keywords_extracted}/{len(papers)}")
    print(f"  Papers with technical terms: {terms_extracted}/{len(papers)}")


if __name__ == "__main__":
    main()
