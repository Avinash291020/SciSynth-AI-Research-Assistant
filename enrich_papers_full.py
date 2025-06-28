# -*- coding: utf-8 -*-
"""
Enhanced paper enrichment with full processing capabilities.
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def extract_title(insights):
    lines = insights.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Title:"):
            title = line.replace("Title:", "").strip()
            if not title and i + 1 < len(lines):
                title = lines[i + 1].strip()
            if title:
                title = re.sub(r"\s+", " ", title)
                return title[:200]
    return ""


def extract_authors_and_affiliations(insights):
    lines = insights.split("\n")
    authors = []
    affiliations = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Title:"):
            for j in range(i + 1, min(i + 6, len(lines))):
                author_line = lines[j].strip()
                # Look for names (Firstname Lastname)
                name_matches = re.findall(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", author_line)
                if name_matches:
                    authors.extend(name_matches)
                # Look for affiliations (University, Institute, Department, Inc, Ltd, Lab, School, Center)
                affil_matches = re.findall(
                    r"([A-Z][A-Za-z&\-\. ]*(University|Institute|Department|Inc|Ltd|Lab|School|Center|College|Faculty|Hospital|Company|Corporation|Research|Academy)[A-Za-z&\-\. ]*)",
                    author_line,
                )
                for affil in affil_matches:
                    affiliations.append(affil[0].strip())
            break
    return list(dict.fromkeys(authors))[:10], list(dict.fromkeys(affiliations))[:10]


def extract_date(insights, processed_date):
    year_patterns = [r"(\d{4})", r"(\d{4}-\d{2}-\d{2})", r"(\d{2}/\d{2}/\d{4})"]
    for pattern in year_patterns:
        matches = re.findall(pattern, insights)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            if match.isdigit() and 1900 <= int(match) <= 2030:
                return match
    if processed_date:
        year_match = re.search(r"(\d{4})", processed_date)
        if year_match:
            return year_match.group(1)
    return ""


def extract_keywords(insights):
    keywords = []
    if "Keywords:" in insights:
        kw_section = insights.split("Keywords:")[1].split("\n")[0]
        kw_section = re.sub(r"\s+", " ", kw_section)
        keywords = [
            kw.strip() for kw in kw_section.split(",") if kw.strip() and len(kw) <= 50
        ]
    if not keywords:
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
    return keywords[:10]


def extract_technical_terms(insights, hypotheses):
    text = insights + " " + hypotheses
    terms = set()
    abbrev_pattern = r"\b([A-Z]{2,})\b(?:\s*\(([^)]+)\))?"
    for match in re.finditer(abbrev_pattern, text):
        abbr = match.group(1)
        definition = match.group(2)
        if definition:
            terms.add(f"{abbr} ({definition})")
        else:
            terms.add(abbr)
    tech_pattern = r"\(([^)]+(?:algorithm|model|framework|method|technique|system|approach|architecture|protocol|standard|network|transformer|encoder|decoder)[^)]*)\)"
    for match in re.finditer(tech_pattern, text):
        terms.add(match.group(1))
    algo_patterns = [
        r"Algorithm\s+\d+[:\s]*([A-Za-z\s]+)",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Algorithm\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Method\b",
    ]
    for pattern in algo_patterns:
        for match in re.finditer(pattern, text):
            terms.add(match.group(1).strip())
    model_patterns = [
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Model\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Network\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Transformer\b",
    ]
    for pattern in model_patterns:
        for match in re.finditer(pattern, text):
            terms.add(match.group(1))
    framework_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Framework\b"
    for match in re.finditer(framework_pattern, text):
        terms.add(match.group(1))
    scientific_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\d*[A-Za-z]*)\b"
    for match in re.finditer(scientific_pattern, text):
        term = match.group(1).strip()
        if len(term) > 3 and any(c.isdigit() for c in term):
            terms.add(term)
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
    return filtered_terms[:15]


def extract_sections_and_text(insights):
    # Find section headers and their text
    section_headers = [
        "abstract",
        "introduction",
        "background",
        "related work",
        "methods",
        "methodology",
        "approach",
        "results",
        "experiments",
        "discussion",
        "conclusion",
        "references",
        "acknowledgments",
        "future work",
    ]
    lines = insights.split("\n")
    sections = defaultdict(list)
    current_section = None
    for line in lines:
        line_clean = line.strip().lower().rstrip(":")
        if not line_clean:
            continue
        # Detect section header
        if line_clean in section_headers or re.match(r"^[0-9]+\.\s+[A-Z]", line):
            current_section = line.strip().rstrip(":")
            continue
        if current_section:
            sections[current_section].append(line.strip())
    # Join lines for each section
    return {k: " ".join(v)[:2000] for k, v in sections.items() if v}


def extract_references_structured(insights, hypotheses):
    text = insights + "\n" + hypotheses
    refs = []
    if "references" in text.lower():
        ref_start = text.lower().find("references")
        refs_text = text[ref_start:]
        for line in refs_text.split("\n")[1:]:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            if line.lower().startswith(
                ("abstract", "introduction", "conclusion", "keywords")
            ):
                break
            # Try to parse reference
            ref = {}
            # DOI
            doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", line, re.I)
            if doi_match:
                ref["doi"] = doi_match.group(1)
            # URL
            url_match = re.search(r"(https?://\S+)", line)
            if url_match:
                ref["url"] = url_match.group(1)
            # Year
            year_match = re.search(r"(19|20)\d{2}", line)
            if year_match:
                ref["year"] = year_match.group(0)
            # Authors (before year)
            if "year" in ref:
                before_year = line.split(ref["year"])[0]
                author_names = re.findall(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", before_year)
                if author_names:
                    ref["authors"] = author_names
            # Title (between authors and year, or after year)
            if "year" in ref:
                after_year = line.split(ref["year"])[1] if ref["year"] in line else ""
                title_match = re.search(r"([A-Z][^\.]+)", after_year)
                if title_match:
                    ref["title"] = title_match.group(1).strip()
            refs.append(ref)
    return refs[:20]


def extract_doi_url(insights, hypotheses):
    text = insights + "\n" + hypotheses
    dois = re.findall(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, re.I)
    urls = re.findall(r"(https?://\S+)", text)
    return list(set(dois)), list(set(urls))


def classify_paper_type(insights):
    # Heuristic: look for keywords in the first 500 chars
    text = insights[:500].lower()
    if "survey" in text or "overview" in text:
        return "survey"
    if "review" in text:
        return "review"
    if "method" in text or "approach" in text or "algorithm" in text:
        return "method"
    if "application" in text or "case study" in text:
        return "application"
    if "experiment" in text or "result" in text:
        return "original research"
    return "unknown"


def enrich_paper(paper):
    insights = paper.get("insights", "")
    hypotheses = paper.get("hypotheses", "")
    processed_date = paper.get("processed_date", "")
    title = extract_title(insights)
    authors, affiliations = extract_authors_and_affiliations(insights)
    date = extract_date(insights, processed_date)
    keywords = extract_keywords(insights)
    technical_terms = extract_technical_terms(insights, hypotheses)
    sections = extract_sections_and_text(insights)
    references = extract_references_structured(insights, hypotheses)
    dois, urls = extract_doi_url(insights, hypotheses)
    paper_type = classify_paper_type(insights)
    metadata = {
        "title": title,
        "authors": authors,
        "affiliations": affiliations,
        "date": date,
        "keywords": keywords,
        "technical_terms": technical_terms,
        "sections": sections,
        "references": references,
        "dois": dois,
        "urls": urls,
        "paper_type": paper_type,
    }
    paper["metadata"] = metadata
    return paper


def main():
    input_path = Path("results/all_papers_results.json")
    output_path = Path("results/all_papers_results_full.json")
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
    print(f"✅ Full enriched file saved to {output_path}")
    # Print some statistics
    titles_extracted = sum(1 for p in enriched if p["metadata"]["title"])
    authors_extracted = sum(1 for p in enriched if p["metadata"]["authors"])
    affiliations_extracted = sum(1 for p in enriched if p["metadata"]["affiliations"])
    keywords_extracted = sum(1 for p in enriched if p["metadata"]["keywords"])
    terms_extracted = sum(1 for p in enriched if p["metadata"]["technical_terms"])
    print(f"\n📊 Extraction Statistics:")
    print(f"  Papers with titles: {titles_extracted}/{len(papers)}")
    print(f"  Papers with authors: {authors_extracted}/{len(papers)}")
    print(f"  Papers with affiliations: {affiliations_extracted}/{len(papers)}")
    print(f"  Papers with keywords: {keywords_extracted}/{len(papers)}")
    print(f"  Papers with technical terms: {terms_extracted}/{len(papers)}")
    # Show sample
    print(f"\n📄 Sample Metadata (First Paper):")
    print(json.dumps(enriched[0]["metadata"], indent=2, ensure_ascii=False)[:1000])


if __name__ == "__main__":
    main()
