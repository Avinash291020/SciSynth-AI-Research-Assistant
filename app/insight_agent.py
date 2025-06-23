from typing import List, Tuple
import re
from .model_cache import ModelCache

def clean_title(title: str) -> str:
    """Clean and format the title."""
    # Remove author names and affiliations (typically after commas or numbers)
    title = re.split(r'[,\d]', title)[0].strip()
    
    # Remove dates in parentheses
    title = re.sub(r'\s*\([^)]*\)', '', title)
    
    # Remove special characters but keep basic punctuation
    title = re.sub(r'[^\w\s.,;:!?-]', '', title)
    
    # Fix spacing
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title

def extract_title_and_abstract(text: str) -> Tuple[str, str]:
    """Extract title and abstract from the text."""
    lines = text.split('\n')
    title = ""
    abstract = ""
    
    # Find title (first substantial line that's not a header)
    for line in lines:
        line = line.strip()
        if (len(line) > 20 and  # Must be substantial
            not line.lower().startswith(('abstract', 'keywords', 'introduction', 'contents')) and
            not re.match(r'^\d+\.', line)):  # Not a numbered section
            title = clean_title(line)
            break
    
    # Find abstract
    abstract_started = False
    abstract_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.lower().startswith('abstract'):
            abstract_started = True
            continue
        elif abstract_started:
            if line.lower().startswith(('keywords', 'introduction', 'contents')) or re.match(r'^\d+\.', line):
                break
            if line:  # Non-empty line
                # Clean the line
                line = re.sub(r'[^\w\s.,;:!?()-]', '', line)
                line = re.sub(r'\s+', ' ', line).strip()
                abstract_lines.append(line)
    
    abstract = ' '.join(abstract_lines)
    
    return title, abstract

def extract_topic(title: str, abstract: str) -> str:
    """Extract the main topic from title and abstract."""
    # Try to get topic from title first
    if ':' in title:
        topic = title.split(':')[0].strip()
    else:
        # Use the first part of the title
        words = title.split()
        if len(words) > 5:
            topic = ' '.join(words[:5]) + "..."
        else:
            topic = title
    
    # If topic is too short, try to augment with abstract
    if len(topic.split()) < 3 and abstract:
        first_sentence = re.split(r'(?<=[.!?])\s+', abstract)[0]
        topic = f"{topic} - {first_sentence}"
    
    # Limit length
    words = topic.split()
    if len(words) > 10:
        topic = ' '.join(words[:10]) + "..."
    
    return topic

def generate_insights(chunks: List[str]) -> str:
    """Generate insights from the document chunks."""
    if not chunks:
        return "No text content available for analysis."
    
    # Get title and abstract
    title, abstract = extract_title_and_abstract(chunks[0])
    if not title:
        title = "Untitled Document"
    
    # Extract main topic
    topic = extract_topic(title, abstract)
    
    # Generate insights
    insights = [
        "Title:",
        title,
        "\nMain Topic:",
        f"This paper focuses on {topic}",
        "\nAbstract Summary:"
    ]
    
    if abstract:
        insights.append(abstract)
    else:
        # If no abstract found, use first few chunks
        summary = []
        for chunk in chunks[1:3]:
            # Get first sentence of each chunk
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            if sentences:
                summary.append(sentences[0])
        insights.append(' '.join(summary))
    
    # Add key points from later chunks
    insights.append("\nKey Points:")
    point_count = 0
    
    for chunk in chunks[1:5]:  # Look at next few chunks
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        for sentence in sentences:
            # Clean the sentence
            sentence = re.sub(r'[^\w\s.,;:!?()-]', '', sentence)
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            
            # Only use substantial sentences that aren't headers
            if (len(sentence) > 50 and 
                not sentence.endswith(':') and 
                not sentence.startswith(('Keywords', 'Abstract', 'Introduction')) and
                not re.match(r'^\d+\.', sentence)):
                point_count += 1
                insights.append(f"{point_count}. {sentence}")
                if point_count >= 3:  # Limit to 3 key points
                    break
        if point_count >= 3:
            break
    
    # If no key points were found, add a placeholder
    if point_count == 0:
        insights.append("1. No substantial key points found in the text.")
    
    return "\n".join(insights)

if __name__ == "__main__":
    from ingest_paper import chunk_and_index, extract_text_from_pdf
    text = extract_text_from_pdf("example_paper.pdf")
    chunks = chunk_and_index(text)
    insights = generate_insights(chunks)
    print("🧠 Insights:\n", insights) 