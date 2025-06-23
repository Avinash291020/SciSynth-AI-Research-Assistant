"""Hypothesis generation from insights."""
import re

def generate_hypotheses(insights: str) -> str:
    """Generate hypotheses based on insights text."""
    # Extract the main topic and key points
    lines = insights.split('\n')
    title = ""
    topic = ""
    abstract = ""
    key_points = []
    
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line == "Title:":
            current_section = "title"
        elif line == "Main Topic:":
            current_section = "topic"
        elif line == "Abstract Summary:":
            current_section = "abstract"
        elif line == "Key Points:":
            current_section = "points"
        elif current_section == "title" and not line.endswith(':'):
            title = line
        elif current_section == "topic" and line.startswith("This paper focuses on"):
            topic = line.replace("This paper focuses on ", "").rstrip('.')
        elif current_section == "abstract":
            if not line.startswith(("Abstract", "Keywords")):
                abstract = line
        elif current_section == "points" and line[0].isdigit():
            key_points.append(line[3:])  # Remove the "N. " prefix
    
    # Clean up topic if it's too long
    if len(topic.split()) > 10:
        topic = ' '.join(topic.split()[:10]) + "..."
    
    # Generate hypotheses
    hypotheses = ["Research Hypotheses:"]
    
    # Generate from title/topic
    if topic:
        hypotheses.append("\n1. Research exploring " + topic + " could lead to novel approaches that enhance the efficiency and effectiveness of current methodologies.")
    
    # Generate from abstract
    if abstract:
        # Get the first sentence of the abstract
        abstract_sentence = re.split(r'(?<=[.!?])\s+', abstract)[0]
        hypotheses.append(f"\n2. Building upon the finding that {abstract_sentence}, we hypothesize that integrating these insights with emerging technologies could lead to breakthrough applications.")
    
    # Generate from key points
    for i, point in enumerate(key_points, 3):
        if i > 4:  # Limit to 3 hypotheses from key points
            break
        # Create a hypothesis by extending the key point
        hypothesis = f"\n{i}. Based on the observation that {point}, "
        hypothesis += "we hypothesize that further investigation could reveal new theoretical frameworks and practical applications."
        hypotheses.append(hypothesis)
    
    # If we don't have enough hypotheses, add general ones based on the title
    while len(hypotheses) < 5:
        i = len(hypotheses)
        if i == 1:
            hypotheses.append(f"\n{i}. The methodologies presented in this research could potentially be extended to solve similar challenges in related domains.")
        elif i == 2:
            hypotheses.append(f"\n{i}. Integration of these findings with other state-of-the-art approaches might yield synergistic benefits.")
        else:
            hypotheses.append(f"\n{i}. Further investigation of the underlying principles could reveal unexpected applications in adjacent fields.")
    
    return "\n".join(hypotheses)

if __name__ == "__main__":
    from insight_agent import generate_insights
    from ingest_paper import chunk_and_index, extract_text_from_pdf
    text = extract_text_from_pdf("example_paper.pdf")
    chunks = chunk_and_index(text)
    insights = generate_insights(chunks)
    hypotheses = generate_hypotheses(insights)
    print("💡 Hypotheses:\n", hypotheses) 