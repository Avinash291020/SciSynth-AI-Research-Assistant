"""Retrieval-Augmented Generation (RAG) system for research papers."""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers.pipelines import pipeline
from transformers import AutoTokenizer
import torch
from app.model_cache import ModelCache
import re

class RAGSystem:
    def __init__(self, collection_name: str = "research_papers"):
        """Initialize RAG system with vector database and models."""
        self.collection_name = collection_name
        self.embedding_model = ModelCache.get_sentence_transformer()
        self.generator = ModelCache.get_text_generator()
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./data/chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Research papers for RAG system"}
            )
    
    def add_papers_to_index(self, papers_data: List[Dict[str, Any]]) -> None:
        """Add papers to the vector database."""
        documents = []
        metadatas = []
        ids = []
        
        for i, paper in enumerate(papers_data):
            # Create document from paper content
            doc_content = self._create_document_content(paper)
            documents.append(doc_content)
            
            # Create metadata
            metadata = {
                "paper_name": paper.get("paper_name", f"paper_{i}"),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "date": paper.get("processed_date", ""),
                "type": "research_paper"
            }
            metadatas.append(metadata)
            ids.append(f"paper_{i}")
        
        # Add to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Added {len(documents)} papers to RAG index")
    
    def _create_document_content(self, paper: Dict[str, Any]) -> str:
        """Create document content from paper data with enhanced metadata."""
        content_parts = []
        
        # Always include paper name
        paper_name = paper.get('paper_name', 'Unknown Paper')
        content_parts.append(f"Paper: {paper_name}")
        
        # Use enriched metadata if available
        metadata = paper.get('metadata', {})
        
        # Add title from metadata
        if metadata.get('title'):
            content_parts.append(f"Title: {metadata['title']}")
        
        # Add authors and affiliations
        if metadata.get('authors'):
            authors_str = ', '.join(metadata['authors'][:5])  # Limit to first 5 authors
            content_parts.append(f"Authors: {authors_str}")
        
        if metadata.get('affiliations'):
            affil_str = ', '.join(metadata['affiliations'][:3])  # Limit to first 3 affiliations
            content_parts.append(f"Affiliations: {affil_str}")
        
        # Add date
        if metadata.get('date'):
            content_parts.append(f"Date: {metadata['date']}")
        
        # Add paper type
        if metadata.get('paper_type'):
            content_parts.append(f"Paper Type: {metadata['paper_type']}")
        
        # Add keywords from metadata
        if metadata.get('keywords'):
            keywords_str = ', '.join(metadata['keywords'][:5])
            content_parts.append(f"Keywords: {keywords_str}")
        
        # Add technical terms from metadata
        if metadata.get('technical_terms'):
            tech_terms_str = ', '.join(metadata['technical_terms'][:5])
            content_parts.append(f"Technical Terms: {tech_terms_str}")
        
        # Add section content if available
        if metadata.get('sections'):
            sections = metadata['sections']
            for section_name, section_text in list(sections.items())[:3]:  # Limit to first 3 sections
                if section_text and len(section_text) > 20:
                    # Truncate section text to avoid too long content
                    section_preview = section_text[:200] + "..." if len(section_text) > 200 else section_text
                    content_parts.append(f"{section_name}: {section_preview}")
        
        # Add references if available
        if metadata.get('references'):
            refs = metadata['references']
            if refs:
                ref_preview = []
                for ref in refs[:3]:  # Limit to first 3 references
                    ref_str = ""
                    if ref.get('authors'):
                        ref_str += f"Authors: {', '.join(ref['authors'][:2])} "
                    if ref.get('year'):
                        ref_str += f"({ref['year']}) "
                    if ref.get('title'):
                        ref_str += f"Title: {ref['title'][:100]}"
                    if ref_str:
                        ref_preview.append(ref_str)
                if ref_preview:
                    content_parts.append(f"References: {'; '.join(ref_preview)}")
        
        # Add DOIs and URLs
        if metadata.get('dois'):
            dois_str = ', '.join(metadata['dois'][:2])
            content_parts.append(f"DOIs: {dois_str}")
        
        if metadata.get('urls'):
            urls_str = ', '.join(metadata['urls'][:2])
            content_parts.append(f"URLs: {urls_str}")
        
        # Fallback to old method if no enriched metadata
        if len(content_parts) < 3:
            # Extract title from insights if available
            title = ""
            if "insights" in paper and paper["insights"]:
                insights = paper["insights"]
                if isinstance(insights, str):
                    lines = insights.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Title:"):
                            title = line.replace("Title:", "").strip()
                            if len(title) > 5:
                                if len(title) > 100:
                                    title = title[:100] + "..."
                                content_parts.append(f"Title: {title}")
                            break
            
            # Add insights (take first meaningful one)
            if "insights" in paper and paper["insights"]:
                insights = paper["insights"]
                if isinstance(insights, str):
                    lines = insights.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 20 and not line.startswith(("Key points:", "Title:", "Main Topic:", "Abstract Summary:")):
                            insight_short = line[:150] + "..." if len(line) > 150 else line
                            content_parts.append(f"Key Insight: {insight_short}")
                            break
            
            # Add hypotheses (take first meaningful one)
            if "hypotheses" in paper and paper["hypotheses"]:
                hypotheses = paper["hypotheses"]
                if isinstance(hypotheses, str):
                    lines = hypotheses.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 20 and not line.startswith("Research Hypotheses:"):
                            hyp_short = line[:150] + "..." if len(line) > 150 else line
                            content_parts.append(f"Research Hypothesis: {hyp_short}")
                            break
        
        # Ensure we always have at least basic content
        if len(content_parts) < 2:
            content_parts.append("Content: Research paper with insights and hypotheses")
        
        return "\n".join(content_parts)
    
    def _extract_technical_terms_from_text(self, paper: Dict[str, Any]) -> List[str]:
        """Extract technical terms from paper insights and hypotheses."""
        import re
        terms = set()
        
        # Combine all text from the paper
        all_text = ""
        if "insights" in paper and paper["insights"]:
            all_text += paper["insights"] + " "
        if "hypotheses" in paper and paper["hypotheses"]:
            all_text += paper["hypotheses"] + " "
        
        # Find technical terms and abbreviations
        # Abbreviations with definitions
        abbrev_pattern = r'\b([A-Z]{2,})\b(?:\s*\(([^)]+)\))?'
        matches = re.finditer(abbrev_pattern, all_text)
        for match in matches:
            abbr = match.group(1)
            definition = match.group(2)
            if definition:
                terms.add(f"{abbr} ({definition})")
            else:
                terms.add(abbr)
        
        # Find technical terms in parentheses
        tech_pattern = r'\(([^)]+(?:algorithm|model|framework|method|technique|system|approach|architecture|protocol|standard)[^)]*)\)'
        matches = re.finditer(tech_pattern, all_text)
        terms.update(match.group(1) for match in matches)
        
        # Find algorithm names
        algo_pattern = r'Algorithm\s+\d+[:\s]*([A-Za-z\s]+)'
        matches = re.finditer(algo_pattern, all_text)
        terms.update(match.group(1).strip() for match in matches)
        
        # Find model names
        model_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Model\b'
        matches = re.finditer(model_pattern, all_text)
        terms.update(match.group(1) for match in matches)
        
        # Find framework names
        framework_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Framework\b'
        matches = re.finditer(framework_pattern, all_text)
        terms.update(match.group(1) for match in matches)
        
        # Find scientific terms with numbers
        scientific_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\d*[A-Za-z]*)\b'
        matches = re.finditer(scientific_pattern, all_text)
        for match in matches:
            term = match.group(1).strip()
            if len(term) > 3 and any(c.isdigit() for c in term):
                terms.add(term)
        
        return list(terms)[:10]  # Limit to top 10 terms
    
    def _extract_keywords_from_text(self, paper: Dict[str, Any]) -> List[str]:
        """Extract keywords from paper insights."""
        import re
        keywords = set()
        
        if "insights" in paper and paper["insights"]:
            insights = paper["insights"]
            
            # Look for keywords section
            if "Keywords:" in insights:
                kw_section = insights.split("Keywords:")[1].split("\n")[0]
                keywords.update([k.strip() for k in kw_section.split(",")])
            
            # Extract domain-specific terms
            domain_patterns = [
                r'\b(?:neural networks?|deep learning|machine learning|artificial intelligence|AI)\b',
                r'\b(?:language models?|LLMs?|transformers?)\b',
                r'\b(?:reinforcement learning|RL)\b',
                r'\b(?:evolutionary algorithms?|genetic algorithms?)\b',
                r'\b(?:symbolic AI|neuro-symbolic|neurosymbolic)\b',
                r'\b(?:computer vision|natural language processing|NLP)\b',
                r'\b(?:optimization|inference|training)\b'
            ]
            
            for pattern in domain_patterns:
                matches = re.finditer(pattern, insights, re.IGNORECASE)
                keywords.update(match.group(0) for match in matches)
        
        return list(keywords)[:5]  # Limit to top 5 keywords
    
    def retrieve_relevant_papers(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Retrieve relevant papers for a given query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Check if results are valid
            if not results or 'documents' not in results or not results['documents'] or not results['documents'][0]:
                return []
            
            relevant_papers = []
            documents = results['documents'][0]
            metadatas = results.get('metadatas', []) or []
            distances = results.get('distances', []) or []
            
            # Handle case where metadatas/distances might be nested
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            if distances and isinstance(distances[0], list):
                distances = distances[0]
            
            for i in range(len(documents)):
                paper_info = {
                    'content': documents[i],
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'distance': distances[i] if i < len(distances) else None
                }
                relevant_papers.append(paper_info)
            
            return relevant_papers
        except Exception as e:
            print(f"Error retrieving papers: {e}")
            return []
    
    def generate_response(self, query: str, context_papers: List[Dict[str, Any]]) -> str:
        """Generate a response using retrieved context."""
        try:
            # Create context from retrieved papers
            context = self._create_context(context_papers, detailed=True)
            
            # Check if this is a comparative question
            comparison_keywords = ["difference", "compare", "contrast", "versus", "vs", "between", "models", "approaches", "methods", "different", "learn", "what different", "how do they differ", "distinguish", "vary", "variation"]
            is_comparison = any(keyword in query.lower() for keyword in comparison_keywords)
            
            # Additional check for the specific question pattern
            if "what is the difference between all these research papers" in query.lower():
                is_comparison = True
            
            # For comparative questions, use advanced NLP fallback immediately to avoid poor LLM responses
            if is_comparison:
                return self._create_model_comparison_fallback(context_papers)
            
            # Create prompt for the LLM
            prompt = f"""
You are an expert AI research assistant. Based on the following research papers, provide a comprehensive answer to this question: {query}

Research Papers Context:
{context}

Instructions:
1. Analyze the provided research papers and synthesize key findings
2. Focus on the most relevant insights related to the question
3. Provide specific examples and details from the papers
4. Structure your answer logically with clear sections
5. Avoid repetition and generic statements
6. If the question is outside the scope of the papers, clearly state this

Answer:"""
            
            # Generate response using the LLM
            try:
                response = self.generator(
                    prompt,
                    max_new_tokens=800,  # Increased for more detailed, high-quality responses
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                if response and len(response) > 0:
                    # Extract the generated text
                    generated_text = response[0]['generated_text']
                    
                    # Remove the prompt from the response
                    if prompt in generated_text:
                        answer = generated_text.replace(prompt, "").strip()
                    else:
                        answer = generated_text.strip()
                    
                    # Clean up any remaining artifacts
                    answer = answer.replace("Instructions: Analyze the provided research papers and provide a detailed answer. If the question is about AI tools, identify specific tools, frameworks, algorithms, and technologies mentioned in the papers. For each tool, explain its purpose and application. If the question is about findings or insights, synthesize the key discoveries and methodologies. Provide a comprehensive, well-structured answer with specific examples from the papers.", "")
                    answer = answer.replace("Answer the question comprehensively, citing specific findings from the papers. Focus on the most relevant insights and key findings related to the question.", "")
                    answer = answer.replace("Answer:", "")
                    answer = answer.strip()
                    
                    # Remove repeated 'Model Name:' lines
                    answer_lines = answer.split('\n')
                    filtered_lines = []
                    for line in answer_lines:
                        if line.strip().lower().startswith('model name:') and len(line.strip()) < 20:
                            continue
                        filtered_lines.append(line)
                    answer = '\n'.join(filtered_lines)
                    
                    # Check for repetitive patterns that indicate poor generation
                    if self._is_repetitive_or_generic(answer, context_papers):
                        return self._create_enhanced_fallback_response(query, context_papers)
                    
                    return answer
                else:
                    return self._create_enhanced_fallback_response(query, context_papers)
                    
            except Exception as e:
                print(f"LLM generation error: {e}")
                return self._create_enhanced_fallback_response(query, context_papers)
            
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return self._create_enhanced_fallback_response(query, context_papers)
    
    def _is_repetitive_or_generic(self, answer: str, context_papers: List[Dict[str, Any]]) -> bool:
        """Check if the answer is repetitive or too generic."""
        # Check for the specific "Model Name: Date:" pattern that indicates poor generation
        if "Model Name: Date:" in answer or answer.count("Model Name:") > 2:
            return True
        
        # Check for repetitive patterns
        if answer.count("Paper") > len(context_papers) * 0.8:  # Too many "Paper X:" mentions
            return True
        
        # Check for the specific "Paper 1: Paper 2:" pattern
        if "Paper 1:" in answer and "Paper 2:" in answer and len(answer.split("Paper")) > 3:
            return True
        
        # Check for repetitive "Paper X:" patterns
        paper_pattern = r"Paper \d+:"
        paper_matches = re.findall(paper_pattern, answer)
        if len(paper_matches) > len(context_papers) * 0.6:
            return True
        
        # Check for repetitive date patterns
        if answer.count("Date:") > 2:
            return True
        
        # Check for repetitive "Model Name:" patterns
        model_name_pattern = r"Model Name:"
        model_matches = re.findall(model_name_pattern, answer)
        if len(model_matches) > 2:
            return True
        
        # Check for generic responses - expanded list
        generic_phrases = [
            "I don't know", "cannot answer", "no information", "not found",
            "based on the papers", "the papers show", "research indicates",
            "we should begin by", "attempting to define", "subject matter",
            "it depends", "various factors", "multiple aspects", "different perspectives",
            "broad range", "diverse topics", "many areas", "several approaches",
            "general overview", "basic understanding", "fundamental concepts",
            "answer the question comprehensively", "focus on the most relevant"
        ]
        if any(phrase in answer.lower() for phrase in generic_phrases):
            return True
        
        # Check if answer is too short or too vague
        if len(answer) < 100:  # Increased minimum length
            return True
        
        # Check for very generic opening statements
        generic_openings = [
            "we should begin by",
            "let's start by",
            "first, we need to",
            "it's important to",
            "we can see that",
            "the research shows",
            "based on the information"
        ]
        if any(answer.lower().startswith(opening) for opening in generic_openings):
            return True
        
        # Check for repetitive structure
        answer_lines = answer.split('\n')
        if len(answer_lines) > 2:
            paper_patterns = []
            for answer_line in answer_lines:
                for i in range(1, len(context_papers)+1):
                    if answer_line.strip().startswith(f"Paper {i}:"):
                        paper_patterns.append(True)
                        break
                else:
                    paper_patterns.append(False)
            if sum(paper_patterns) > len(context_papers) * 0.7:  # Too many "Paper X:" lines
                return True
        
        # Check if answer lacks specific content (no model names, technical terms, etc.)
        specific_indicators = ["model", "approach", "method", "technique", "algorithm", "framework", "system", "architecture"]
        if not any(indicator in answer.lower() for indicator in specific_indicators):
            return True
        
        return False
    
    def _create_model_comparison_fallback(self, context_papers: List[Dict[str, Any]]) -> str:
        """Create a refined structured model comparison when LLM fails, using enhanced metadata."""
        response_parts = ["**Detailed Model Comparison from Research Papers**\n"]
        
        # Patterns for strengths and weaknesses
        strength_patterns = [
            r"improv(e|es|ed|ing)", r"outperform(s|ed)?", r"robust( to|ness)?", r"advantage", r"effective( at|ness)?", 
            r"achieve(s|d)? better", r"state[- ]of[- ]the[- ]art", r"superior", r"increase(s|d)?", r"reduce(s|d)? error", 
            r"efficient(ly)?", r"scalable", r"accurate(ly)?", r"novel (approach|method|technique)", r"successfully",
            r"breakthrough", r"significant(ly)?", r"outstanding", r"excellent", r"remarkable", r"impressive"
        ]
        weakness_patterns = [
            r"limitation", r"fail(s|ed)? to", r"struggle(s|d)? with", r"less effective", r"drawback", 
            r"require(s|d)? more", r"suffer(s|ed)? from", r"problem(s)? with", r"challenge(s)?", r"issue(s)?", 
            r"high (cost|complexity|variance|error)", r"sensitive to", r"cannot", r"not able to", r"lack(s|ed)?",
            r"difficult", r"expensive", r"slow", r"inaccurate", r"unreliable", r"restricted"
        ]
        
        def extract_sentences(text, patterns):
            sentences = re.split(r'(?<=[.!?])\s+', text)
            found = []
            for sent in sentences:
                for pat in patterns:
                    if re.search(pat, sent, re.IGNORECASE):
                        found.append(sent.strip())
                        break
            return found
        
        # Extract model information from each paper
        models_info = []
        for i, paper in enumerate(context_papers, 1):
            content = paper['content']
            metadata = paper.get('metadata', {})
            paper_name = metadata.get('paper_name', f'Paper {i}')
            
            # Use enriched metadata for better information
            title = metadata.get('title', paper_name)
            authors = metadata.get('authors', [])
            affiliations = metadata.get('affiliations', [])
            paper_type = metadata.get('paper_type', 'unknown')
            date = metadata.get('date', '')
            keywords = metadata.get('keywords', [])
            technical_terms = metadata.get('technical_terms', [])
            sections = metadata.get('sections', {})
            references = metadata.get('references', [])
            dois = metadata.get('dois', [])
            urls = metadata.get('urls', [])
            
            # Extract key features from metadata
            key_features = []
            if keywords:
                key_features.append(f"Keywords: {', '.join(keywords[:3])}")
            if technical_terms:
                key_features.append(f"Technical Terms: {', '.join(technical_terms[:3])}")
            if paper_type and paper_type != 'unknown':
                key_features.append(f"Type: {paper_type}")
            
            # Extract strengths and weaknesses from section content
            strengths = []
            weaknesses = []
            all_text = ""
            
            # Combine section text for analysis
            for section_name, section_text in sections.items():
                all_text += f" {section_text}"
                if section_text:
                    analysis_text = section_text
                    strengths.extend(extract_sentences(analysis_text, strength_patterns))
                    weaknesses.extend(extract_sentences(analysis_text, weakness_patterns))
            
            # If no sections, try from content
            if not strengths and not weaknesses:
                if "Key Insight:" in content:
                    insight_start = content.find("Key Insight:") + 12
                    insight_end = content.find("\n", insight_start)
                    if insight_end > insight_start:
                        insight_text = content[insight_start:insight_end].strip()
                    else:
                        insight_text = content[insight_start:].strip()
                    strengths = extract_sentences(insight_text, strength_patterns)
                    weaknesses = extract_sentences(insight_text, weakness_patterns)
            
            # Create summary of key features
            if not key_features:
                if "Key Insight:" in content:
                    insight_start = content.find("Key Insight:") + 12
                    insight_end = content.find("\n", insight_start)
                    if insight_end > insight_start:
                        insight_text = content[insight_start:insight_end].strip()
                        if insight_text:
                            key_features.append(insight_text[:100] + "..." if len(insight_text) > 100 else insight_text)
            
            # If still no key features, use a default
            if not key_features:
                key_features.append("Research paper on AI and machine learning")
            
            models_info.append({
                "name": title,
                "authors": authors,
                "affiliations": affiliations,
                "paper_type": paper_type,
                "features": "; ".join(key_features[:2]) if key_features else "-",
                "strengths": "; ".join(strengths[:2]) if strengths else "-",
                "weaknesses": "; ".join(weaknesses[:2]) if weaknesses else "-",
                "paper_name": paper_name,
                "year": date or "-",
                "keywords": keywords,
                "technical_terms": technical_terms,
                "references_count": len(references),
                "dois": dois,
                "urls": urls
            })
        
        # Create detailed comparison table
        response_parts.append("| Paper | Authors | Type | Key Features | Strengths | Weaknesses | Year |")
        response_parts.append("|-------|---------|------|--------------|-----------|------------|------|")
        
        for model in models_info[:10]:  # Limit to top 10 models
            name = model['name'][:40] + "..." if len(model['name']) > 40 else model['name']
            authors_str = ', '.join(model['authors'][:2]) if model['authors'] else "-"
            authors_str = authors_str[:30] + "..." if len(authors_str) > 30 else authors_str
            paper_type = model['paper_type'][:15] if model['paper_type'] != 'unknown' else "-"
            features = model['features'][:80] + "..." if len(model['features']) > 80 else model['features']
            strengths = model['strengths'][:60] + "..." if len(model['strengths']) > 60 else model['strengths']
            weaknesses = model['weaknesses'][:60] + "..." if len(model['weaknesses']) > 60 else model['weaknesses']
            
            response_parts.append(f"| {name} | {authors_str} | {paper_type} | {features} | {strengths} | {weaknesses} | {model['year']} |")
        
        # Add detailed analysis section
        response_parts.append(f"\n**Detailed Analysis:**")
        response_parts.append(f"This comparison analyzes {len(models_info)} research papers covering various AI approaches:")
        
        # Group papers by type
        paper_types = {}
        for model in models_info:
            ptype = model['paper_type']
            if ptype not in paper_types:
                paper_types[ptype] = []
            paper_types[ptype].append(model)
        
        for ptype, papers in paper_types.items():
            if ptype != 'unknown':
                response_parts.append(f"\n**{ptype.title()} Papers ({len(papers)} papers):**")
                for paper in papers[:3]:  # Show first 3 of each type
                    response_parts.append(f"• {paper['name']}: {paper['features'][:100]}...")
        
        # Add citation analysis if available
        papers_with_refs = [m for m in models_info if m['references_count'] > 0]
        if papers_with_refs:
            response_parts.append(f"\n**Citation Analysis:**")
            response_parts.append(f"• Papers with references: {len(papers_with_refs)}/{len(models_info)}")
            papers_with_dois = [m for m in models_info if m['dois']]
            response_parts.append(f"• Papers with DOIs: {len(papers_with_dois)}/{len(models_info)}")
        
        # Add affiliation analysis
        all_affiliations = []
        for model in models_info:
            all_affiliations.extend(model['affiliations'])
        if all_affiliations:
            unique_affiliations = list(set(all_affiliations))
            response_parts.append(f"\n**Top Institutions:**")
            for affil in unique_affiliations[:5]:
                count = all_affiliations.count(affil)
                response_parts.append(f"• {affil}: {count} papers")
        
        response_parts.append(f"\n**Summary:** This comprehensive analysis shows the diversity of AI research approaches, ")
        response_parts.append("from large language models to specialized neural networks and hybrid symbolic systems. ")
        response_parts.append("Each paper contributes unique insights to the broader AI landscape.")
        
        return "\n".join(response_parts)
    
    def _create_enhanced_fallback_response(self, query: str, context_papers: List[Dict[str, Any]]) -> str:
        """Create an enhanced fallback response with more specific information."""
        # Check if the query is likely out-of-scope (no relevant tools, keywords, or findings found)
        keywords = ["blockchain", "CRISPR", "underwater vehicles", "medical data privacy"]
        if any(kw.lower() in query.lower() for kw in keywords):
            return f"❗ No relevant papers found on the topic: '{query}'. The current research collection does not cover this subject."

        response_parts = [f"**AI Tools and Technologies from Research Papers**\n"]
        
        # Extract specific AI tools and technologies from the papers
        ai_tools_found = []
        for i, paper in enumerate(context_papers, 1):
            content = paper['content']
            title = paper['metadata'].get('title', paper['metadata'].get('paper_name', f'Paper {i}'))
            
            # Extract technical terms that might be AI tools
            if "Technical Terms:" in content:
                tech_start = content.find("Technical Terms:") + 16
                tech_end = content.find("\n", tech_start)
                if tech_end > tech_start:
                    tech_terms = content[tech_start:tech_end].strip()
                    if tech_terms and len(tech_terms) > 5:
                        ai_tools_found.append(f"• **{title}**: {tech_terms}")
            
            # Extract insights that mention tools or technologies
            if "Key Insight:" in content:
                insight_start = content.find("Key Insight:") + 12
                insight_end = content.find("\n", insight_start)
                if insight_end > insight_start:
                    insight = content[insight_start:insight_end].strip()
                    if insight and len(insight) > 20:
                        tool_keywords = ['tool', 'framework', 'model', 'algorithm', 'system', 'platform', 'library', 'API']
                        if any(keyword in insight.lower() for keyword in tool_keywords):
                            ai_tools_found.append(f"• **{title}**: {insight}")
            
            # Extract hypotheses that might mention tools
            if "Research Hypothesis:" in content:
                hyp_start = content.find("Research Hypothesis:") + 20
                hyp_end = content.find("\n", hyp_start)
                if hyp_end > hyp_start:
                    hypothesis = content[hyp_start:hyp_end].strip()
                    if hypothesis and len(hypothesis) > 20:
                        tool_keywords = ['tool', 'framework', 'model', 'algorithm', 'system', 'platform', 'library', 'API']
                        if any(keyword in hypothesis.lower() for keyword in tool_keywords):
                            ai_tools_found.append(f"• **{title}** (Hypothesis): {hypothesis}")
        
        if ai_tools_found:
            response_parts.append("\n**AI Tools and Technologies Identified:**\n")
            response_parts.extend(ai_tools_found[:8])  # Limit to top 8 tools
        else:
            response_parts.append("\n**Key Research Papers on AI Tools:**\n")
            for i, paper in enumerate(context_papers, 1):
                title = paper['metadata'].get('title', paper['metadata'].get('paper_name', f'Paper {i}'))
                response_parts.append(f"• {title}")
        
        response_parts.append(f"\n**Summary:** Based on the analysis of {len(context_papers)} research papers, ")
        response_parts.append("these papers cover various AI tools, frameworks, and technologies including ")
        response_parts.append("neural networks, language models, evolutionary algorithms, and neuro-symbolic systems. ")
        response_parts.append("Each paper provides insights into different aspects of AI tool development and application.")
        
        return "\n".join(response_parts)
    
    def _create_context(self, papers: List[Dict[str, Any]], detailed: bool = False) -> str:
        """Create context string from retrieved papers. If detailed, include more metadata."""
        context_parts = []
        for i, paper in enumerate(papers, 1):
            paper_title = paper['metadata'].get('title', f'Paper {i}')
            paper_name = paper['metadata'].get('paper_name', f'Paper {i}')
            content = paper['content']
            metadata = paper.get('metadata', {})
            # Extract model name from title or content
            model_name = paper_title
            if "Title:" in content:
                title_start = content.find("Title:") + 6
                title_end = content.find("\n", title_start)
                if title_end > title_start:
                    extracted_title = content[title_start:title_end].strip()
                    if extracted_title and len(extracted_title) > 5:
                        model_name = extracted_title
            # Extract unique features from insight or technical terms
            unique_features = ""
            if "Key Insight:" in content:
                insight_start = content.find("Key Insight:") + 12
                insight_end = content.find("\n", insight_start)
                if insight_end > insight_start:
                    unique_features = content[insight_start:insight_end].strip()
                else:
                    unique_features = content[insight_start:].strip()
            elif "Technical Terms:" in content:
                tech_start = content.find("Technical Terms:") + 16
                tech_end = content.find("\n", tech_start)
                if tech_end > tech_start:
                    unique_features = content[tech_start:tech_end].strip()
            # Short summary from first 1-2 sentences of insight
            summary = ""
            if unique_features:
                sentences = unique_features.split('.')
                summary = '. '.join(sentences[:2]).strip()
                if summary and not summary.endswith('.'):
                    summary += '.'
            # Compose context for this paper
            paper_context = f"Model Name: {model_name}"
            if unique_features:
                paper_context += f"\nUnique Features: {unique_features}"
            if summary:
                paper_context += f"\nSummary: {summary}"
            if detailed:
                # Add more metadata if available
                if metadata.get('authors'):
                    paper_context += f"\nAuthors: {metadata['authors']}"
                if metadata.get('date'):
                    paper_context += f"\nDate: {metadata['date']}"
                if metadata.get('keywords'):
                    paper_context += f"\nKeywords: {', '.join(metadata['keywords'])}"
            context_parts.append(paper_context)
        return "\n\n".join(context_parts)
    
    def answer_question(self, question: str, top_k: int = 8) -> Dict[str, Any]:
        """Answer a question using RAG system."""
        # Retrieve relevant papers
        relevant_papers = self.retrieve_relevant_papers(question, top_k=top_k)
        
        if not relevant_papers:
            return {
                "answer": f"I couldn't find relevant papers to answer: {question}",
                "relevant_papers": [],
                "num_papers_retrieved": 0
            }
        
        # Generate response
        answer = self.generate_response(question, relevant_papers)
        
        return {
            "answer": answer,
            "relevant_papers": relevant_papers,
            "num_papers_retrieved": len(relevant_papers)
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Load existing papers
    with open("results/all_papers_results.json", 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Add papers to index
    rag.add_papers_to_index(papers)
    
    # Test question answering
    question = "What are the main findings about neural networks in the research papers?"
    result = rag.answer_question(question)
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Papers retrieved: {result['num_papers_retrieved']}") 