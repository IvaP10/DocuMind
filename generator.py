from typing import Dict, Any, List
from openai import OpenAI
import config
import logging
import json
import re

logger = logging.getLogger(__name__)


class Generator:
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS
    
    def generate_answer(self, query: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Generating answer")
        
        context = context_data.get("context", "")
        sources = context_data.get("sources", [])
        
        if not context:
            return {
                "answer": "I cannot find relevant information in the document to answer this question.",
                "sources": [],
                "confidence": 0.0,
                "verified": False,
                "citation_metrics": {
                    "citation_recall": 0.0,
                    "citation_precision": 0.0,
                    "unsupported_claims": [],
                    "hallucinated_sentences": []
                }
            }
        
        answer = self.generate(query, context)
        
        verification = self.verify_answer(query, answer, context)
        citation_metrics = self.analyze_citations(answer, context)
        
        confidence = self.calculate_confidence(verification, len(sources), citation_metrics)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "verified": verification["verified"],
            "citation_metrics": citation_metrics,
            "verification_reason": verification.get("reason", "")
        }
    
    def generate(self, query: str, context: str) -> str:
        system_prompt = """You are a precise document analysis assistant. 
Your task is to answer questions based ONLY on the provided context.

Rules:
1. Only use information from the provided context
2. Cite specific page numbers when relevant
3. If the context doesn't contain the answer, say so clearly
4. Be concise and factual
5. Never make assumptions or add external knowledge"""

        user_prompt = f"""Context from document:
{context}

Question: {query}

Answer the question using ONLY the information above. Include page references where possible."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            logger.info("Answer generated")
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    def verify_answer(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        verification_prompt = f"""You are a fact-checker. Verify if the answer is supported by the context.

Context:
{context}

Question: {query}

Answer: {answer}

Is this answer:
1. Fully supported by the context? (yes/no)
2. Contains any information not in the context? (yes/no)
3. Contradicts the context? (yes/no)

Respond with ONLY a JSON object:
{{"verified": true/false, "reason": "brief explanation"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            
            verification_text = response.choices[0].message.content
            verification_result = json.loads(verification_text)
            
            logger.info(f"Verification: {verification_result.get('verified', False)}")
            return verification_result
            
        except Exception as e:
            logger.warning(f"Verification error: {str(e)}")
            return {"verified": False, "reason": "Verification failed"}
    
    def analyze_citations(self, answer: str, context: str) -> Dict[str, Any]:
        logger.info("Analyzing citations")
        
        sentences = self.split_sentences(answer)
        
        total_claims = len([s for s in sentences if self.is_factual_claim(s)])
        if total_claims == 0:
            return {
                "citation_recall": 1.0,
                "citation_precision": 1.0,
                "unsupported_claims": [],
                "hallucinated_sentences": []
            }
        
        supported_claims = 0
        unsupported_claims = []
        hallucinated_sentences = []
        
        for sentence in sentences:
            if not self.is_factual_claim(sentence):
                continue
            
            is_supported = self.check_sentence_support(sentence, context)
            
            if is_supported:
                supported_claims += 1
            else:
                unsupported_claims.append(sentence)
                if self.is_likely_hallucination(sentence, context):
                    hallucinated_sentences.append(sentence)
        
        citation_recall = supported_claims / total_claims if total_claims > 0 else 0.0
        
        total_citations = len([s for s in sentences if "page" in s.lower() or "[" in s])
        valid_citations = sum(1 for s in sentences if self.check_sentence_support(s, context))
        citation_precision = valid_citations / total_citations if total_citations > 0 else 1.0
        
        return {
            "citation_recall": citation_recall,
            "citation_precision": citation_precision,
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "hallucinated_sentences": hallucinated_sentences
        }
    
    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def is_factual_claim(self, sentence: str) -> bool:
        non_claim_patterns = [
            r'^(I|The document|According to)',
            r'(cannot|unable to|do not|does not)\s+(find|answer|contain)',
            r'^(However|Therefore|Thus|In conclusion)',
        ]
        
        for pattern in non_claim_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return False
        
        return len(sentence.split()) > 3
    
    def check_sentence_support(self, sentence: str, context: str) -> bool:
        sentence_clean = re.sub(r'\[.*?\]', '', sentence).lower()
        sentence_clean = re.sub(r'page \d+', '', sentence_clean)
        
        words = sentence_clean.split()
        key_words = [w for w in words if len(w) > 4]
        
        if not key_words:
            return True
        
        context_lower = context.lower()
        matches = sum(1 for word in key_words if word in context_lower)
        
        return matches / len(key_words) > 0.5
    
    def is_likely_hallucination(self, sentence: str, context: str) -> bool:
        sentence_clean = re.sub(r'\[.*?\]', '', sentence).lower()
        context_lower = context.lower()
        
        words = sentence_clean.split()
        key_words = [w for w in words if len(w) > 4]
        
        if not key_words:
            return False
        
        matches = sum(1 for word in key_words if word in context_lower)
        
        return matches / len(key_words) < 0.3
    
    def calculate_confidence(
        self, 
        verification: Dict[str, Any], 
        num_sources: int,
        citation_metrics: Dict[str, Any]
    ) -> float:
        
        base_confidence = 0.3
        
        if verification.get("verified", False):
            base_confidence += 0.3
        
        if num_sources > 0:
            source_boost = min(0.2, num_sources * 0.04)
            base_confidence += source_boost
        
        citation_recall = citation_metrics.get("citation_recall", 0.0)
        base_confidence += citation_recall * 0.2
        
        return min(1.0, base_confidence)


generator = Generator()