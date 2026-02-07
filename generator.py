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
        logger.info("=" * 80)
        logger.info("GENERATING ANSWER")
        logger.info("=" * 80)
        
        context = context_data.get("context", "")
        sources = context_data.get("sources", [])
        
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"Context length: {len(context)} chars, Sources: {len(sources)}")
        
        if not context:
            logger.warning("No context available")
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
        logger.info(f"Generated answer: {answer[:150]}...")
        
        verification = self.verify_answer(query, answer, context)
        logger.info(f"Verification result: {verification}")
        
        citation_metrics = self.analyze_citations(answer, context)
        logger.info(f"Citation metrics: {citation_metrics}")
        
        confidence = self.calculate_confidence(verification, len(sources), citation_metrics)
        logger.info(f"Final confidence: {confidence:.2%}")
        
        logger.info("=" * 80)
        
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
                max_completion_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            logger.info("Answer generated")
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    def verify_answer(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        # Truncate context if too long to avoid token limits
        max_context_chars = 2000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "... [truncated]"
        
        # Simpler prompt that's more likely to produce valid JSON
        verification_prompt = f"""Context: {context}

Question: {query}

Answer: {answer}

Is the answer fully supported by the context? Respond ONLY with valid JSON:
{{"verified": true, "reason": "supported"}} or {{"verified": false, "reason": "not supported"}}"""

        try:
            logger.info(f"Verifying answer with model: {self.model}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You respond ONLY with valid JSON. No markdown, no explanation, just JSON."},
                    {"role": "user", "content": verification_prompt}
                ],
                max_completion_tokens=100,
                temperature=0.0,
                response_format={"type": "json_object"}  # Force JSON mode
            )
            
            verification_text = response.choices[0].message.content.strip()
            logger.info(f"Raw verification response: {verification_text}")
            
            # Try parsing directly first
            try:
                verification_result = json.loads(verification_text)
            except:
                # Fallback: extract JSON from markdown
                if "```json" in verification_text:
                    verification_text = verification_text.split("```json")[1].split("```")[0].strip()
                elif "```" in verification_text:
                    verification_text = verification_text.split("```")[1].split("```")[0].strip()
                
                # Find JSON boundaries
                start_idx = verification_text.find("{")
                end_idx = verification_text.rfind("}") + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    verification_text = verification_text[start_idx:end_idx]
                
                verification_result = json.loads(verification_text)
            
            # Validate and normalize
            if "verified" not in verification_result:
                # Try alternate key names
                if "is_verified" in verification_result:
                    verification_result["verified"] = verification_result["is_verified"]
                elif "valid" in verification_result:
                    verification_result["verified"] = verification_result["valid"]
                else:
                    raise ValueError("Missing 'verified' key in response")
            
            # Ensure boolean type (handle string "true"/"false")
            verified_value = verification_result["verified"]
            if isinstance(verified_value, str):
                verification_result["verified"] = verified_value.lower() in ["true", "yes", "1"]
            else:
                verification_result["verified"] = bool(verified_value)
            
            # Ensure reason exists
            if "reason" not in verification_result:
                verification_result["reason"] = "verified" if verification_result["verified"] else "not verified"
            
            logger.info(f"✓ Verification successful: {verification_result['verified']}")
            return verification_result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse failed: {str(e)}")
            logger.error(f"Raw text: {verification_text}")
            # Fallback: use heuristic
            return self._heuristic_verification(answer, context)
            
        except Exception as e:
            logger.error(f"❌ Verification API error: {str(e)}")
            # Fallback: use heuristic
            return self._heuristic_verification(answer, context)
    
    def _heuristic_verification(self, answer: str, context: str) -> Dict[str, Any]:
        """Fallback verification using simple heuristics"""
        logger.warning("Using heuristic verification fallback")
        
        # Simple word overlap check
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'}
        answer_words = answer_words - stopwords
        context_words = context_words - stopwords
        
        if not answer_words:
            return {"verified": True, "reason": "No specific claims to verify"}
        
        # Calculate overlap
        overlap = len(answer_words & context_words)
        overlap_ratio = overlap / len(answer_words) if answer_words else 0
        
        # If >60% of content words appear in context, consider verified
        is_verified = overlap_ratio > 0.6
        
        logger.info(f"Heuristic verification: {is_verified} (overlap: {overlap_ratio:.2%})")
        
        return {
            "verified": is_verified,
            "reason": f"Heuristic check: {overlap_ratio:.0%} word overlap"
        }
    
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
        """Calculate confidence score based on multiple factors"""
        
        confidence = 0.0
        
        # Verification is most important (50% weight)
        if verification.get("verified", False):
            confidence += 0.50
        else:
            # Even if not verified, give some credit if reason is heuristic
            if "Heuristic" in verification.get("reason", ""):
                confidence += 0.25
        
        # Source count (20% weight)
        if num_sources > 0:
            # More sources = more confidence, up to 5 sources
            source_score = min(num_sources / 5.0, 1.0) * 0.20
            confidence += source_score
        
        # Citation quality (30% weight)
        citation_recall = citation_metrics.get("citation_recall", 0.0)
        citation_precision = citation_metrics.get("citation_precision", 1.0)
        
        # Average of recall and precision
        citation_score = ((citation_recall + citation_precision) / 2.0) * 0.30
        confidence += citation_score
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        logger.info(f"Confidence breakdown: verified={verification.get('verified')}, sources={num_sources}, citation_recall={citation_recall:.2f}, final={confidence:.2f}")
        
        return confidence


generator = Generator()