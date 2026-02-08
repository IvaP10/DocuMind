from typing import Dict, Any, List, Optional
from openai import OpenAI
import config
import logging
import json
import re
import math

logger = logging.getLogger(__name__)


class EnhancedGenerator:
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.provider = "openai"
        self.model = "gpt-4o-mini"
        self.temperature = getattr(config, 'LLM_TEMPERATURE', 0.3)
        self.max_tokens = getattr(config, 'LLM_MAX_TOKENS', 1000)
        
        logger.info(f"Generator initialized: {self.provider}/{self.model}")
    
    def generate_answer(self, query: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("GENERATING ANSWER")
        logger.info("=" * 80)
        
        context = context_data.get("context", "")
        sources = context_data.get("sources", [])
        retrieval_metrics = context_data.get("metrics", {})
        
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"Context: {len(context)} chars, {len(sources)} sources")
        
        if not context:
            logger.warning("No context available")
            return self._no_context_response()
        
        retrieval_confidence = self._calculate_retrieval_confidence(sources, retrieval_metrics)
        logger.info(f"Retrieval confidence: {retrieval_confidence:.2%}")
        
        if retrieval_confidence < config.ABSTENTION_THRESHOLD:
            logger.warning(f"Retrieval confidence {retrieval_confidence:.2%} below threshold {config.ABSTENTION_THRESHOLD:.2%}")
            return self._abstain_response(retrieval_confidence)
        
        answer = self._generate(query, context)
        logger.info(f"Generated answer: {len(answer)} chars")
        
        verification = self._verify_answer(query, answer, context)
        logger.info(f"Verification: {verification['verified']}")
        
        if config.ATOMIC_FACT_VERIFICATION:
            atomic_verification = self._atomic_fact_verification(answer, context)
            logger.info(f"Atomic verification: {atomic_verification['support_rate']:.2%}")
        else:
            atomic_verification = {"support_rate": 1.0, "atomic_facts": []}
        
        citation_metrics = self._analyze_citations(answer, context)
        logger.info(f"Citation metrics: recall={citation_metrics['citation_recall']:.2f}, "
                   f"precision={citation_metrics['citation_precision']:.2f}")
        
        confidence = self._calculate_calibrated_confidence(
            verification,
            atomic_verification,
            len(sources),
            citation_metrics,
            retrieval_confidence
        )
        logger.info(f"Calibrated confidence: {confidence:.2%}")
        
        logger.info("=" * 80)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "verified": verification["verified"],
            "citation_metrics": citation_metrics,
            "verification_reason": verification.get("reason", ""),
            "atomic_verification": atomic_verification,
            "retrieval_confidence": retrieval_confidence
        }
    
    def _calculate_retrieval_confidence(
        self,
        sources: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> float:
        if not sources:
            return 0.0
        
        avg_score = metrics.get("avg_rerank_score", 0.0)
        max_score = metrics.get("max_rerank_score", 0.0)
        num_sources = len(sources)
        
        score_component = (avg_score + max_score) / 2
        
        source_count_component = min(math.log(num_sources + 1) / math.log(8), 1.0)
        
        confidence = 0.7 * score_component + 0.3 * source_count_component
        
        return max(0.0, min(1.0, confidence))
    
    def _abstain_response(self, retrieval_confidence: float) -> Dict[str, Any]:
        return {
            "answer": "I cannot find sufficient reliable information in the document to answer this question confidently.",
            "sources": [],
            "confidence": retrieval_confidence,
            "verified": False,
            "citation_metrics": {
                "citation_recall": 0.0,
                "citation_precision": 0.0,
                "citation_f1": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": [],
                "hallucinated_sentences": []
            },
            "verification_reason": "Abstained due to low retrieval confidence",
            "atomic_verification": {"support_rate": 0.0, "atomic_facts": []},
            "retrieval_confidence": retrieval_confidence
        }
    
    def _generate(self, query: str, context: str) -> str:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)
        
        try:
            answer = self._generate_openai(system_prompt, user_prompt)
            logger.info("Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    def _build_system_prompt(self) -> str:
        return """You are a precise document analyst. Your task is to answer questions using ONLY information from the provided context.

CRITICAL RULES:
1. Use ONLY facts explicitly stated in the context
2. Always cite page numbers for claims
3. If information is not in the context, clearly state: "The provided context does not contain this information"
4. Never infer, assume, or add external knowledge
5. Quote directly for important facts
6. If sources conflict, present both perspectives
7. Be comprehensive when context allows

ANSWER FORMAT:
- Direct answer to the question
- Supporting evidence with page references
- Relevant quotes when appropriate
- Clear statement if information is missing"""
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        return f"""CONTEXT FROM DOCUMENT:
{context}

QUESTION: {query}

Provide a precise, evidence-based answer using only the information above. Cite page numbers for all claims.

ANSWER:"""
    
    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    def _atomic_fact_verification(self, answer: str, context: str) -> Dict[str, Any]:
        atomic_facts = self._extract_atomic_facts(answer)
        
        if not atomic_facts:
            return {
                "support_rate": 1.0,
                "atomic_facts": [],
                "supported": [],
                "unsupported": []
            }
        
        supported = []
        unsupported = []
        
        for fact in atomic_facts:
            if self._verify_atomic_fact(fact, context):
                supported.append(fact)
            else:
                unsupported.append(fact)
        
        support_rate = len(supported) / len(atomic_facts) if atomic_facts else 1.0
        
        logger.info(f"Atomic fact verification: {len(supported)}/{len(atomic_facts)} facts supported ({support_rate:.2%})")
        
        return {
            "support_rate": support_rate,
            "atomic_facts": atomic_facts,
            "supported": supported,
            "unsupported": unsupported,
            "total_facts": len(atomic_facts),
            "supported_count": len(supported)
        }
    
    def _extract_atomic_facts(self, answer: str) -> List[str]:
        sentences = self._split_sentences(answer)
        
        atomic_facts = []
        for sentence in sentences:
            if self._is_factual_claim(sentence):
                facts = self._break_into_atomic_claims(sentence)
                atomic_facts.extend(facts)
        
        return atomic_facts
    
    def _break_into_atomic_claims(self, sentence: str) -> List[str]:
        if "and" in sentence.lower() or "," in sentence:
            parts = re.split(r'\s+and\s+|,\s+', sentence)
            parts = [p.strip() for p in parts if p.strip() and len(p.split()) > 3]
            
            if len(parts) > 1:
                return parts
        
        return [sentence]
    
    def _verify_atomic_fact(self, fact: str, context: str) -> bool:
        fact_clean = re.sub(r'\[.*?\]', '', fact).lower()
        fact_clean = re.sub(r'(page \d+|p\.\s*\d+)', '', fact_clean)
        
        words = [w for w in fact_clean.split() if len(w) > 3]
        
        if not words:
            return True
        
        context_lower = context.lower()
        matches = sum(1 for word in words if word in context_lower)
        
        support_ratio = matches / len(words)
        return support_ratio > 0.6
    
    def _verify_answer(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        max_context_chars = 3000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "... [truncated]"
        
        verification_prompt = f"""Context: {context}

Question: {query}

Answer: {answer}

Is this answer fully supported by the context? Respond with JSON only:
{{"verified": true, "reason": "explanation"}}
OR
{{"verified": false, "reason": "what's unsupported"}}"""
        
        try:
            result = self._verify_openai(verification_prompt)
            return result
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return self._heuristic_verification(answer, context)
    
    def _verify_openai(self, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Respond ONLY with valid JSON. No markdown, no explanation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        if "verified" not in result:
            result["verified"] = False
        if isinstance(result["verified"], str):
            result["verified"] = result["verified"].lower() in ["true", "yes"]
        
        return result
    
    def _heuristic_verification(self, answer: str, context: str) -> Dict[str, Any]:
        logger.warning("Using heuristic verification")
        
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'
        }
        answer_words -= stopwords
        context_words -= stopwords
        
        if not answer_words:
            return {"verified": True, "reason": "No specific claims"}
        
        overlap = len(answer_words & context_words)
        overlap_ratio = overlap / len(answer_words)
        
        verified = overlap_ratio > 0.65
        
        return {
            "verified": verified,
            "reason": f"Heuristic: {overlap_ratio:.0%} word overlap"
        }
    
    def _analyze_citations(self, answer: str, context: str) -> Dict[str, Any]:
        sentences = self._split_sentences(answer)
        
        total_claims = len([s for s in sentences if self._is_factual_claim(s)])
        
        if total_claims == 0:
            return {
                "citation_recall": 1.0,
                "citation_precision": 1.0,
                "citation_f1": 1.0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": [],
                "hallucinated_sentences": []
            }
        
        supported_claims = 0
        unsupported_claims = []
        hallucinated = []
        
        for sentence in sentences:
            if not self._is_factual_claim(sentence):
                continue
            
            if self._check_sentence_support(sentence, context):
                supported_claims += 1
            else:
                unsupported_claims.append(sentence)
                if self._is_likely_hallucination(sentence, context):
                    hallucinated.append(sentence)
        
        citation_recall = supported_claims / total_claims if total_claims > 0 else 0.0
        
        total_citations = len([s for s in sentences if self._has_citation(s)])
        valid_citations = len([s for s in sentences if self._has_citation(s) and self._check_sentence_support(s, context)])
        citation_precision = valid_citations / total_citations if total_citations > 0 else 1.0
        
        if citation_recall + citation_precision > 0:
            citation_f1 = 2 * (citation_recall * citation_precision) / (citation_recall + citation_precision)
        else:
            citation_f1 = 0.0
        
        return {
            "citation_recall": citation_recall,
            "citation_precision": citation_precision,
            "citation_f1": citation_f1,
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "hallucinated_sentences": hallucinated
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Inc|Ltd|Co|vs|etc|e\.g|i\.e)\.\s',
                     r'\1<PERIOD> ', text, flags=re.IGNORECASE)
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_factual_claim(self, sentence: str) -> bool:
        non_claim_patterns = [
            r'(cannot|unable to|do not|does not)\s+(find|answer|contain)',
            r'^(However|Therefore|Thus|In conclusion)',
            r'(may|might|could|possibly|perhaps)',
        ]
        
        for pattern in non_claim_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return False
        
        if len(sentence.split()) < 5:
            return False
        
        return True
    
    def _has_citation(self, sentence: str) -> bool:
        return bool(re.search(r'(page \d+|p\.\s*\d+|\[.*?\])', sentence, re.IGNORECASE))
    
    def _check_sentence_support(self, sentence: str, context: str) -> bool:
        sentence_clean = re.sub(r'\[.*?\]', '', sentence).lower()
        sentence_clean = re.sub(r'(page \d+|p\.\s*\d+)', '', sentence_clean)
        
        words = sentence_clean.split()
        key_words = [w for w in words if len(w) > 3]
        
        if not key_words:
            return True
        
        context_lower = context.lower()
        matches = sum(1 for word in key_words if word in context_lower)
        
        support_ratio = matches / len(key_words)
        return support_ratio > 0.5
    
    def _is_likely_hallucination(self, sentence: str, context: str) -> bool:
        sentence_clean = re.sub(r'\[.*?\]', '', sentence).lower()
        words = [w for w in sentence_clean.split() if len(w) > 3]
        
        if not words:
            return False
        
        context_lower = context.lower()
        matches = sum(1 for word in words if word in context_lower)
        match_ratio = matches / len(words)
        
        return match_ratio < 0.25
    
    def _calculate_calibrated_confidence(
        self,
        verification: Dict[str, Any],
        atomic_verification: Dict[str, Any],
        num_sources: int,
        citation_metrics: Dict[str, Any],
        retrieval_confidence: float
    ) -> float:
        if not config.CONFIDENCE_CALIBRATION.get("use_calibration", True):
            return self._calculate_confidence_legacy(verification, num_sources, citation_metrics)
        
        cal = config.CONFIDENCE_CALIBRATION
        
        confidence = 0.0
        
        verification_component = 0.0
        if verification.get("verified", False):
            verification_component = 1.0
        elif "Heuristic" in verification.get("reason", ""):
            verification_component = 0.5
        
        atomic_support = atomic_verification.get("support_rate", 1.0)
        verification_score = 0.6 * verification_component + 0.4 * atomic_support
        confidence += cal["verification_weight"] * verification_score
        
        if num_sources > 0:
            source_score = min(math.log(num_sources + 1) / math.log(8), 1.0)
            confidence += cal["source_quality_weight"] * source_score
        
        citation_f1 = citation_metrics.get("citation_f1", 0.0)
        confidence += cal["citation_quality_weight"] * citation_f1
        
        confidence += cal["retrieval_confidence_weight"] * retrieval_confidence
        
        hallucinated = citation_metrics.get("hallucinated_sentences", [])
        if len(hallucinated) > 0:
            penalty = min(len(hallucinated) * 0.20, 0.50)
            confidence = confidence * (1 - penalty)
        
        unsupported_atomic = atomic_verification.get("unsupported", [])
        if len(unsupported_atomic) > 2:
            penalty = min((len(unsupported_atomic) - 2) * 0.10, 0.30)
            confidence = confidence * (1 - penalty)
        
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _calculate_confidence_legacy(
        self,
        verification: Dict[str, Any],
        num_sources: int,
        citation_metrics: Dict[str, Any]
    ) -> float:
        confidence = 0.0
        
        if verification.get("verified", False):
            confidence += 0.40
        elif "Heuristic" in verification.get("reason", ""):
            confidence += 0.20
        
        if num_sources > 0:
            source_score = min(math.log(num_sources + 1) / math.log(8), 1.0) * 0.25
            confidence += source_score
        
        citation_f1 = citation_metrics.get("citation_f1", 0.0)
        confidence += citation_f1 * 0.35
        
        hallucinated = citation_metrics.get("hallucinated_sentences", [])
        if len(hallucinated) > 0:
            penalty = min(len(hallucinated) * 0.15, 0.40)
            confidence = confidence * (1 - penalty)
        
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _no_context_response(self) -> Dict[str, Any]:
        return {
            "answer": "I cannot find relevant information in the document to answer this question.",
            "sources": [],
            "confidence": 0.0,
            "verified": False,
            "citation_metrics": {
                "citation_recall": 0.0,
                "citation_precision": 0.0,
                "citation_f1": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": [],
                "hallucinated_sentences": []
            },
            "verification_reason": "No context available",
            "atomic_verification": {"support_rate": 0.0, "atomic_facts": []},
            "retrieval_confidence": 0.0
        }


generator = EnhancedGenerator()
