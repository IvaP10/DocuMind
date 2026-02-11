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
        self.temperature = getattr(config, 'LLM_TEMPERATURE', 0.0)
        self.max_tokens = getattr(config, 'LLM_MAX_TOKENS', 2000)
    
    def generate_answer(self, query: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        context = context_data.get("context", "")
        sources = context_data.get("sources", [])
        retrieval_metrics = context_data.get("metrics", {})
        mode = context_data.get("mode", "hybrid_quality")
        
        if not context:
            return self._no_context_response()
        
        experiment = config.EXPERIMENT_MODES.get(mode, config.EXPERIMENT_MODES["hybrid_quality"])
        
        retrieval_confidence = self._calculate_retrieval_confidence(sources, retrieval_metrics)
        
        if experiment.get("abstention_enabled", False) and retrieval_confidence < config.ABSTENTION_THRESHOLD:
            return self._abstain_response(retrieval_confidence)
        
        answer = self._generate(query, context, mode, experiment)
        
        if mode == "fast_response":
            return self._fast_response(answer, sources, retrieval_confidence)
        
        if experiment.get("numeric_verification", False):
            numeric_verification = self._verify_numeric_accuracy(answer, context)
        else:
            numeric_verification = {"passed": True, "mismatches": []}
        
        verification = self._verify_answer(query, answer, context)
        
        if experiment.get("atomic_fact_verification", False):
            atomic_verification = self._atomic_fact_verification(answer, context, experiment)
        else:
            atomic_verification = {"support_rate": 1.0, "atomic_facts": []}
        
        citation_metrics = self._analyze_citations(answer, context, experiment)
        
        confidence = self._calculate_calibrated_confidence(
            verification,
            atomic_verification,
            len(sources),
            citation_metrics,
            retrieval_confidence,
            experiment,
            numeric_verification
        )
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "verified": verification["verified"] and numeric_verification["passed"],
            "citation_metrics": citation_metrics,
            "verification_reason": verification.get("reason", ""),
            "atomic_verification": atomic_verification,
            "retrieval_confidence": retrieval_confidence,
            "numeric_verification": numeric_verification
        }
    
    def _verify_numeric_accuracy(self, answer: str, context: str) -> Dict[str, Any]:
        answer_numbers = self._extract_numbers_from_text(answer)
        context_numbers = self._extract_numbers_from_text(context)
        
        if not answer_numbers:
            return {"passed": True, "mismatches": [], "answer_numbers": [], "context_numbers": context_numbers}
        
        mismatches = []
        for num in answer_numbers:
            if num not in context_numbers:
                mismatches.append(num)
        
        passed = len(mismatches) == 0
        
        return {
            "passed": passed,
            "mismatches": mismatches,
            "answer_numbers": answer_numbers,
            "context_numbers": context_numbers
        }
    
    def _extract_numbers_from_text(self, text: str) -> List[str]:
        numeric_patterns = [
            r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[KMBkmb](?:illion)?)?',
            r'-?\d+(?:\.\d+)?%',
            r'\b(?:19|20)\d{2}\b',
            r'\b(?:Q[1-4]|FY)\s*\d{2,4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        ]
        
        numbers = []
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        return list(set(numbers))
    
    def _fast_response(self, answer: str, sources: List[Dict], retrieval_confidence: float) -> Dict[str, Any]:
        return {
            "answer": answer,
            "sources": sources,
            "confidence": min(0.85, retrieval_confidence * 1.1),
            "verified": True,
            "citation_metrics": {
                "citation_recall": 1.0,
                "citation_precision": 1.0,
                "citation_f1": 1.0,
                "total_claims": 1,
                "supported_claims": 1,
                "unsupported_claims": [],
                "hallucinated_sentences": []
            },
            "verification_reason": "Fast mode - minimal verification",
            "atomic_verification": {"support_rate": 1.0, "atomic_facts": []},
            "retrieval_confidence": retrieval_confidence,
            "numeric_verification": {"passed": True, "mismatches": []}
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
            "retrieval_confidence": retrieval_confidence,
            "numeric_verification": {"passed": True, "mismatches": []}
        }
    
    def _generate(self, query: str, context: str, mode: str, experiment: Dict) -> str:
        system_prompt = self._build_system_prompt(mode, experiment)
        user_prompt = self._build_user_prompt(query, context, mode)
        
        try:
            answer = self._generate_openai(system_prompt, user_prompt)
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    def _build_system_prompt(self, mode: str, experiment: Dict) -> str:
        if mode == "fast_response":
            return """You are a document analyst. Answer questions using the provided context.

RULES:
1. Use only information from the context
2. Be concise and direct
3. Cite page numbers when available

ANSWER FORMAT:
- Brief, direct answer
- Key facts with page references"""
        
        return """You are a precise, rigorous document analyst. Your task is to answer questions using ONLY information explicitly stated in the provided context.

CRITICAL QUOTE-FIRST FORMAT:
For every factual claim you make, you MUST:
1. First provide the exact quote from the context in quotation marks
2. Then explain or paraphrase if needed
3. Always cite the page number

Example format:
"The company's revenue was $2.5 billion" (Page 3). This represents a 15% increase from the previous year.

RULES:
1. ALWAYS start each claim with a direct quote from context
2. Use exact quotes especially for ALL numbers, dates, names, and statistics
3. After the quote, provide the page citation in (Page X) format
4. Never infer, assume, or add information not in the context
5. If information is missing, state: "The context does not contain this information"
6. For numbers, copy them EXACTLY as they appear in the context
7. If sources conflict, present both with their page references

ANSWER FORMAT:
- Start with quoted facts from context
- Support every claim with explicit page citations
- Use direct quotes for all critical data points
- Be comprehensive when context allows"""
    
    def _build_user_prompt(self, query: str, context: str, mode: str) -> str:
        if mode == "fast_response":
            return f"""CONTEXT:
{context}

QUESTION: {query}

Provide a concise answer with page references.

ANSWER:"""
        
        return f"""CONTEXT:
{context}

QUESTION: {query}

IMPORTANT: Follow the quote-first format. Start each factual claim with a direct quote from the context, then cite the page number.

ANSWER:"""
    
    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
    
    def _verify_answer(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        try:
            verification_prompt = f"""Verify if this answer is fully supported by the context.

CONTEXT:
{context[:2000]}

ANSWER:
{answer}

Return JSON with:
- verified: true/false
- reason: brief explanation

JSON:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "user", "content": verification_prompt}
                ]
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            if "verified" not in result:
                result["verified"] = False
            if isinstance(result["verified"], str):
                result["verified"] = result["verified"].lower() in ["true", "yes"]
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return self._heuristic_verification(answer, context)
    
    def _heuristic_verification(self, answer: str, context: str) -> Dict[str, Any]:
        logger.warning("Using heuristic verification")
        
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'from', 'is', 'are', 'was', 'were', 'this', 'that'
        }
        
        answer_content_words = answer_words - stopwords
        matches = len(answer_content_words & context_words)
        
        if not answer_content_words:
            return {"verified": True, "reason": "No content words to verify"}
        
        overlap = matches / len(answer_content_words)
        
        if overlap > 0.7:
            return {"verified": True, "reason": f"High word overlap: {overlap:.2%}"}
        else:
            return {"verified": False, "reason": f"Low word overlap: {overlap:.2%}"}
    
    def _atomic_fact_verification(self, answer: str, context: str, experiment: Dict) -> Dict[str, Any]:
        try:
            atomic_prompt = f"""Extract atomic facts from this answer and verify each against the context.

CONTEXT:
{context[:2000]}

ANSWER:
{answer}

Return JSON with:
- facts: list of atomic facts
- supported: list of booleans (one per fact)

JSON:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=500,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "user", "content": atomic_prompt}
                ]
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            facts = result.get("facts", [])
            supported = result.get("supported", [])
            
            if len(supported) == len(facts):
                support_rate = sum(supported) / len(supported) if supported else 1.0
            else:
                support_rate = 1.0
            
            return {
                "support_rate": support_rate,
                "atomic_facts": facts,
                "supported": supported
            }
            
        except Exception as e:
            logger.error(f"Atomic fact verification failed: {e}")
            return {"support_rate": 1.0, "atomic_facts": []}
    
    def _analyze_citations(self, answer: str, context: str, experiment: Dict) -> Dict[str, Any]:
        claims = self._extract_claims(answer)
        
        if not claims:
            return {
                "citation_recall": 1.0,
                "citation_precision": 1.0,
                "citation_f1": 1.0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": [],
                "hallucinated_sentences": []
            }
        
        supported = []
        unsupported = []
        hallucinated = []
        
        for claim in claims:
            if self._claim_has_citation(claim):
                if self._verify_claim_in_context(claim, context):
                    supported.append(claim)
                else:
                    hallucinated.append(claim)
            else:
                unsupported.append(claim)
        
        total_claims = len(claims)
        supported_count = len(supported)
        cited_count = supported_count + len(hallucinated)
        
        citation_recall = supported_count / total_claims if total_claims > 0 else 1.0
        citation_precision = supported_count / cited_count if cited_count > 0 else 1.0
        
        citation_f1 = (2 * citation_precision * citation_recall) / (citation_precision + citation_recall) if (citation_precision + citation_recall) > 0 else 0.0
        
        if experiment.get("hallucination_penalty", 0) > 0 and hallucinated:
            penalty = experiment["hallucination_penalty"]
            citation_recall *= penalty
            citation_precision *= penalty
            citation_f1 *= penalty
        
        return {
            "citation_recall": citation_recall,
            "citation_precision": citation_precision,
            "citation_f1": citation_f1,
            "total_claims": total_claims,
            "supported_claims": supported_count,
            "unsupported_claims": unsupported,
            "hallucinated_sentences": hallucinated
        }
    
    def _extract_claims(self, answer: str) -> List[str]:
        sentences = self._split_sentences(answer)
        return [s for s in sentences if self._is_factual_claim(s)]
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_factual_claim(self, sentence: str) -> bool:
        sentence_clean = sentence.lower().strip()
        
        if len(sentence_clean.split()) < 4:
            return False
        
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(sentence_clean.startswith(w) for w in question_words):
            return False
        
        if sentence_clean.endswith('?'):
            return False
        
        weak_phrases = ['i think', 'i believe', 'perhaps', 'maybe', 'possibly']
        if any(phrase in sentence_clean for phrase in weak_phrases):
            return False
        
        return True
    
    def _claim_has_citation(self, claim: str) -> bool:
        page_patterns = [
            r'\[page\s+\d+\]',
            r'\(page\s+\d+\)',
            r'page\s+\d+',
            r'p\.\s*\d+',
            r'\[p\.\s*\d+\]'
        ]
        
        return any(re.search(pattern, claim.lower()) for pattern in page_patterns)
    
    def _verify_claim_in_context(self, claim: str, context: str) -> bool:
        claim_clean = re.sub(r'\[.*?\]', '', claim).lower()
        claim_clean = re.sub(r'\(.*?\)', '', claim_clean)
        claim_clean = re.sub(r'(page \d+|p\.\s*\d+)', '', claim_clean)
        
        claim_words = [w for w in claim_clean.split() if len(w) > 3]
        
        if not claim_words:
            return True
        
        context_lower = context.lower()
        matches = sum(1 for word in claim_words if word in context_lower)
        
        support_ratio = matches / len(claim_words)
        return support_ratio > 0.65
    
    def _calculate_calibrated_confidence(
        self,
        verification: Dict[str, Any],
        atomic_verification: Dict[str, Any],
        num_sources: int,
        citation_metrics: Dict[str, Any],
        retrieval_confidence: float,
        experiment: Dict,
        numeric_verification: Dict[str, Any]
    ) -> float:
        
        if not config.CONFIDENCE_CALIBRATION.get("use_calibration", True):
            return 0.8 if verification.get("verified", False) else 0.4
        
        weights = config.CONFIDENCE_CALIBRATION
        
        verification_score = 1.0 if verification.get("verified", False) else 0.3
        
        source_quality = min(math.log(num_sources + 1) / math.log(10), 1.0)
        
        citation_score = citation_metrics.get("citation_f1", 0.0)
        
        confidence = (
            weights.get("verification_weight", 0.3) * verification_score +
            weights.get("source_quality_weight", 0.2) * source_quality +
            weights.get("citation_quality_weight", 0.4) * citation_score +
            weights.get("retrieval_confidence_weight", 0.1) * retrieval_confidence
        )
        
        atomic_support_rate = atomic_verification.get("support_rate", 1.0)
        confidence *= atomic_support_rate
        
        if not numeric_verification.get("passed", True):
            confidence *= 0.7
        
        hallucinated = citation_metrics.get("hallucinated_sentences", [])
        if hallucinated:
            hallucination_penalty = experiment.get("hallucination_penalty", 0.95)
            confidence *= (hallucination_penalty ** len(hallucinated))
        
        return max(0.0, min(1.0, confidence))
    
    def _no_context_response(self) -> Dict[str, Any]:
        return {
            "answer": "No relevant context found to answer this question.",
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
            "retrieval_confidence": 0.0,
            "numeric_verification": {"passed": True, "mismatches": []}
        }


generator = EnhancedGenerator()
