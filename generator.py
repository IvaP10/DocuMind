from typing import Dict, Any, List
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
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS

    def generate_answer(self, query: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        context = context_data.get("context", "")
        sources = context_data.get("sources", [])
        metrics = context_data.get("metrics", {})

        if not context:
            return self._no_context_response()

        retrieval_confidence = self._calc_retrieval_confidence(sources, metrics)

        answer = self._generate(query, context)

        numeric_verification = self._verify_numeric_accuracy(answer, context)
        combined_verification = self._combined_verification(query, answer, context)

        verification = combined_verification.get("answer_verification", {"verified": True, "reason": "default"})
        atomic_verification = combined_verification.get("atomic_verification", {"support_rate": 1.0, "atomic_facts": [], "supported": []})
        citation_metrics = self._analyze_citations(answer, context)

        confidence = self._calc_confidence(verification, atomic_verification, len(sources), citation_metrics, retrieval_confidence, numeric_verification)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "verified": verification.get("verified", True) and numeric_verification["passed"],
            "citation_metrics": citation_metrics,
            "verification_reason": verification.get("reason", ""),
            "atomic_verification": atomic_verification,
            "retrieval_confidence": retrieval_confidence,
            "numeric_verification": numeric_verification,
        }

    def _generate(self, query: str, context: str) -> str:
        system = """You are a precise document analyst. Answer questions using ONLY information from the provided context.

MANDATORY FORMAT RULES:
1. Every single sentence in your answer MUST end with a page citation in (Page X) format
2. For factual claims, include the exact quote from context in quotation marks before the citation
3. Use exact numbers, dates, names, and statistics as they appear in the context
4. Never infer or add information not in the context
5. If information is missing, state: "The context does not contain this information"
6. If sources conflict, present both with their page references

Example format:
"The company's revenue was $2.5 billion" (Page 3). This represents a significant increase over the prior year (Page 3)."""

        user = f"""CONTEXT:
{context}

QUESTION: {query}

IMPORTANT: Every sentence MUST include a (Page X) citation. No sentence should lack a page reference. Use direct quotes for key facts.

ANSWER:"""

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return resp.choices[0].message.content.strip()

    def _combined_verification(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=600,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": f"""Perform two tasks on this answer:

TASK 1 - Answer Verification: Is the answer fully supported by the context?
TASK 2 - Atomic Fact Verification: Extract each atomic fact from the answer and check if it is supported by the context.

CONTEXT:
{context[:2500]}

ANSWER:
{answer}

Return JSON with:
- answer_verification: {{"verified": true/false, "reason": "brief explanation"}}
- atomic_verification: {{"facts": ["fact1", "fact2", ...], "supported": [true, false, ...]}}

JSON:"""}]
            )
            result = json.loads(resp.choices[0].message.content.strip())

            av = result.get("answer_verification", {})
            if "verified" not in av:
                av["verified"] = True
            if isinstance(av.get("verified"), str):
                av["verified"] = av["verified"].lower() in ["true", "yes"]

            atv = result.get("atomic_verification", {})
            facts = atv.get("facts", [])
            supported = atv.get("supported", [])
            support_rate = sum(supported) / len(supported) if supported and len(supported) == len(facts) else 1.0
            atv["support_rate"] = support_rate

            return {"answer_verification": av, "atomic_verification": atv}
        except Exception:
            heuristic = self._heuristic_verification(answer, context)
            return {
                "answer_verification": heuristic,
                "atomic_verification": {"support_rate": 1.0, "atomic_facts": [], "supported": []}
            }

    def _heuristic_verification(self, answer: str, context: str) -> Dict[str, Any]:
        stopwords = {'the','a','an','and','or','but','in','on','at','to','for','of','with','from','is','are','was','were','this','that'}
        aw = set(answer.lower().split()) - stopwords
        cw = set(context.lower().split())
        if not aw:
            return {"verified": True, "reason": "No content words"}
        overlap = len(aw & cw) / len(aw)
        return {"verified": overlap > 0.7, "reason": f"Word overlap: {overlap:.2%}"}

    def _verify_numeric_accuracy(self, answer: str, context: str) -> Dict[str, Any]:
        answer_nums = self._extract_numbers(answer)
        context_nums = self._extract_numbers(context)
        if not answer_nums:
            return {"passed": True, "mismatches": []}
        mismatches = [n for n in answer_nums if n not in context_nums]
        return {"passed": len(mismatches) == 0, "mismatches": mismatches}

    def _extract_numbers(self, text: str) -> List[str]:
        patterns = [
            r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[KMBkmb](?:illion)?)?',
            r'-?\d+(?:\.\d+)?%',
            r'\b(?:19|20)\d{2}\b',
            r'\b(?:Q[1-4]|FY)\s*\d{2,4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        ]
        nums = []
        for p in patterns:
            nums.extend(re.findall(p, text, re.IGNORECASE))
        return list(set(nums))

    def _analyze_citations(self, answer: str, context: str) -> Dict[str, Any]:
        claims = [s for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip() and self._is_factual(s)]
        if not claims:
            return {"citation_recall": 1.0, "citation_precision": 1.0, "citation_f1": 1.0, "total_claims": 0, "supported_claims": 0, "unsupported_claims": [], "hallucinated_sentences": []}

        page_re = re.compile(r'\[page\s+\d+\]|\(page\s+\d+\)|page\s+\d+|p\.\s*\d+|\[p\.\s*\d+\]', re.IGNORECASE)
        supported, unsupported, hallucinated = [], [], []

        for claim in claims:
            has_cite = bool(page_re.search(claim))
            in_ctx = self._claim_in_context(claim, context)
            if has_cite:
                (supported if in_ctx else hallucinated).append(claim)
            else:
                unsupported.append(claim)

        total = len(claims)
        sup_count = len(supported)
        cited = sup_count + len(hallucinated)
        recall = sup_count / total if total else 1.0
        precision = sup_count / cited if cited else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        penalty = config.MODE.get("hallucination_penalty", 0.98)
        if hallucinated:
            recall *= penalty
            precision *= penalty
            f1 *= penalty

        return {"citation_recall": recall, "citation_precision": precision, "citation_f1": f1, "total_claims": total, "supported_claims": sup_count, "unsupported_claims": unsupported, "hallucinated_sentences": hallucinated}

    def _is_factual(self, s: str) -> bool:
        s = s.lower().strip()
        if len(s.split()) < 4 or s.endswith('?'):
            return False
        if any(s.startswith(w) for w in ['what','how','why','when','where','who','which']):
            return False
        if any(p in s for p in ['i think','i believe','perhaps','maybe','possibly']):
            return False
        return True

    def _claim_in_context(self, claim: str, context: str) -> bool:
        clean = re.sub(r'\[.*?\]|\(.*?\)', '', claim).lower()
        clean = re.sub(r'(page \d+|p\.\s*\d+)', '', clean)
        words = [w for w in clean.split() if len(w) > 3]
        if not words:
            return True
        ctx = context.lower()
        return sum(1 for w in words if w in ctx) / len(words) > 0.50

    def _calc_retrieval_confidence(self, sources: List[Dict], metrics: Dict) -> float:
        if not sources:
            return 0.0
        avg = metrics.get("avg_rerank_score", 0.0)
        mx = metrics.get("max_rerank_score", 0.0)
        n = len(sources)
        score_c = (avg + mx) / 2
        count_c = min(math.log(n + 1) / math.log(8), 1.0)
        return max(0.0, min(1.0, 0.7 * score_c + 0.3 * count_c))

    def _calc_confidence(self, verification, atomic, num_sources, citations, retrieval_c, numeric) -> float:
        w = config.CONFIDENCE_CALIBRATION
        v_score = 1.0 if verification.get("verified") else 0.3
        src_q = min(math.log(num_sources + 1) / math.log(10), 1.0)
        cit_score = citations.get("citation_f1", 0.0)
        conf = (w["verification_weight"] * v_score + w["source_quality_weight"] * src_q + w["citation_quality_weight"] * cit_score + w["retrieval_confidence_weight"] * retrieval_c)
        conf *= atomic.get("support_rate", 1.0)
        if not numeric.get("passed", True):
            conf *= 0.7
        hallucinated = citations.get("hallucinated_sentences", [])
        if hallucinated:
            penalty = config.MODE.get("hallucination_penalty", 0.98)
            conf *= penalty ** len(hallucinated)
        return max(0.0, min(1.0, conf))

    def _no_context_response(self) -> Dict[str, Any]:
        empty_cit = {"citation_recall": 0.0, "citation_precision": 0.0, "citation_f1": 0.0, "total_claims": 0, "supported_claims": 0, "unsupported_claims": [], "hallucinated_sentences": []}
        return {"answer": "No relevant context found to answer this question.", "sources": [], "confidence": 0.0, "verified": False, "citation_metrics": empty_cit, "verification_reason": "No context available", "atomic_verification": {"support_rate": 0.0, "atomic_facts": []}, "retrieval_confidence": 0.0, "numeric_verification": {"passed": True, "mismatches": []}}


generator = EnhancedGenerator()
