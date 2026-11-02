"""
Hybrid Lexical-Semantic Reranker and Context Deduplicator for KnowledgeForge AI.
"""
from typing import List, Dict, Any
import re


class HybridReranker:
    """
    Reranks retrieved candidate chunks by combining semantic vector scores
    with query keyword density and removes redundant/duplicated passages.
    """

    def __init__(self, semantic_weight: float = 0.65, lexical_weight: float = 0.35):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

    def _tokenize(self, text: str) -> set:
        """Extracts lowercase alphabetic tokens."""
        return set(re.findall(r'\b[a-z0-9_]{2,}\b', text.lower()))

    def _calculate_lexical_overlap(self, query_tokens: set, text_tokens: set) -> float:
        """Computes query keyword coverage."""
        if not query_tokens or not text_tokens:
            return 0.0
        intersection = query_tokens.intersection(text_tokens)
        return len(intersection) / len(query_tokens)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Reranks chunks and deduplicates overlapping content.
        """
        if not candidates:
            return []

        query_tokens = self._tokenize(query)
        scored_candidates = []

        for chunk in candidates:
            semantic_score = float(chunk.get("similarity_score", 0.0))
            text = chunk.get("text", "")
            chunk_tokens = self._tokenize(text)

            lexical_score = self._calculate_lexical_overlap(query_tokens, chunk_tokens)

            # Boost if query appears as exact substring
            phrase_boost = 0.15 if query.lower().strip() in text.lower() else 0.0

            composite_score = (
                (self.semantic_weight * max(0.0, semantic_score)) +
                (self.lexical_weight * lexical_score) +
                phrase_boost
            )

            c_copy = dict(chunk)
            c_copy["rerank_score"] = float(round(composite_score, 4))
            c_copy["lexical_coverage"] = float(round(lexical_score, 2))
            scored_candidates.append(c_copy)

        # Sort descending by composite rerank score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Deduplicate passages with near-identical token sets (>85% overlap)
        deduped = []
        seen_token_sets = []

        for candidate in scored_candidates:
            c_tokens = self._tokenize(candidate["text"])
            is_dup = False
            for prev_tokens in seen_token_sets:
                if not c_tokens or not prev_tokens:
                    continue
                jaccard = len(c_tokens.intersection(prev_tokens)) / len(c_tokens.union(prev_tokens))
                if jaccard > 0.85:
                    is_dup = True
                    break

            if not is_dup:
                deduped.append(candidate)
                seen_token_sets.append(c_tokens)

            if len(deduped) >= top_k:
                break

        return deduped
