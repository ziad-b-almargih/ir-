from typing import List, Dict
import numpy as np


class RetrievalService:
    @staticmethod
    def get_top_results(scores: np.ndarray, raw_documents: List[str], top_k: int = 10) -> List[Dict]:
        # Pure ranking: every strategy hands us a uniform score array, so no model-specific branching here.
        scores = np.asarray(scores)
        ranked_indices = scores.argsort()[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score > 0:
                results.append({
                    "rank": len(results) + 1,
                    "score": round(score, 4),
                    "document": raw_documents[idx],
                })

        return results
