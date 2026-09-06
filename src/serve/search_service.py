"""
The online search path: photo in, ranked products out.

This is the two-stage retrieval described in README section 4. It is not a
Databricks job — it is what a serving endpoint or application runs per request.

Stage 1 (recall)  Truncate the query embedding to 64 dimensions and search the
                  whole filtered catalogue. Cheap, wide net, roughly 500 results.

Stage 2 (rerank)  Score those 500 with the full 512-dimensional embedding, then
                  blend in business signals. Expensive, narrow, precise.

Total cost is dominated by stage 1; total quality is dominated by stage 2. Doing
either alone is worse: full-precision search over everything is too slow, and
64-dim results shown directly are noticeably worse at the top of the list, which
is the only part users look at.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SearchResult:
    product_id: str
    score: float
    vector_score: float
    category: str
    rank: int


class FashionSearchService:

    def __init__(self, detector, encoder, vs_index,
                 recall_k: int = 500, recall_dim: int = 64,
                 final_k: int = 20):
        self.detector = detector
        self.encoder = encoder
        self.index = vs_index
        self.recall_k = recall_k
        self.recall_dim = recall_dim
        self.final_k = final_k

    # -- step 1: find the garments in the photo ----------------------------
    def detect(self, image, threshold: float = 0.4):
        boxes = self.detector(image, threshold=threshold)
        # Sort by area * confidence: the item the user probably means is the
        # large, confidently-detected one, not a shoe in the corner.
        return sorted(boxes, key=lambda b: -(b["area"] * b["score"]))

    # -- step 2+3: crop and embed ------------------------------------------
    def embed_crop(self, image, box):
        crop = image.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
        return self.encoder(crop)                       # [512], L2-normalised

    # -- step 4: cheap wide recall -----------------------------------------
    def recall(self, embedding, category: str,
               region: Optional[str] = None, in_stock_only: bool = True):
        short = embedding[:self.recall_dim]
        short = short / np.linalg.norm(short)

        filters = {"category": category}
        if in_stock_only:
            filters["in_stock"] = True
        if region:
            filters["region"] = region

        return self.index.similarity_search(
            query_vector=short.tolist(),
            columns=["product_id", "category", "embedding", "price", "brand"],
            filters=filters,
            num_results=self.recall_k,
        )

    # -- step 5: precise rerank --------------------------------------------
    def rerank(self, embedding, candidates, business_weight: float = 0.15):
        """
        Rescore with the full embedding, then blend in a business score.

        Keep business_weight small. It is tempting to push promoted stock up the
        list, but visual search only works because users trust that result #1
        really is the thing in their photo. Break that and they stop using it,
        and no amount of promotion recovers a search box nobody opens.
        """
        vecs = np.array([c["embedding"] for c in candidates], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        vector_scores = vecs @ embedding

        results = []
        for c, vs in zip(candidates, vector_scores):
            biz = c.get("business_score", 0.0)
            final = (1 - business_weight) * float(vs) + business_weight * biz
            results.append((c, float(vs), final))

        results.sort(key=lambda r: -r[2])
        return [SearchResult(product_id=c["product_id"], score=f,
                             vector_score=vs, category=c["category"], rank=i)
                for i, (c, vs, f) in enumerate(results[:self.final_k], start=1)]

    # -- the whole thing ----------------------------------------------------
    def search(self, image, selected_box=None, region=None):
        boxes = self.detect(image)
        if not boxes:
            return {"results": [], "reason": "no_items_detected"}

        box = selected_box or boxes[0]
        emb = self.embed_crop(image, box)
        candidates = self.recall(emb, box["category"], region=region)
        if not candidates:
            return {"results": [], "reason": "no_candidates_after_filter"}

        return {
            "results": self.rerank(emb, candidates),
            "detected_boxes": boxes,       # so the UI can offer the other items
            "selected_box": box,
        }
