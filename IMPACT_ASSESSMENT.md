# RAG System Impact Assessment & Improvement Plan

## 1. Executive Summary

**Current Status**: The system is a **hybrid text-retrieval engine** with a "Multimodal" veneer. While it has a clean codebase and standard evaluation metrics, it fails to leverage true multimodal capabilities (VLMs are blind to actual image data) and relies on rigid, hardcoded logic for retrieval (specifically hardwired to "Harvey" queries).

**Verdict**: Currently at **Proof-of-Concept (PoC)** level. Not yet ready for academic publication due to significant architectural limitations in true multimodality and query generalization.

## 2. Critical Analysis

### Weaknesses (The "Paper Killers")

1.  **"Fake" Multimodality**:
    *   **Issue**: The `VLMClient` does not pass image pixels or embeddings to the model. It only passes `tile_ids` (strings) and `text_snippets` in the prompt.
    *   **Impact**: The VLM is hallucinating visual analysis based on text context or metadata. It cannot actually "see" flood waters or damage.
    *   **Evidence**: `app/core/models/prompts.py` constructs a text-only prompt. `vlm_client.py` sends this text prompt to Gemini without image data attachment.

2.  **Rigid Retrieval Logic**:
    *   **Issue**: `HybridTextRetriever` hardcodes the search query: `f"Harvey flood damage impact summary..."`.
    *   **Impact**: The system cannot answer general questions or questions about other events. It ignores the user's specific semantic intent (e.g., "Are roads blocked?" vs "Is there structural damage?") and always fetches a generic summary.

3.  **Naive Spatial Indexing**:
    *   **Issue**: `SpatialIndex` uses a linear scan (`for tile in self.imagery`) over in-memory lists.
    *   **Impact**: O(N) complexity. This will fail at scale (e.g., millions of tiles). No use of geospatial indexing standards like H3, S2, or R-trees.

4.  **Lack of Agentic Planning**:
    *   **Issue**: The pipeline is linear: `Parse -> Retrieve -> Select -> Generate`.
    *   **Impact**: Complex queries (e.g., "Compare the damage between Zip A and Zip B") will fail because the system cannot decompose them into sub-steps.

### Strengths

*   **Evaluation Framework**: The inclusion of `cohen_kappa` and `f1` metrics is excellent and academic-standard.
*   **Clean Architecture**: The separation of `dataio`, `indexing`, `retrieval`, and `models` is sound and extensible.
*   **Hybrid Search**: Implementing both BM25 and Dense search is a best practice (though the weighting is currently static).

## 3. Proposed SOTA Architecture (2024/2025 Standard)

To make this publication-ready, we must move to a **Agentic Spatial-Multimodal RAG**.

### A. True Multimodal Indexing (ColPali / CLIP)
Instead of relying on metadata, we must index the images semantically.
*   **Method**: Use **ColPali** (Late Interaction) or **SigLIP** embeddings for imagery.
*   **Implementation**:
    *   Generate visual embeddings for every tile.
    *   Store these in a Vector DB (e.g., Qdrant/Milvus) alongside text embeddings.
    *   Pass **actual image bytes** to Gemini/GPT-4o during generation.

### B. Geospatial Vector Search
*   **Method**: Use **H3** (Hexagonal Hierarchical Spatial Index) for bucketing.
*   **Implementation**:
    *   Convert Lat/Lon to H3 indices (resolution 8-10).
    *   Filter retrieval candidates by H3 cells rather than linear scanning.
    *   Allow "Radius Search" by finding neighbor cells.

### C. Agentic Query Planner
*   **Method**: **ReAct** or **Plan-and-Solve** pattern.
*   **Implementation**:
    *   **Planner**: Decomposes "How is the flooding in 77096?" into:
        1.  `Get_Bounding_Box(77096)`
        2.  `Retrieve_Imagery(bbox, time)`
        3.  `Retrieve_Social_Media(bbox, time)`
        4.  `Synthesize()`
    *   **Tools**: Expose `retrieve_imagery` and `retrieve_text` as tools to an LLM Agent.

### D. Dynamic Hybrid Retrieval
*   **Method**: **Reciprocal Rank Fusion (RRF)**.
*   **Implementation**: Replace fixed `0.4/0.6` weights with RRF to dynamically balance sparse (keyword) and dense (semantic) results based on rank position.

## 4. Implementation Plan

1.  **Fix VLM Integration**: Update `VLMClient` to accept and transmit base64 encoded images or public URLs to Gemini.
2.  **Implement H3 Indexing**: Replace `SpatialIndex` list scan with an H3-based dictionary lookup.
3.  **Generalize Retriever**: Remove "Harvey" hardcoding. Pass the actual user query to the embedding model.
4.  **Upgrade Evaluation**: Add a "Faithfulness" metric (using an LLM judge) to verify if the answer is actually grounded in the retrieved context.


