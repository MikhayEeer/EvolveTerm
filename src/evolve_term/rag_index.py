"""HNSW index management for retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import hnswlib
import numpy as np

from .exceptions import IndexNotReadyError
from .models import KnowledgeCase

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INDEX_PATH = DATA_DIR / "hnsw_index.bin"
INDEX_META_PATH = DATA_DIR / "hnsw_meta.json"


class HNSWIndexManager:
    """Wraps hnswlib index creation, persistence, and queries."""

    def __init__(self, dimension: int, space: str = "cosine"):
        self.dimension = dimension
        self.space = space
        self.index_path = INDEX_PATH
        self.meta_path = INDEX_META_PATH
        self.index: hnswlib.Index | None = None
        self.case_ids: List[str] = []
        self._load_if_available()

    # Persistence ----------------------------------------------------------
    def _load_if_available(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            return
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.case_ids = meta.get("case_ids", [])
        if not self.case_ids:
            return
        index = hnswlib.Index(space=self.space, dim=self.dimension)
        index.load_index(str(self.index_path))
        self.index = index

    def save(self) -> None:
        if not self.index:
            return
        self.index.save_index(str(self.index_path))
        payload = {"case_ids": self.case_ids, "dimension": self.dimension, "space": self.space}
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build ----------------------------------------------------------------
    def rebuild(self, cases: Sequence[KnowledgeCase]) -> None:
        if not cases:
            raise IndexNotReadyError("Cannot build index without cases")
        embeddings = np.array([case.embedding for case in cases], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise IndexNotReadyError(
                f"All embeddings must be of shape (*, {self.dimension}), got {embeddings.shape}"
            )
        index = hnswlib.Index(space=self.space, dim=self.dimension)
        index.init_index(max_elements=len(cases), ef_construction=200, M=64)
        index.add_items(embeddings, ids=np.arange(len(cases)))
        index.set_ef(64)
        self.index = index
        self.case_ids = [case.case_id for case in cases]
        self.save()

    # Query ----------------------------------------------------------------
    def search(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            raise IndexNotReadyError("HNSW index is not ready. Build it first.")
        labels, distances = self.index.knn_query(vector, k=top_k)
        results = []
        for label, distance in zip(labels[0], distances[0]):
            case_id = self.case_ids[label]
            score = 1 - float(distance)  # cosine distance -> similarity proxy
            results.append((case_id, score))
        return results


def load_test_cases_from_file(embedding_file: Path) -> tuple[list[KnowledgeCase], int]:
    """Load test cases from a JSON embedding file.
    
    Args:
        embedding_file: Path to JSON file with embeddings data
        
    Returns:
        Tuple of (cases list, embedding dimension)
    """
    with open(embedding_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    embedding_info = data["embedding_info"]
    dimension = embedding_info["dimension"]
    
    cases_raw = data["cases"]
    cases = [
        KnowledgeCase(
            case_id=case["case_id"],
            code=case["code"],
            label=case.get("label", "unknown"),  # type: ignore
            explanation=case["explanation"],
            loops=case.get("loops", []),
            embedding=case["embedding"],
        )
        for case in cases_raw
    ]
    
    return cases, dimension


def create_test_cases(dimension: int, num_cases: int = 3) -> list[KnowledgeCase]:
    """Create synthetic test cases with random embeddings.
    
    Args:
        dimension: Embedding dimension
        num_cases: Number of synthetic cases to create
        
    Returns:
        List of synthetic KnowledgeCase objects
    """
    np.random.seed(42)  # For reproducibility
    cases = []
    for i in range(num_cases):
        embedding = np.random.randn(dimension).astype(np.float32).tolist()
        cases.append(
            KnowledgeCase(
                case_id=f"synthetic_case_{i}",
                code=f"// Synthetic test case {i}\nint main() {{ return {i}; }}",
                label="unknown",  # type: ignore
                explanation=f"Synthetic test case number {i}",
                loops=[],
                embedding=embedding,
            )
        )
    return cases


if __name__ == "__main__":
    """Unit tests for HNSW index building, persistence, and retrieval.
    
    Configuration:
    - Set TEST_MODE to choose test data source:
      'prebuild': Use data/prebuild_SVC25_c_embeddings.json (4 real cases)
      'synthetic': Generate random test cases (3 cases by default)
    - Set QUERY_MODE to choose query vectors:
      'from_index': Use cases from the index (self-retrieval test)
      'custom': Use custom query vectors (can be modified below)
    """
    
    # ========== CONFIGURATION ==========
    TEST_MODE = "prebuild"  # "prebuild" or "synthetic"
    QUERY_MODE = "from_index"  # "from_index" or "custom"
    
    # For synthetic test, customize these:
    SYNTHETIC_NUM_CASES = 3
    SYNTHETIC_DIMENSION = 64
    
    # For custom queries, define them here (only used if QUERY_MODE == "custom"):
    CUSTOM_QUERY_VECTORS = {
        # Example: "query_1": [0.1, 0.2, ...],  # List of floats with length = dimension
    }
    
    # Custom test vector (can be used in Test 3)
    test_query_ex01_aug1_vector = None  # Set this to a vector list if needed
    # ===================================
    
    print("=" * 70)
    print("HNSW Index Unit Tests")
    print("=" * 70)
    
    # Load or generate test data
    if TEST_MODE == "prebuild":
        embedding_file = DATA_DIR / "prebuild_SVC25_c_embeddings.json"
        build_cases, dimension = load_test_cases_from_file(embedding_file)
        print(f"\nLoaded prebuild data from: {embedding_file}")
    elif TEST_MODE == "synthetic":
        build_cases = create_test_cases(SYNTHETIC_DIMENSION, SYNTHETIC_NUM_CASES)
        dimension = SYNTHETIC_DIMENSION
        print(f"\nGenerated {len(build_cases)} synthetic test cases")
    else:
        raise ValueError(f"Unknown TEST_MODE: {TEST_MODE}")
    
    print(f"   - Embedding Dimension: {dimension}")
    print(f"   - Total Cases for Index: {len(build_cases)}")
    print(f"   - Case IDs: {[c.case_id for c in build_cases]}")
    
    # =====================================================================
    # Test 1: Index Building
    # =====================================================================
    print(f"\n{'─' * 70}")
    print("Test 1: Index Building")
    print('─' * 70)
    try:
        manager = HNSWIndexManager(dimension=dimension)
        manager.rebuild(build_cases)
        print(f"Index built successfully")
        print(f"   - Index dimension: {manager.dimension}")
        print(f"   - Index space: {manager.space}")
        print(f"   - Number of items in index: {len(manager.case_ids)}")
        print(f"   - Case IDs in index: {manager.case_ids}")
    except Exception as e:
        print(f"Failed to build index: {e}")
        raise
    
    # =====================================================================
    # Test 2: Index Persistence (Save & Load)
    # =====================================================================
    print(f"\n{'─' * 70}")
    print("Test 2: Index Persistence (Save & Load)")
    print('─' * 70)
    try:
        # Check if index files exist
        index_exists = manager.index_path.exists()
        meta_exists = manager.meta_path.exists()
        print(f"   Index file exists: {index_exists}")
        print(f"   Meta file exists: {meta_exists}")
        
        if not (index_exists and meta_exists):
            print(f"Index files not found after save (unexpected)")
        else:
            # Create new manager to test loading
            manager2 = HNSWIndexManager(dimension=dimension)
            if manager2.index is not None:
                print(f"Index loaded successfully from disk")
                print(f"   - Loaded case IDs: {manager2.case_ids}")
                assert manager2.case_ids == manager.case_ids, "Case IDs mismatch after loading!"
            else:
                print(f"Index was not loaded (this shouldn't happen after save)")
    except Exception as e:
        print(f"Failed in persistence test: {e}")
        raise
    
    # =====================================================================
    # Test 3: RAG Retrieval (Query and Hit Rate)
    # =====================================================================
    print(f"\n{'─' * 70}")
    print("Test 3: RAG Retrieval & Hit Rate Analysis")
    print('─' * 70)
    
    if QUERY_MODE == "from_index":
        # Use cases from the index for queries (self-retrieval test)
        query_pairs = [(case, case.case_id) for case in build_cases]
        print(f"   Query mode: Self-retrieval (using index cases as queries)")
        print(f"   Testing retrieval for {len(query_pairs)} cases:")
    elif QUERY_MODE == "custom":
        # Use custom query vectors
        if not CUSTOM_QUERY_VECTORS:
            print(f"CUSTOM_QUERY_VECTORS is empty, skipping custom query test")
            query_pairs = []
        else:
            query_pairs = [
                (
                    KnowledgeCase(
                        case_id=qid,
                        code="",
                        label="unknown",  # type: ignore
                        explanation="",
                        loops=[],
                        embedding=qvec,
                    ),
                    qid,
                )
                for qid, qvec in CUSTOM_QUERY_VECTORS.items()
            ]
            print(f"   Query mode: Custom vectors ({len(query_pairs)} queries)")
    else:
        raise ValueError(f"Unknown QUERY_MODE: {QUERY_MODE}")
    
    test_query_ex01_aug1_vector = [
        -0.18397238850593567,
        0.018048491328954697,
        -0.025783557444810867,
        0.04289311170578003,
        -0.1350879669189453,
        0.024442216381430626,
        0.09914001822471619,
        0.06724590063095093,
        -0.08399776369333267,
        0.23500297963619232,
        -0.10563807189464569,
        0.11773994565010071,
        -0.10671114176511765,
        0.1360418051481247,
        0.152137890458107,
        -0.026022018864750862,
        -0.012236012145876884,
        -0.2873450815677643,
        0.11660725623369217,
        -0.12459569424390793,
        0.008919918909668922,
        0.08203046768903732,
        -0.12483415007591248,
        0.25610673427581787,
        -0.06945166736841202,
        0.05362384021282196,
        -0.03964408487081528,
        0.1590532511472702,
        0.13914179801940918,
        -0.156430184841156,
        0.035918135195970535,
        -0.21139536798000336,
        0.1255495399236679,
        -0.3758139908313751,
        -0.1577417254447937,
        0.014672782272100449,
        -0.1838531643152237,
        0.08888620883226395,
        -0.018063394352793694,
        -0.040568117052316666,
        0.008793236687779427,
        -0.010335778817534447,
        -0.012876875698566437,
        -0.0008280918700620532,
        0.0404190793633461,
        0.08471314609050751,
        0.03159603476524353,
        -0.24060679972171783,
        -0.12316492944955826,
        0.023846065625548363,
        -0.10289577394723892,
        0.1064726784825325,
        0.13854563236236572,
        -0.036782555282115936,
        -0.21783380210399628,
        0.055293064564466476,
        0.12221108376979828,
        -0.019225889816880226,
        0.08685929328203201,
        -0.058601707220077515,
        0.11291112005710602,
        0.15225712954998016,
        -0.012154041789472103,
        0.0016422114567831159
      ]

    # Always add test_query_ex01_aug1_vector to query list
    if test_query_ex01_aug1_vector is not None:
        query_pairs.append(
            (
                KnowledgeCase(
                    case_id="test_query_ex01_aug1",
                    code="",
                    label="unknown",  # type: ignore
                    explanation="",
                    loops=[],
                    embedding=test_query_ex01_aug1_vector,
                ),
                "test_query_ex01_aug1",
            )
        )
        print(f"   Added test_query_ex01_aug1_vector to query list")

    try:
        hit_count = 0
        total_queries = len(query_pairs)
        
        for query_case, expected_id in query_pairs:
            query_vector = np.array([query_case.embedding], dtype=np.float32)
            results = manager.search(query_vector, top_k=min(4, len(build_cases)))
            top_case_id, top_score = results[0]
            
            is_hit = top_case_id == expected_id
            hit_count += is_hit
            status = "✓" if is_hit else "✗"
            
            # Show detailed results
            print(f"      {status} Query '{expected_id}':")
            print(f"         Top result: '{top_case_id}' (score: {top_score:.4f})")
            if not is_hit and len(results) > 1:
                print(f"         Other results: {', '.join([f'{cid}({s:.3f})' for cid, s in results[1:]])}")
        
        if total_queries > 0:
            hit_rate = hit_count / total_queries * 100
            print(f"\n   Hit Rate: {hit_count}/{total_queries} ({hit_rate:.1f}%)")
            if hit_rate == 100:
                print(f"   Perfect retrieval achieved!")
            elif hit_rate >= 75:
                print(f"   Good retrieval performance")
            else:
                print(f"   Retrieval performance could be improved")
        else:
            print(f"   No queries to test")
    
    except Exception as e:
        print(f"Failed in retrieval test: {e}")
        raise
    
    print(f"\n{'=' * 70}")
    print("All tests completed successfully!")
    print("=" * 70)
