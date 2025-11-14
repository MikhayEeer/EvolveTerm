from pathlib import Path

from evolve_term.knowledge_base import KnowledgeBase
from evolve_term.models import KnowledgeCase


def test_pending_rebuild_flag(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.json"
    kb_path.write_text("{\"cases\": [], \"pending_since_rebuild\": 0}")
    kb = KnowledgeBase(path=kb_path, rebuild_threshold=2)

    case = KnowledgeCase(
        case_id="1",
        code="int main(){}",
        label="unknown",
        explanation="",
        loops=[],
        embedding=[0.0] * 64,
    )
    kb.add_case(case)
    assert not kb.needs_rebuild()

    kb.add_case(case)
    assert kb.needs_rebuild()
