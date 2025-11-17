import json
from pathlib import Path

from evolve_term.embeddings import bulk_vectorize_directory


def test_bulk_vectorize_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "svc"
    source_dir.mkdir()
    (source_dir / "Ex01.c").write_text("int main(){return 0;}", encoding="utf-8")

    config_path = tmp_path / "embed_config.json"
    config_path.write_text(
        json.dumps(
            {
                "provider": "mock",
                "model": "mock-mini",
                "dimension": 8,
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "out.json"
    summary = bulk_vectorize_directory(
        source_dir=source_dir,
        output_path=output_path,
        config_name=str(config_path),
    )

    assert summary["files_processed"] == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["embedding_info"]["model"] == "mock-mini"
    case = payload["cases"][0]
    assert len(case["embedding"]) == 8
    assert case["metadata"]["embedding_model"] == "mock-mini"
