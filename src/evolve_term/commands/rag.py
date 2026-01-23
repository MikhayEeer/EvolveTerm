from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table

from ..knowledge_base import KnowledgeBase
from ..rag_index import HNSWIndexManager
from ..embeddings import build_embedding_client
from ..llm_client import build_llm_client
from ..loop_extractor import LoopExtractor
from ..prompts_loader import PromptRepository
from ..models import KnowledgeCase

console = Console()

class RAGHandler:
    def __init__(self, 
                 kb_path: Optional[Path] = None, 
                 embed_config: str = "embed_config.json",
                 llm_config: str = "llm_config.json"):
        
        self.kb_path = kb_path
        # Derive index paths from kb_path if provided
        self.index_path = None
        self.meta_path = None
        if self.kb_path:
            self.index_path = self.kb_path.parent / (self.kb_path.stem + "_index.bin")
            self.meta_path = self.kb_path.parent / (self.kb_path.stem + "_index_meta.json")

        self.knowledge_base = KnowledgeBase(storage_path=self.kb_path)
        self.embed_config = embed_config
        self.llm_config = llm_config
        self._embedding_client = None
        self._llm_client = None
        self._loop_extractor = None

    @property
    def embedding_client(self):
        if not self._embedding_client:
            self._embedding_client = build_embedding_client(self.embed_config)
        return self._embedding_client

    @property
    def index_manager(self):
        # We need embedding client to get dimension
        dim = self.embedding_client.dimension
        return HNSWIndexManager(dimension=dim, index_path=self.index_path, meta_path=self.meta_path)

    @property
    def loop_extractor(self):
        if not self._loop_extractor:
            llm = build_llm_client(self.llm_config)
            repo = PromptRepository()
            self._loop_extractor = LoopExtractor(llm, repo)
        return self._loop_extractor

    def status(self):
        count = len(self.knowledge_base.cases)
        console.print(f"Knowledge Base: {self.knowledge_base.path}")
        console.print(f"Total Cases: {count}")
        
        idx_mgr = self.index_manager
        if idx_mgr.index:
            console.print(f"Index: Ready (Count: {idx_mgr.index.get_current_count()})")
            console.print(f"Index Path: {idx_mgr.index_path}")
        else:
            console.print("[yellow]Index: Not built or empty[/yellow]")

    def add(self, files: List[Path], label: str = "unknown", use_loops: bool = True):
        added_count = 0
        for file_path in files:
            if not file_path.exists():
                console.print(f"[red]File not found: {file_path}[/red]")
                continue
            
            console.print(f"Processing {file_path}...")
            code = file_path.read_text(encoding='utf-8')
            
            # Extract loops
            loops = []
            if use_loops:
                loops = self.loop_extractor.extract(code)
                console.print(f"  - Extracted {len(loops)} loops")
            
            # Embed
            text_to_embed = "\n".join(loops) if (loops and use_loops) else code
            vector = self.embedding_client.embed(text_to_embed)
            
            try:
                embedding_list = vector.astype(float).tolist()
            except:
                embedding_list = list(vector)

            # Create Case
            case_id = uuid.uuid4().hex
            case = KnowledgeCase(
                case_id=case_id,
                code=code,
                label=label,
                explanation=f"Imported from {file_path.name}",
                loops=loops,
                embedding=embedding_list,
                metadata={"source_file": str(file_path)}
            )
            
            self.knowledge_base.add_case(case)
            added_count += 1
            
        console.print(f"[green]Added {added_count} cases.[/green]")
        console.print("Rebuilding index...")
        self.rebuild()

    def rebuild(self):
        cases = self.knowledge_base.cases
        if not cases:
            console.print("[yellow]Knowledge base is empty. Nothing to build.[/yellow]")
            return
        
        self.index_manager.rebuild(cases)
        self.knowledge_base.mark_rebuilt()
        console.print("[green]Index rebuilt successfully.[/green]")

    def search(self, query_file: Optional[Path] = None, text: Optional[str] = None, top_k: int = 5, use_loops: bool = True):
        if not query_file and not text:
            console.print("[red]Either query file or text must be provided.[/red]")
            return

        query_text = text
        if query_file:
            code = query_file.read_text(encoding='utf-8')
            if use_loops:
                loops = self.loop_extractor.extract(code)
                query_text = "\n".join(loops) if loops else code
            else:
                query_text = code
        
        vector = self.embedding_client.embed(query_text)
        results = self.index_manager.search(vector, top_k=top_k)
        
        table = Table(title=f"Search Results (Top {top_k})")
        table.add_column("Similarity", style="cyan")
        table.add_column("Case ID", style="magenta")
        table.add_column("Source", style="green")
        table.add_column("Label")
        
        for case_id, score in results:
            case = self.knowledge_base.get(case_id)
            source = case.metadata.get("source_file", "n/a") if case else "Unknown"
            label = case.label if case else "Unknown"
            table.add_row(f"{score:.4f}", case_id[:8], Path(source).name, label)
            
        console.print(table)
