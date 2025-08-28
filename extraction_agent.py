# server/agents/extraction_agent.py

from pathlib import Path
from typing import List
from server.utils.excel_parser import parse_excel_or_csv
from server.utils.file_parser import parse_file

class ExtractionAgent:
    def __init__(self, kb_paths: List[str], fi_paths: List[str], prd_paths: List[str] = None):
        self.kb_paths = [Path(p) for p in (kb_paths or [])]
        self.fi_paths = [Path(p) for p in (fi_paths or [])]
        self.prd_paths = [Path(p) for p in (prd_paths or [])]

        # Validation
        if not self.kb_paths:
            raise ValueError("At least one Knowledge Bank file must be provided.")
        if not self.fi_paths:
            raise ValueError("At least one Field Issues file must be provided.")
        for path in self.kb_paths + self.fi_paths + self.prd_paths:
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")

    def load_knowledge_bank(self):
        all_rows = []
        for path in self.kb_paths:
            print(f"[ExtractionAgent] Loading Knowledge Bank: {path}")
            all_rows.extend(parse_excel_or_csv(str(path)))
        return all_rows

    def load_field_issues(self):
        all_rows = []
        for path in self.fi_paths:
            print(f"[ExtractionAgent] Loading Field Issues: {path}")
            all_rows.extend(parse_excel_or_csv(str(path)))
        return all_rows

    def load_prds(self):
        all_rows = []
        for path in self.prd_paths:
            print(f"[ExtractionAgent] Loading PRD: {path}")
            all_rows.extend(parse_file(str(path)))
        return all_rows
