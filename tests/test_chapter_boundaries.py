"""Architecture tests: earlier lessons must not import later optimizations."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


class ChapterBoundaryTest(unittest.TestCase):
    def test_chapter_one_reference_model_has_no_pool_dependency(self) -> None:
        imports = imported_modules(ROOT / "mini_vllm" / "toy_model.py")
        self.assertNotIn("mini_vllm.cache", imports)
        self.assertNotIn("mini_vllm.pooled_attention", imports)
        self.assertNotIn("mini_vllm.pooled_runner", imports)
        self.assertNotIn("mini_vllm.hf_backend", imports)

    def test_chapter_two_runner_has_no_pool_dependency(self) -> None:
        imports = imported_modules(ROOT / "mini_vllm" / "runner.py")
        self.assertNotIn("mini_vllm.cache", imports)
        self.assertNotIn("mini_vllm.pooled_attention", imports)
        self.assertNotIn("mini_vllm.pooled_runner", imports)
        self.assertNotIn("mini_vllm.hf_backend", imports)
