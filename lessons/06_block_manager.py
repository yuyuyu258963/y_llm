"""Chapter 06: observe token positions becoming block-table entries."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.block_manager import BlockManager


def main() -> None:
    blocks = BlockManager(num_blocks=8, block_size=4)
    blocks.allocate_for_tokens("A", token_count=5)
    print(f"A after a 5-token prompt: logical table={blocks.table('A')}")
    physical, offset = blocks.append_token("A", position=8)
    print(f"A token position 8 -> physical block {physical}, offset {offset}")
    print(f"free blocks before release: {blocks.free_block_count}")
    blocks.release("A")
    print(f"free blocks after release: {blocks.free_block_count}")


if __name__ == "__main__":
    main()
