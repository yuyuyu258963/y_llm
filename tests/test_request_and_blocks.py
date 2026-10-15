import unittest

from mini_vllm.block_manager import BlockExhaustedError, BlockManager
from mini_vllm.requests import Request, RequestStatus


class RequestStateTest(unittest.TestCase):
    def test_last_generated_token_is_not_yet_in_cache(self) -> None:
        request = Request("r", [1, 2, 3], max_new_tokens=2)
        request.start(cache_slot=0)
        request.accept_token(8)
        self.assertEqual(request.cache_length, 3)
        self.assertEqual(request.next_input_token_id, 8)

        request.accept_token(5)
        self.assertEqual(request.status, RequestStatus.FINISHED)
        self.assertEqual(request.cache_length, 4)

    def test_terminal_request_cannot_accept_another_token(self) -> None:
        request = Request("r", [1], max_new_tokens=1)
        request.start(cache_slot=0)
        request.accept_token(2)
        with self.assertRaises(RuntimeError):
            request.accept_token(3)


class BlockManagerTest(unittest.TestCase):
    def test_logical_positions_map_to_fixed_size_blocks(self) -> None:
        manager = BlockManager(num_blocks=4, block_size=4)
        manager.allocate_for_tokens("one", token_count=5)
        physical, offset = manager.append_token("one", position=8)

        self.assertEqual(manager.table("one"), (0, 1, 2))
        self.assertEqual((physical, offset), (2, 0))
        manager.release("one")
        self.assertEqual(manager.free_block_count, 4)

    def test_exhaustion_does_not_partially_allocate(self) -> None:
        manager = BlockManager(num_blocks=2, block_size=4)
        with self.assertRaises(BlockExhaustedError):
            manager.allocate_for_tokens("too-large", token_count=9)
        self.assertEqual(manager.free_block_count, 2)
        self.assertEqual(manager.table("too-large"), ())
