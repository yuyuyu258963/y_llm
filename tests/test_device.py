import unittest

import torch

from mini_vllm.device import require_cuda, synchronize


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class DeviceBoundaryTest(unittest.TestCase):
    def test_gpu_result_can_be_explicitly_returned_to_cpu(self) -> None:
        accelerator = require_cuda()
        host_input = torch.tensor([2.0], device="cpu")

        device_result = host_input.to(accelerator.device) * 3
        synchronize(accelerator.device)
        host_result = device_result.cpu()

        self.assertEqual(host_result.device.type, "cpu")
        self.assertEqual(host_result.item(), 6.0)
