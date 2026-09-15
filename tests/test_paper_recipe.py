"""CPU regressions for the released training and controller conventions.

Run: python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_source_dependency/code/src"))
sys.path.insert(0, str(ROOT / "shortcut_solvability/code"))

from data_source_dependency.widowx.tasks import CONTROL_MODE, TASKS, make_env

try:
    import torch
except ImportError:
    torch = None


class ControllerTest(unittest.TestCase):
    def test_original_controller_is_accepted_for_all_tasks(self):
        for task_key, task in TASKS.items():
            with self.subTest(task=task_key):
                env = SimpleNamespace(unwrapped=SimpleNamespace(control_mode=CONTROL_MODE), close=Mock())
                factory = Mock(return_value=env)
                with patch.dict(sys.modules, {"simpler_env": SimpleNamespace(make=factory)}):
                    self.assertIs(make_env(task_key), env)
                factory.assert_called_once_with(task.env_task)
                env.close.assert_not_called()

    def test_relative_controller_is_rejected_and_closed_for_all_tasks(self):
        relative = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        for task_key in TASKS:
            with self.subTest(task=task_key):
                env = SimpleNamespace(unwrapped=SimpleNamespace(control_mode=relative), close=Mock())
                with patch.dict(sys.modules, {"simpler_env": SimpleNamespace(make=lambda name: env)}):
                    with self.assertRaisesRegex(RuntimeError, "Install the pinned X-VLA forks"):
                        make_env(task_key)
                env.close.assert_called_once_with()


@unittest.skipIf(torch is None, "PyTorch is required for the embedding regressions")
class TaskEmbeddingTest(unittest.TestCase):
    def build(self, benchmark, **overrides):
        from shortcut_policy.models import build_policy_from_cfg

        cfg = {"chunk_size": 5, "vision_backbone": "dino", "use_proprio": False, **overrides}
        # No backbone RNG or downloads: the task embedding is the first random layer.
        with patch("torch.hub.load", return_value=torch.nn.Identity()):
            return build_policy_from_cfg(cfg, benchmark=benchmark, num_tasks=10)

    def test_libero_preserves_original_random_initialization(self):
        for seed in (2, 3):
            with self.subTest(seed=seed), torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                expected = torch.nn.Embedding(10, 32).weight.detach().clone()
                torch.manual_seed(seed)
                actual = self.build("libero").task_embedding.weight
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_calvin_keeps_normalized_scaled_embeddings(self):
        for addition in (False, True):
            for scale in (1.0, 10.0):
                with self.subTest(addition=addition, scale=scale), torch.random.fork_rng(devices=[]):
                    model = self.build("calvin", use_addition=addition, task_embedding_scale=scale)
                    torch.testing.assert_close(model.task_embed.weight.norm(dim=-1), torch.full((10,), scale))


if __name__ == "__main__":
    unittest.main()
