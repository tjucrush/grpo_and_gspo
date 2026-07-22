import unittest

import torch

from utils import grpo_loss, group_advantages, gspo_loss


class GroupAdvantagesTest(unittest.TestCase):
    def test_normalizes_each_question_independently(self):
        rewards = torch.tensor([1.0, 3.0, 10.0, 14.0])

        advantages = group_advantages(rewards, num_answers_per_question=2)
        grouped = advantages.view(2, 2)

        torch.testing.assert_close(grouped.mean(dim=1), torch.zeros(2))
        torch.testing.assert_close(grouped.std(dim=1), torch.ones(2))

    def test_rejects_incomplete_groups(self):
        with self.assertRaisesRegex(AssertionError, "divisible"):
            group_advantages(torch.tensor([1.0, 2.0, 3.0]), 2)


class PolicyLossTest(unittest.TestCase):
    def setUp(self):
        # Three input tokens produce two next-token log probabilities. The first
        # token is the prompt and the second one is the generated token.
        self.ref = torch.zeros((1, 2))
        self.old = torch.zeros((1, 2))
        self.new = torch.zeros((1, 2), requires_grad=True)
        self.mask = torch.tensor([[1, 1, 0]])
        self.advantages = torch.tensor([[1.0]])

    def test_grpo_ignores_padding(self):
        loss = grpo_loss(
            self.ref, self.old, self.new, self.mask, self.advantages, prefix_len=1
        )

        torch.testing.assert_close(loss, torch.tensor(-1.0))
        loss.backward()
        self.assertTrue(torch.isfinite(self.new.grad).all())

    def test_gspo_ignores_padding(self):
        loss = gspo_loss(
            self.ref, self.old, self.new, self.mask, self.advantages, prefix_len=1
        )

        torch.testing.assert_close(loss, torch.tensor(-1.0))
        loss.backward()
        self.assertTrue(torch.isfinite(self.new.grad).all())


if __name__ == "__main__":
    unittest.main()
