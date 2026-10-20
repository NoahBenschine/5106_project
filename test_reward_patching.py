import unittest

from reward_patching import apply_reward_patches


def patch(step=51, agent=0, bonus=0.5):
    return {
        "step": step,
        "agent_idx": agent,
        "reward_bonus": bonus,
        "reasoning": "Test feedback",
    }


class RewardPatchingTests(unittest.TestCase):
    def setUp(self):
        self.base = [[20.0, 20.0], [0.0, 3.0]]
        self.steps = [51, 52]

    def test_correct_agent_and_step(self):
        rewards, audit = apply_reward_patches(
            self.base, self.steps, [patch(52, 1, -0.5)]
        )
        self.assertEqual(rewards, [[20.0, 20.0], [0.0, 2.5]])
        self.assertEqual(audit[0]["status"], "applied")
        self.assertEqual(self.base, [[20.0, 20.0], [0.0, 3.0]])

    def test_invalid_patches_leave_rewards_unchanged(self):
        invalid = [
            patch(bonus=999.0),
            patch(bonus=float("nan")),
            patch(bonus=float("inf")),
            patch(agent=-1),
            patch(agent=2),
            patch(step=50),
            patch(step=True),
            patch(bonus="0.5"),
        ]
        for item in invalid:
            with self.subTest(item=item):
                rewards, audit = apply_reward_patches(
                    self.base, self.steps, [item]
                )
                self.assertEqual(rewards, self.base)
                self.assertEqual(audit[0]["status"], "rejected")

    def test_duplicate_target_applied_once(self):
        rewards, audit = apply_reward_patches(
            self.base, self.steps, [patch(), patch()]
        )
        self.assertEqual(rewards[0][0], 20.5)
        self.assertEqual(audit[1]["reason"], "duplicate_target")

    def test_empty_feedback_preserves_rewards(self):
        rewards, audit = apply_reward_patches(
            self.base, self.steps, []
        )
        self.assertEqual(rewards, self.base)
        self.assertEqual(audit, [])

    def test_nonconsecutive_step_mapping(self):
        rewards, _ = apply_reward_patches(
            self.base, [51, 60], [patch(60, 1, 1.0)]
        )
        self.assertEqual(rewards[1][1], 4.0)


if __name__ == "__main__":
    unittest.main()