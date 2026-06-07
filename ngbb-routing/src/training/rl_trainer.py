"""Core RL (REINFORCE) fine-tuning logic.

Fine-tunes a pre-trained IL policy using solve quality as reward,
with KL penalty to prevent forgetting.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.models.policy import NGBBPolicy
from src.training.losses import rl_policy_gradient_loss
from src.utils.logging import get_logger

logger = get_logger("ngbb.training.rl")


class RLTrainer:
    """REINFORCE trainer for NGBB policy fine-tuning."""

    def __init__(self, policy: NGBBPolicy, il_policy: NGBBPolicy, config: dict):
        self.policy = policy
        self.il_policy = il_policy  # frozen reference
        self.il_policy.eval()
        for p in self.il_policy.parameters():
            p.requires_grad = False

        self.config = config
        self.device = config.get("device", "cpu")
        self.policy.to(self.device)
        self.il_policy.to(self.device)

        rl_cfg = config.get("rl", {})
        self.optimizer = Adam(self.policy.parameters(), lr=rl_cfg.get("lr", 3e-5))
        self.kl_weight = rl_cfg.get("kl_penalty_weight", 0.01)
        self.entropy_bonus = rl_cfg.get("entropy_bonus", 0.001)
        self.baseline_momentum = rl_cfg.get("baseline_momentum", 0.99)
        self.running_baseline = 0.0
        self.global_step = 0

    def train_episode(self, graph, reward: float) -> dict:
        """Train on a single episode (one solve).

        Args:
            graph: HeteroData from the solve.
            reward: -(nodes_ngbb / nodes_baseline).

        Returns:
            Dict with 'loss', 'reward', 'baseline'.
        """
        self.policy.train()
        graph = graph.to(self.device)

        logits = self.policy(graph)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze()
        log_prob = F.log_softmax(logits, dim=-1)[action]

        # IL reference log prob
        with torch.no_grad():
            il_logits = self.il_policy(graph)
            il_log_prob = F.log_softmax(il_logits, dim=-1)[action]

        # Update baseline
        self.running_baseline = (self.baseline_momentum * self.running_baseline
                                  + (1 - self.baseline_momentum) * reward)

        loss = rl_policy_gradient_loss(
            log_probs=log_prob.unsqueeze(0),
            rewards=torch.tensor([reward], device=self.device),
            baseline=torch.tensor([self.running_baseline], device=self.device),
            kl_penalty=self.kl_weight,
            il_log_probs=il_log_prob.unsqueeze(0),
            entropy_bonus=self.entropy_bonus,
            probs=probs.unsqueeze(0),
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        return {"loss": loss.item(), "reward": reward, "baseline": self.running_baseline}
