"""Knowledge Distillation trainer: Teacher -> Student GNN transfer.

Trains Student to mimic Teacher branching via soft-label distillation.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
# pyrefly: ignore [missing-import]
from torch_geometric.loader import DataLoader

# pyrefly: ignore [missing-import]
from src.models.distillation import (
    TeacherStudentWrapper, KnowledgeDistillationLoss, StudentPolicy,
)
# pyrefly: ignore [missing-import]
from src.models.policy import NGBBPolicy
# pyrefly: ignore [missing-import]
from src.training.losses import compute_top_k_accuracy


class KDTrainer:
    """Knowledge Distillation trainer."""

    def __init__(self, teacher_policy: NGBBPolicy,
                 student_policy: StudentPolicy, config: dict):
        self.config = config
        self.device = config.get("device", "cpu")
        kd_cfg = config.get("distillation", {})
        tr_cfg = config.get("training", {})

        self.wrapper = TeacherStudentWrapper(
            teacher_policy.gnn, teacher_policy.head,
            student_policy.gnn, student_policy.head,
        ).to(self.device)

        self.kd_loss = KnowledgeDistillationLoss(
            temperature=kd_cfg.get("temperature", 4.0),
            alpha=kd_cfg.get("alpha", 0.7),
            feature_weight=kd_cfg.get("feature_weight", 0.1),
            edge_penalty_weight=kd_cfg.get("edge_penalty_weight", 0.05),
        )

        self.optimizer = AdamW(
            self.wrapper.get_student_params(),
            lr=tr_cfg.get("lr", 5e-4),
            weight_decay=tr_cfg.get("weight_decay", 1e-4),
        )

        warmup = LinearLR(self.optimizer, start_factor=0.01,
                          total_iters=tr_cfg.get("warmup_steps", 500))
        cosine = CosineAnnealingLR(self.optimizer,
                                    T_max=tr_cfg.get("n_steps", 100000))
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine],
            milestones=[tr_cfg.get("warmup_steps", 500)]
        )
        self.grad_clip = tr_cfg.get("grad_clip", 1.0)
        self.best_val_acc = 0.0
        self.global_step = 0

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.wrapper.train()
        self.wrapper.teacher_gnn.eval()
        self.wrapper.teacher_head.eval()
        tot_loss = tot_acc = 0.0
        n = 0
        for batch in dataloader:
            batch = batch.to(self.device)
            labels = batch.y if hasattr(batch, 'y') else batch["label"]
            out = self.wrapper(batch)
            s_log = out['student_logits']
            t_log = out['teacher_logits']
            if s_log.dim() == 1:
                s_log = s_log.unsqueeze(0)
                t_log = t_log.unsqueeze(0)
            ld = self.kd_loss(s_log, t_log, labels,
                              out.get('student_features'),
                              out.get('teacher_features'))
            loss = ld['total_loss']
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.wrapper.get_student_params(),
                                     self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            tot_loss += loss.item()
            tot_acc += compute_top_k_accuracy(s_log, labels)
            n += 1
        return {"loss": tot_loss / max(n, 1), "top1_acc": tot_acc / max(n, 1)}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        self.wrapper.eval()
        tot_loss = tot_acc = 0.0
        n = 0
        for batch in dataloader:
            batch = batch.to(self.device)
            labels = batch.y if hasattr(batch, 'y') else batch["label"]
            out = self.wrapper(batch)
            s_log = out['student_logits']
            t_log = out['teacher_logits']
            if s_log.dim() == 1:
                s_log = s_log.unsqueeze(0)
                t_log = t_log.unsqueeze(0)
            ld = self.kd_loss(s_log, t_log, labels)
            tot_loss += ld['total_loss'].item()
            tot_acc += compute_top_k_accuracy(s_log, labels)
            n += 1
        return {"loss": tot_loss / max(n, 1), "top1_acc": tot_acc / max(n, 1)}

    def save_checkpoint(self, filepath: str):
        student = StudentPolicy(self.wrapper.student_gnn,
                                self.wrapper.student_head)
        torch.save({
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_acc": self.best_val_acc,
        }, filepath)

    def compare_sizes(self) -> dict:
        tp = sum(p.numel() for p in self.wrapper.teacher_gnn.parameters())
        tp += sum(p.numel() for p in self.wrapper.teacher_head.parameters())
        sp = sum(p.numel() for p in self.wrapper.student_gnn.parameters())
        sp += sum(p.numel() for p in self.wrapper.student_head.parameters())
        return {"teacher": tp, "student": sp,
                "ratio": tp / max(sp, 1)}
