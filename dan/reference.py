"""The correctness oracle: a plain Hugging Face forward per question.

Slow on purpose. Every faster runner must match what this returns.
"""
import torch


class HFReference:
    def __init__(self, model, device="cpu", dtype=torch.float32):
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def read(self, plan):
        """Per branch, the logits of its label tokens at its read position."""
        out = []
        for b in plan.branches:
            ids = torch.tensor([plan.prefix + b.suffix], device=self.device)
            logits = self.model(input_ids=ids).logits[0, -1]
            out.append(logits[b.label_ids].float().cpu())
        return out
