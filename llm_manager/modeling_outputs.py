from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import SequenceClassifierOutput


class SequenceClassifierOutputOverRanges(SequenceClassifierOutput):
    def __init__(
        self,
        logits: torch.FloatTensor,
        loss: Optional[torch.FloatTensor] = None,
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None,
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None,
        token_type_ranges: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions
        )  # type: ignore
        self.token_type_ranges = token_type_ranges

    def to_tuple(self):
        # Ensure that your custom field is included when converting to a tuple
        return tuple(
            v
            for v in (
                self.loss,
                self.logits,
                self.hidden_states,
                self.attentions,
                self.token_type_ranges,
            )
            if v is not None
        )
