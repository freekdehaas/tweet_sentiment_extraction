import torch.nn as nn
from transformers import AlbertForQuestionAnswering


class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        # RoBERTa encoder
        self.model = AlbertForQuestionAnswering.from_pretrained("albert-base-v2")

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, ids, masks):  # , token_type_ids
        """
        start_scores torch.FloatTensor of shape (batch_size, sequence_length,)
        Span-start scores (before SoftMax).

        end_scores: torch.FloatTensor of shape (batch_size, sequence_length,)
        Span-end scores (before SoftMax).


        """
        start_logits, end_logits = self.model(input_ids=ids, attention_mask=masks)

        return start_logits, end_logits
