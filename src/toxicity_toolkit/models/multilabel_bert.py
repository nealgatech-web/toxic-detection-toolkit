from torch import nn
from transformers import AutoModel

class MultiLabelBertHead(nn.Module):
    def __init__(self, base, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0,:]
        return self.classifier(self.dropout(cls))
