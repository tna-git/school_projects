import torch
import torchmetrics
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from transformers import BertTokenizer, BertGenerationEncoder

class ProteinClassifier(LightningModule):
    def __init__(self, n_classes=25):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.embedder = BertGenerationEncoder.from_pretrained("Rostlab/prot_bert")
        dmodel = 1024
        self.model = nn.Linear(dmodel, n_classes)
        self.save_hyperparameters()
        self.hparams.lr = 1e-3
        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                 num_classes=n_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                   num_classes=n_classes)
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        lengths = torch.tensor([len(i) for i in x]).to(self.device)
        ids = self.tokenizer(x, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device).to(self.dtype)
        with torch.no_grad():
            embeddings = self.embedder(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
        embeddings = embeddings.sum(dim=1)/lengths.view(-1, 1)
        logits = self.model(embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        f1 = self.val_f1(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
