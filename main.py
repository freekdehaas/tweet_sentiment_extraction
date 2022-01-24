import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AlbertTokenizer

import utils
import opts
from model import TextModel
from data import train_val_dataloaders, test_loader


opt = opts.parse_opt()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_path = os.path.join(opt.train_data, "train.csv")
train_df = pd.read_csv(train_path)
train_df.drop(
    314, inplace=True
)  # This row was found to have 'nan' values, so dropping it
train_df.reset_index(drop=True, inplace=True)
train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"]

test_path = os.path.join(opt.test_data, "test.csv")
test_df = pd.read_csv(test_path).reset_index(drop=True)
test_df["text"] = test_df["text"].astype(str)


def loss_function(start_logits, end_logits, start_positions, end_positions):
    # calculating cross entropy losses for both the start and end logits
    loss = nn.CrossEntropyLoss(
        reduction="mean"
    )  # for a multi-class classification problem
    start_loss = loss(start_logits, start_positions)
    end_loss = loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def train(
    tokenizer,
    model,
    dataloaders_dict,
    optimizer,
    num_epochs,
    scheduler,
    device,
    filename,
):
    """
    Train pytorch model on a single pass through the data loader.

        This function is built with reusability in mind: it can be used as is as long
        as the `dataloader` outputs a batch in dictionary format that can be passed
        straight into the model - `model(**batch)`.

      Some of the arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

      optimizer_ (:obj:`transformers.optimization.AdamW`):
          Optimizer used for training.

      scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
          PyTorch scheduler.

      device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.
    """
    # Set device as `cuda` (GPU)
    model.to(device)
    # store the loss value for plotting.
    train_loss = []
    val_loss = []
    jac_train = []
    jac_val = []
    for epoch in range(num_epochs):
        for key in ["train", "val"]:
            if key == "train":
                model.train()
                dataloaders = dataloaders_dict["train"]
            else:
                model.eval()
                dataloaders = dataloaders_dict["val"]

            epoch_loss = 0.0
            epoch_jaccard = 0.0

            # Set tqdm to add loading screen and set the length
            loader = tqdm(dataloaders, total=len(dataloaders))
            # print(len(dataloaders))

            # loop over the data iterator, and feed the inputs to the network
            # Train the model on each batch
            for (idx, data) in enumerate(loader):
                ids = data["ids"]
                masks = data["masks"]
                text = data["text"]
                offsets = data["offsets"].numpy()
                start_idx = data["start_index"]
                end_idx = data["end_index"]
                sentiment = data["sentiment"]

                model.zero_grad()
                optimizer.zero_grad()

                ids = ids.to(device, dtype=torch.long)
                masks = masks.to(device, dtype=torch.long)
                start_idx = start_idx.to(device, dtype=torch.long)
                end_idx = end_idx.to(device, dtype=torch.long)

                with torch.set_grad_enabled(key == "train"):

                    output = model(ids, masks, start_idx=start_idx, end_idx=end_idx)

                    loss = output.loss

                    if key == "train":
                        if idx != 0:
                            loss.backward()  # Perform a backward pass to calculate the gradients
                        optimizer.step()  # Update parameters and take a step using the computed gradient
                        scheduler.step()  # Update learning rate schedule

                        # Clip the norm of the gradients to 1.0.
                        # This is to help prevent the "exploding gradients" problem.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    epoch_loss += loss.item() * len(ids)

                    # Move logits to CPU
                    # detaching these outputs so that the backward passes stop at this point
                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = output.start_logits.cpu().detach().numpy()
                    end_logits = output.end_logits.cpu().detach().numpy()
                    selected_text = data["selected_text"]

                    filtered_sentences = []
                    for i, t_data in enumerate(text):
                        # for i in range(len(ids)):
                        jaccard_score, filtered_output = utils.find_jaccard_score(
                            tokenizer,
                            t_data,
                            selected_text[i],
                            sentiment[i],
                            offsets[i],
                            start_logits[i],
                            end_logits[i],
                        )
                        epoch_jaccard += jaccard_score
                        filtered_sentences.append(filtered_output)

            # Calculate the average loss over the training data
            epoch_loss = epoch_loss / len(dataloaders.dataset)
            # Calculate the average jaccard score over the training data
            epoch_jaccard = epoch_jaccard / len(dataloaders.dataset)

            print(
                "Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}".format(
                    epoch + 1, num_epochs, key, epoch_loss, epoch_jaccard
                )
            )

            # Store the loss value for plotting the learning curve.
            if key == "train":
                train_loss.append(epoch_loss)
                jac_train.append(epoch_jaccard)

            else:
                val_loss.append(epoch_loss)
                jac_val.append(epoch_jaccard)

    torch.save(model.state_dict(), filename)


# stratifiedKfold validation
seed = 3
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
for fold, (idxTrain, idxVal) in enumerate(
    skf.split(train_df, train_df.sentiment), start=1
):

    print("#" * 10)
    print("### FOLD %i" % (fold))
    print("#" * 10)

    model = TextModel()
    optimizer = AdamW(
        model.parameters(), lr=3e-5, weight_decay=0.01, correct_bias=False
    )
    dataloaders_dict = train_val_dataloaders(
        train_df, idxTrain, idxVal, batch_size=opt.train_batch_size
    )
    num_training_steps = int(len(train_df) / opt.epoch * opt.train_batch_size)
    # warmup_proportion = float(num_warmup_steps) / float(num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # default #use a linear scheduler with no warmup steps
        num_training_steps=num_training_steps,
    )
    train(
        tokenizer,
        model,
        dataloaders_dict,
        optimizer,
        opt.epoch,
        scheduler,
        device,
        f"bert_fold{fold}.pth",
    )

t_loader = test_loader(test_df)
predictions = []
models = []
for fold in range(skf.n_splits):
    model = TextModel()
    model.to(device)
    model.load_state_dict(torch.load(f"./bert_fold{fold+1}.pth"))
    model.eval()
    models.append(model)

loader = tqdm(t_loader, total=len(t_loader))
for (idx, data) in enumerate(loader):
    ids = data["ids"].to(device)
    masks = data["masks"].to(device)
    text = data["text"]
    offsets = data["offsets"].numpy()

    start_logits = []
    end_logits = []
    for model in models:
        with torch.no_grad():
            output = model(ids, masks)
            start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
            end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

    start_logits = np.mean(start_logits, axis=0)
    end_logits = np.mean(end_logits, axis=0)

    for i, t_data in enumerate(text):
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        if start_pred >= end_pred:
            enc = tokenizer.encode(t_data)
            prediction = tokenizer.decode(enc.ids[start_pred - 1 : end_pred])
        else:
            prediction = utils.get_selected_text(
                t_data, start_pred, end_pred, offsets[i]
            )
        predictions.append(prediction)
