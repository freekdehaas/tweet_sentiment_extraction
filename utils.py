import random
import torch
import numpy as np

# predict the selected_text using outputs of the model
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    d = len(a) + len(b) - len(c)
    if d != 0:
        return float(len(c) / d)
    else:
        return 0.0


# evaluation metric using Jaccard
def get_selected_text(text, start_idx, end_idx, offsets):
    if end_idx < start_idx:
        end_idx = start_idx
    select_text = ""
    for idx in range(start_idx, end_idx + 1):
        select_text += text[offsets[idx][0] : offsets[idx][1]]
        if (idx + 1) < len(offsets) and offsets[idx][1] < offsets[idx + 1][0]:
            select_text += " "
    return select_text


def set_seed(seed):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def find_jaccard_score(
    tokenizer, text, selected_text, sentiment, offsets, start_logits, end_logits
):  # start_idx, end_idx
    start_pred = np.argmax(start_logits)  # Predicted start index using argmax
    end_pred = np.argmax(end_logits)  # Predicted end index using argmax
    if (end_pred <= start_pred) or sentiment == "neutral" or len(text.split()) < 2:
        enc = tokenizer(text)
        prediction = tokenizer.decode(enc["input_ids"][start_pred - 1 : end_pred])
    else:
        prediction = get_selected_text(text, start_pred, end_pred, offsets)
    true = selected_text
    # true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, prediction), prediction


seed = 777
set_seed(seed)
