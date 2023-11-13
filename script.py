import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import time
import math
import pandas as pd
from pathlib import Path
from joblib import dump
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

# Set a non-interactive backend
matplotlib.use("Agg")

# find out dimension (shape) of the word embeddings
# print out the encoding/embedding for sentence - best representation of meaning of text
# use print statement within model class at each layer print the input and output shape, dimensions (shape) of inputs/outputs of each layer
# use a pytorch command to display summary of the layer shapes

# look at cosine dist between embedding for two setennces (same/similar meaning) and two opposite, two unrelated

# PCA on embedding vectors, scatter plot on set of 10 sentences

# linear reg to count num of tokens from embedding (unlabelled dataset?)
# repurposing - transfer learning

# focus on encoder

try:
    DATA_DIR = Path(__file__).resolve().parent / "data"
    ATTENTION_DIR = Path(__file__).resolve().parent / "attention"
    LOSS_DIR = Path(__file__).resolve().parent / "loss"
except NameError:
    DATA_DIR = Path.cwd() / "data"
    ATTENTION_DIR = Path.cwd() / "attention"
    LOSS_DIR = Path.cwd() / "loss"

# Ensure the directories exists
DATA_DIR.mkdir(exist_ok=True)
ATTENTION_DIR.mkdir(exist_ok=True)
LOSS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LANG1 = "eng"
LANG2 = "spa"
BATCH_SIZE = 32
EPOCHS = 30
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
MAX_LENGTH = 5
SOS_token = 0
EOS_token = 1
PAD_token = 2
eng_prefixes = (
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s ",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re ",
)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS, EOS and PAD

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(lang1="eng", lang2="spa", reverse=False):
    data_file = DATA_DIR / f"{lang2}.txt"

    # Read the data into a DataFrame
    df = pd.read_csv(data_file, sep="\t")
    df.columns = [lang1, lang2, "license"]

    # Normalize strings and create pairs
    pairs = [
        [normalizeString(s1), normalizeString(s2)]
        for s1, s2 in zip(list(df[lang1]), list(df[lang2]))
    ]

    # Reverse pairs and Lang instances if needed
    if reverse:
        pairs = [[p[1], p[0]] for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return (
        len(p[0].split(" ")) < MAX_LENGTH
        and len(p[1].split(" ")) < MAX_LENGTH
        and p[1].startswith(eng_prefixes)
    )


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)

    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        print("Input size:", input_size, "Hidden size:", hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Can increase GRU layers
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True
        )

    def forward(self, input):
        print("Input:", input, "Input shape:", input.shape)
        embedded = self.dropout(self.embedding(input))
        print("Embedding shape:", embedded.shape)
        output, hidden = self.gru(embedded)
        print("Output shape:", output.shape, "Hidden shape:", hidden.shape)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(SOS_token)
        print("Decoder input shape:", decoder_input.shape)
        decoder_hidden = encoder_hidden
        print("Decoder hidden state shape:", decoder_hidden.shape)
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(lang1, lang2, batch_size):
    input_lang, output_lang, pairs = prepareData(lang1, lang2, True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, num_workers=8
    )

    return input_lang, output_lang, train_dataloader, pairs


def train_epoch(
    dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
        )

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def save_model(encoder, decoder, path):
    dump(
        {
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        },
        path,
    )


def train_and_evaluate(
    train_dataloader, encoder, decoder, pairs, n_epochs, learning_rate=0.001
):
    start = time.time()
    training_log = []
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    # Select a random sentence pair to visualise the attention weights over training run
    visualization_pairs = random.sample(pairs, 1)

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )

        training_log.append((epoch, loss))

        print(
            "%s (Epoch %d, %d%%) Loss: %.4f"
            % (
                timeSince(start, epoch / n_epochs),
                epoch,
                epoch / n_epochs * 100,
                loss,
            )
        )

        if epoch % 2 == 0:
            for eng, spa in visualization_pairs:
                evaluateAndShowAttention(eng)

    # Saving the model
    save_model(encoder, decoder, DATA_DIR / "model_checkpoint.tar")

    return training_log


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden
        )

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig, ax = plt.subplots()
    cax = ax.matshow(attentions, cmap="bone")
    fig.colorbar(cax)

    # Set up axes
    # Set the locations of the x-ticks
    ax.set_xticks(range(len([""] + input_sentence.split(" ") + ["<EOS>"])))
    # Then set the labels with a FixedFormatter
    ax.set_xticklabels([""] + input_sentence.split(" ") + ["<EOS>"], rotation=90)

    # Do the same for y-ticks
    ax.set_yticks(range(len([""] + output_words)))
    ax.set_yticklabels([""] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Save the figure to a file
    timestamp_str = datetime.now().strftime("%I:%M")
    filename = f"attention_figure_{timestamp_str}.png"
    filename = filename.replace(":", "-")

    print("Saving attention plot...")
    plt.savefig(ATTENTION_DIR / filename)

    # Close the plot to free up memory
    plt.close(fig)


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder, decoder, input_sentence, input_lang, output_lang
    )
    print("input =", input_sentence)
    print("output =", " ".join(output_words))

    showAttention(
        input_sentence=input_sentence,
        output_words=output_words,
        # Remove the first entry in the attentions tensor,
        # Then move possible GPU tensor to the CPU (NumPy doesnt handle GPU)
        # Convert CPU tensor to NumPy array
        attentions=attentions.squeeze(0).cpu().numpy(),
    )


def plot_loss(training_log):
    # Extract epochs and loss values
    epochs, losses = zip(*training_log)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)

    print("\n Saving training log plot...")
    plt.savefig(LOSS_DIR / "training_loss.png")
    plt.close()  # Close the plot to free up memory


if __name__ == "__main__":
    input_lang, output_lang, train_dataloader, pairs = get_dataloader(
        LANG1, LANG2, BATCH_SIZE
    )

    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE, num_layers=1).to(device)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words).to(device)

    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = encoder_params + decoder_params

    print(f"Encoder has {encoder_params} parameters")
    print(f"Decoder has {decoder_params} parameters")
    print(f"The whole model has {total_params} parameters")

    print(encoder)
    print(decoder)

    training_log = train_and_evaluate(
        train_dataloader, encoder, decoder, pairs, EPOCHS, learning_rate=LEARNING_RATE
    )

    plot_loss(training_log)

# Function that does translation
# Learn about what BLEU is and implement it - edits/updates/deletion
# Save/load and prediction with joblib
# New script that loads this
# Understand the loss functions - log loss?
