import glob
from validate import WordErrorRate
from functions import create_model
from tqdm import tqdm
import torch
import torchaudio
import json

config_file = "configs/EfficientConformerCTCMedium.json"
checkpoint_file = "/kaggle/input/ecv-cp-test-check/checkpoints_56_90h_07.ckpt"

wer = WordErrorRate()

total_diff_score = 0.0
total_tokens = 0.0
count_print = 0

# Load Config
with open(config_file) as json_config:
    config = json.load(json_config)

# Device
device = torch.device("cuda:0")
print("Device:", device)

model = create_model(config).to(device)
model.eval()

# Load Model
model.load(checkpoint_file)

def transcriber(wav_path):
    audio, _ = torchaudio.load(wav_path)
    text = model.beam_search_decoding(audio.to(device), x_len=torch.tensor([len(audio[0])], device=device))[0] # you can test with model.gready_search_decoding
    return text

test_wavs_path = glob.glob("/kaggle/input/vivos-vietnamese-speech-corpus-for-asr/vivos/test/waves/*.wav") # => change your folder test dataset here !

with open("data/test_wers_report.txt", "w", encoding="utf8") as fw:
    for wav_path in tqdm(test_wavs_path):
        hyp = transcriber(wav_path).lower()
        hyp = hyp.replace(".", "")
        txt_path = wav_path.replace(".wav", ".txt")
        ref = open(txt_path, "r", encoding="utf8").readlines()[0].replace("\n", "").lower()
        ref = " ".join(ref.split())
        hyp = " ".join(hyp.split())
        diff_score = wer.diff_words(ref, hyp)
        total_diff_score += diff_score
        total_tokens += wer.n_tokens
        if count_print % 100 == 0:
            print(str(wav_path) +  " => " + hyp + " <= " + ref)
        fw.write(str(wav_path) + " => " + hyp + " <= " + ref + "\n")
        count_print += 1

print("WER: " + str(total_diff_score / total_tokens))