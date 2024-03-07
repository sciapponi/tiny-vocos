from vocos import Vocos 
from vocos.feature_extractors import MelSpectrogramFeatures
import random
import torchaudio
from matplotlib import pyplot as plt
import torch

wavs = ["/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ006-0018.wav",
        "/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ007-0243.wav",
        "/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ014-0101.wav",
        "/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ019-0240.wav",
        "/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ029-0188.wav",
        "/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ035-0162.wav",
        "/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ048-0035.wav",
        "/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ025-0091.wav",
        "/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ048-0063.wav",
        "/workspace/datasets/speech/LJSpeech-1.1/wavs/LJ050-0113.wav"]

model_configs = ["/workspace/projects/sciapponi/tiny-vocos/logs/lj_phinet_128_nfft512/lightning_logs/version_0/",
                    "/workspace/projects/sciapponi/tiny-vocos/logs/xivocos_1.5mb/lightning_logs/version_5/",
                    "/workspace/projects/sciapponi/tiny-vocos/logs/lj_tfvocos_128_128_bn/lightning_logs/version_2/"]

# model = Vocos.from_hparams("/raid/home/e3da/projects/sciapponi/tiny-vocos/logs/lj_phinet_128_nfft512/lightning_logs/version_0/config.yaml")

path = "/home/ste/Datasets/LJSpeech-1.1/wavs/LJ001-0001.wav"
fe = MelSpectrogramFeatures()
num_samples = 48384

for idx, conf in zip(["phinet", "xinet", "convnext"], model_configs):
    model = Vocos.from_hparams(f"{conf}/config.yaml")
    checkpoint = torch.load(f"{conf}/checkpoints/last.ckpt")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    for i, wav in enumerate(wavs):
        y, sr = torchaudio.load(wav)
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
        y = y[:, : num_samples]
        spec = fe(y)
        # print(spec)
        vocos_output = model.decode(spec)
        print(vocos_output)
        # Upsample to 44100 Hz for better reproduction on audio hardware
        vocos_output = torchaudio.functional.resample(vocos_output, orig_freq=24000, new_freq=44100).cpu()
        torchaudio.save(f"audio/{idx}_{i}.mp3", vocos_output, 44100, compression=128)