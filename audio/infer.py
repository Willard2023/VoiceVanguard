# Usage: python audio/infer.py ft_checkpoint_path checkpoint_name

import os
import json
import math
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor
import torchaudio

from audio_model import Model

checkpoint = "best"
if len(sys.argv) > 1:
    ft_model_path = sys.argv[1]
else:
    ft_model_path = "results/task2/audio/lr5e-5"
if len(sys.argv) == 3:
    checkpoint = sys.argv[2]

with open(os.path.join(ft_model_path, "model_config.json")) as f:
    train_config = json.load(f)
pretrain_model_path = train_config["model_path"]
remove_silence = train_config["remove_silence"]
window_length_second = train_config["window_length"]
step_length_second = train_config["step_length"]
only_first_window = train_config["only_first_window"] if "only_first_window" in train_config else False

feature_extractor = AutoFeatureExtractor.from_pretrained(pretrain_model_path)
model = Model(pretrain_model_path, num_label=train_config["num_label"], dropout=train_config["dropout"])
model.load_state_dict(torch.load(os.path.join(ft_model_path, f"{checkpoint}_model.pth")))
model.to("cuda")
model.eval()
print("Model loaded")

task = train_config["task"]

feature_dic = {}
logits_dic = {}
predictions_dic = {}  # 用于存储每个id的最终预测结果

audio_path = f"data/interspeech/test/task-{task}"
segment_path = f"data/interspeech/transcriptions/test/task-{task}"
with torch.no_grad():
    # bar = tqdm(total=len(os.listdir(audio_path)))
    for audio_file in os.listdir(audio_path):
        id = audio_file.split('-')[0]
        # print(id)
        audio, sr = torchaudio.load(os.path.join(audio_path, audio_file))
        assert sr == 16000
        if remove_silence:
            json_file = os.path.join(segment_path, f"{id}-{task}.json")
            with open(json_file) as f:
                utt_list = json.load(f)
            audio = torch.concat([audio[:, int(utt["start"]*16): int(utt["end"]*16)] for utt in utt_list], dim=1)

        audio_sec = audio.shape[1] / 16000
        window_num = math.ceil((audio_sec - (window_length_second - step_length_second)) / step_length_second) - 1
        if window_num <= 1:
            input_values = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")["input_values"][0]
            logit = model(input_values.to(model.backbone_model.device))
            
            prediction = torch.argmax(logit, dim=-1).item()  # 获取预测结果
        else:
            feature_list = []
            logit_list = []
            if only_first_window:
                window_num = 1
            for i in range(window_num):
                start = i * step_length_second * 16000
                end = (i * step_length_second + window_length_second) * 16000
                audio_input = audio[:, start:end]
                input_values = feature_extractor(audio_input, sampling_rate=sr, return_tensors="pt")["input_values"][0]
                logit = model(input_values.to(model.backbone_model.device))
                logit_list.append(logit)
            logit = torch.concat(logit_list, dim=0).mean(dim=0)
            prediction = torch.argmax(logit, dim=-1).item()  # 获取预测结果
        # print(f"id: {id}, prediction: {prediction}")
        predictions_dic[id] = prediction  # 存储每个id的最终预测结果
        # bar.update(1)


# 将 predictions_dic 保存为 JSON 文件
with open(os.path.join(f"test_predictions.json"), "w") as f:
    json.dump(predictions_dic, f, indent=4)

print("Predictions saved to JSON file.")