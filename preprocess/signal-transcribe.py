import os
import json
from funasr import AutoModel

wav_root = "./data/interspeech"
transcribe_root = "./data/interspeech/transcriptions"
model = AutoModel(
    model="paraformer-zh", model_revision="v2.0.4",
    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
    device="cuda"
)
wav_root = "data/interspeech/task-0"
wav_names = ['0089-0.wav',
'0187-0.wav',
'0198-0.wav',
'0241-0.wav',
'0257-0.wav',
'0293-0.wav',
'0316-0.wav',
'0322-0.wav',
'0387-0.wav',
'0401-0.wav',
'0407-0.wav'
]

save_root = "data/interspeech/transcriptions/task-0"
for wav_name in wav_names:
    wav_file = os.path.join(wav_root, wav_name)
    save_path = os.path.join(save_root, wav_name.lower().replace('.wav', '.json'))
    try:
        result_save = []
        res = model.generate(input=wav_file, sentence_timestamp=True)[0]
        for sentence in res["sentence_info"]:
            result_save.append(
                {
                    "start": sentence["timestamp"][0][0],
                    "end": sentence["timestamp"][-1][-1],
                    "text": sentence["text"]
                }
            )
        with open(save_path, 'w') as f:
            json.dump(result_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(e)
        with open(os.path.join(transcribe_root, "fail_list.txt"), 'a+') as f:
            f.write(wav_name + '\n')
