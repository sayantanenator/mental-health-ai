# create_sample_data.py
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

sample_data = {
    'text': [
        "I've been feeling really good lately",
        "Some days are harder than others",
        "I've been having constant negative thoughts"
    ],
    'audio_path': [
        "data/audio/sample1.wav",
        "data/audio/sample2.wav",
        "data/audio/sample3.wav"
    ],
    'label': [0, 1, 2]
}

df = pd.DataFrame(sample_data)
df.to_csv("data/processed/training_data.csv", index=False)
print("Sample data created!")
