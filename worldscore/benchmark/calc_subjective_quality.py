import glob
import os
from worldscore.benchmark.metrics.third_party.clip_mlp_aesthetic_metrics import CLIPMLPAestheticScoreMetric
import pandas as pd
import json

s_metric = CLIPMLPAestheticScoreMetric()

def calculate_subjective_quality(frame_folder_path):
    image_list = sorted(glob.glob(os.path.join(frame_folder_path, "*.png")))
    
    if not image_list:
        return None

    s_quality = s_metric._compute_scores(image_list)
    
    return s_quality

if __name__ == "__main__":
    parent_folder = "/home/ubuntu/world-model-eval/video_frames_real" 
    
    results = {}

    subfolders = sorted(os.listdir(parent_folder))

    print(f"{'Folder Name':<20} | {'CLIP Aesthetic Score':<15}")
    print("-" * 40)

    with open("subjective_results_real.jsonl", "w") as jsonl_file:
        for folder_name in subfolders:
            folder_path = os.path.join(parent_folder, folder_name)

            if os.path.isdir(folder_path):
                score = calculate_subjective_quality(folder_path)
                
                if score is not None:
                    results[folder_name] = score
                    print(f"{folder_name:<20} | {score:.4f}")
                    jsonl_file.write(json.dumps({"folder": folder_name, "subjective_results": score}) + "\n")
                    jsonl_file.flush()
                else:
                    print(f"{folder_name:<20} | No images found.")

    df = pd.DataFrame(list(results.items()), columns=['Folder', 'Subjective Quality'])
    df.to_csv("subjective_results_real.csv", index=False)