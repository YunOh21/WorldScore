import glob
import os
from worldscore.benchmark.metrics.third_party.reprojection_error_metrics import ReprojectionErrorMetric
import pandas as pd
import json

g_metric = ReprojectionErrorMetric()

def calculate_droid_score(frame_folder_path):
    image_list = sorted(glob.glob(os.path.join(frame_folder_path, "*.png")))
    
    if not image_list:
        return None

    g_error = g_metric._compute_scores(image_list)
    
    return g_error

if __name__ == "__main__":
    parent_folder = "/home/ubuntu/world-model-eval/video_frames_real" 
    
    results = {}

    subfolders = sorted(os.listdir(parent_folder))

    print(f"{'Folder Name':<20} | {'DROID Score':<15}")
    print("-" * 40)

    with open("droid_results_real.jsonl", "w") as jsonl_file:
        for folder_name in subfolders:
            folder_path = os.path.join(parent_folder, folder_name)

            if os.path.isdir(folder_path):
                score = calculate_droid_score(folder_path)
                
                if score is not None:
                    results[folder_name] = score
                    print(f"{folder_name:<20} | {score:.4f}")
                    jsonl_file.write(json.dumps({"folder": folder_name, "droid_score": score}) + "\n")
                    jsonl_file.flush()
                else:
                    print(f"{folder_name:<20} | No images found.")

    df = pd.DataFrame(list(results.items()), columns=['Folder', 'DROID Score'])
    df.to_csv("droid_results_real.csv", index=False)