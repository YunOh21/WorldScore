import glob
import os
from worldscore.benchmark.metrics.third_party.flow_aepe_metrics_raft import OpticalFlowAverageEndPointErrorMetric
import pandas as pd
import json

t_metric = OpticalFlowAverageEndPointErrorMetric()

def calculate_droid_score(frame_folder_path):
    image_list = sorted(glob.glob(os.path.join(frame_folder_path, "*.png")))
    
    if not image_list:
        return None

    t_error = t_metric._compute_scores(image_list)
    
    return g_error

if __name__ == "__main__":
    parent_folder = "/home/ubuntu/world-model-eval/video_frames" 
    
    results = {}

    subfolders = sorted(os.listdir(parent_folder))

    print(f"{'Folder Name':<20} | {'RAFT Score':<15}")
    print("-" * 40)

    with open("raft_results.jsonl", "w") as jsonl_file:
        for folder_name in subfolders:
            folder_path = os.path.join(parent_folder, folder_name)

            if os.path.isdir(folder_path):
                score = calculate_droid_score(folder_path)
                
                if score is not None:
                    results[folder_name] = score
                    print(f"{folder_name:<20} | {score:.4f}")
                    jsonl_file.write(json.dumps({"folder": folder_name, "raft_score": score}) + "\n")
                    jsonl_file.flush()
                else:
                    print(f"{folder_name:<20} | No images found.")

    df = pd.DataFrame(list(results.items()), columns=['Folder', 'RAFT Score'])
    df.to_csv("raft_results.csv", index=False)