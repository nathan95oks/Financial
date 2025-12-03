
import pickle
import pandas as pd
import os

MODELS_DIR = "models"
HIST_FILE = f"{MODELS_DIR}/histories.pkl"

if not os.path.exists(HIST_FILE):
    print(f"ERROR: {HIST_FILE} not found.")
    exit(1)

print("Loading history...")
with open(HIST_FILE, "rb") as f:
    histories = pickle.load(f)

data = []
if 'histories' in locals():
    for name, hist in histories.items():
        best_acc = max(hist['val_acc'])
        best_loss = min(hist['val_loss'])
        total_time = hist.get('total_time', 0) 
        
        data.append({
            'Model': name,
            'Best Val Accuracy': best_acc,
            'Best Val Loss': best_loss,
            'Training Time (s)': total_time
        })
    
    df = pd.DataFrame(data).sort_values(by='Best Val Accuracy', ascending=False)
    print(df)
    
    if df['Training Time (s)'].sum() > 0:
        print("Verification successful! Times are present.")
    else:
        print("Verification FAILED! Times are still 0.")
