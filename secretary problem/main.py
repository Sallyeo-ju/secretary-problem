import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from methods import train_full, train_early_stopping, train_secretary

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# ==============================
# LOAD AND PREPROCESS MNIST
# ==============================
print("Loading MNIST...")
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Better: shuffled + stratified split
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}\n")

# ==============================
# EXPERIMENT CONFIGURATION
# ==============================
REPETITIONS = 10

methods = {
    'Full Training':  lambda: train_full(x_train, y_train, x_val, y_val, x_test, y_test),
    'Early Stopping': lambda: train_early_stopping(x_train, y_train, x_val, y_val, x_test, y_test),
    'Secretary 37%':  lambda: train_secretary(x_train, y_train, x_val, y_val, x_test, y_test, 0.37),
    'Secretary 50%':  lambda: train_secretary(x_train, y_train, x_val, y_val, x_test, y_test, 0.50),
    'Secretary 75%':  lambda: train_secretary(x_train, y_train, x_val, y_val, x_test, y_test, 0.75),
}

# ==============================
# RUN EXPERIMENTS
# ==============================
all_results = []

for method_name, method_fn in methods.items():
    print(f"\n{'='*50}")
    print(f"Method: {method_name}")
    print(f"{'='*50}")

    rep_bar = tqdm(
        range(1, REPETITIONS + 1),
        desc="  Repetitions",
        unit="rep",
        bar_format='{l_bar}{bar:20}{r_bar}'
    )

    for rep in rep_bar:
        rep_bar.set_description(f"  Rep {rep}/{REPETITIONS}")

        # Reproducibility: different but controlled random seed each repetition
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(rep)

        epochs_used, test_acc = method_fn()
        rep_bar.set_postfix({'epochs': epochs_used, 'acc': f'{test_acc:.4f}'})

        all_results.append({
            'Method': method_name,
            'Repetition': rep,
            'Epochs Used': epochs_used,
            'Test Accuracy': test_acc
        })

# ==============================
# SAVE AND SUMMARIZE
# ==============================
df = pd.DataFrame(all_results)
df.to_csv('results/raw_results.csv', index=False)

print("\n\n========== SUMMARY ==========")

full_mean_epochs = df[df['Method'] == 'Full Training']['Epochs Used'].mean()
full_mean_acc = df[df['Method'] == 'Full Training']['Test Accuracy'].mean()

summary_rows = []
for method_name in methods.keys():
    subset = df[df['Method'] == method_name]
    mean_ep = subset['Epochs Used'].mean()
    std_ep = subset['Epochs Used'].std()
    mean_acc = subset['Test Accuracy'].mean()
    std_acc = subset['Test Accuracy'].std()
    reduction = (full_mean_epochs - mean_ep) / full_mean_epochs * 100
    acc_diff = mean_acc - full_mean_acc

    summary_rows.append({
        'Method': method_name,
        'Mean Epochs': round(mean_ep, 2),
        'Std Epochs': round(std_ep, 2),
        'Mean Accuracy': round(mean_acc, 4),
        'Std Accuracy': round(std_acc, 4),
        'Epoch Reduction (%)': round(reduction, 2),
        'Accuracy vs Full': round(acc_diff, 4)
    })

    print(f"\n{method_name}")
    print(f"  Epochs  : {mean_ep:.1f} ± {std_ep:.1f}  (reduction: {reduction:.1f}%)")
    print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}  (vs full: {acc_diff:+.4f})")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('results/summary.csv', index=False)

print("\nResults saved to results/ folder.")