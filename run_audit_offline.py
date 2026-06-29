"""
Offline audit script: combines per-rank output files and runs the
leave-one-out logistic regression attack.
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, "misleading-privacy-evals/src")
sys.path.insert(0, ".")
from parallel_audit_model_hamp import audit_multi_canary

output_dir = Path("debug_audit_hamp_64")
num_shadow = 64
num_canaries = 500
num_ranks = 20
seed = 42
num_samples = 50000
alpha = 0.05
delta = 1e-5

# Reconstruct full arrays from per-rank files
correctness_full = np.zeros((num_canaries, num_shadow, 18), dtype=np.float32)
all_train_accs = np.zeros(num_shadow, dtype=np.float32)
all_test_accs = np.zeros(num_shadow, dtype=np.float32)

for r in range(num_ranks):
    indices = np.load(output_dir / f"shadow_indices_rank{r}.npy")
    bvecs = np.load(output_dir / f"binary_vectors_rank{r}.npy")
    tr = np.load(output_dir / f"train_accs_rank{r}.npy")
    te = np.load(output_dir / f"test_accs_rank{r}.npy")
    for local_i, shadow_idx in enumerate(indices):
        correctness_full[:, shadow_idx, :] = bvecs[local_i]
        all_train_accs[shadow_idx] = tr[local_i]
        all_test_accs[shadow_idx] = te[local_i]

print(f"Avg Train Acc: {all_train_accs.mean()*100:.2f}%")
print(f"Avg Test Acc:  {all_test_accs.mean()*100:.2f}%")

# Regenerate membership mask (must match training code exactly)
rng = np.random.default_rng(seed)
canary_order = rng.permutation(num_samples)
canary_indices = canary_order[:num_canaries]

rng_splits = np.random.default_rng(seed + 42)
uniforms = rng_splits.uniform(size=(num_shadow, num_samples))
shadow_in_indices_t = np.argsort(uniforms, axis=0)[:num_shadow // 2].T
shadow_membership_mask = np.zeros((num_samples, num_shadow), dtype=bool)
for sample_idx in range(num_samples):
    shadow_membership_mask[sample_idx, shadow_in_indices_t[sample_idx]] = True
canary_mask_arr = np.zeros(num_samples, dtype=bool)
canary_mask_arr[canary_indices] = True
shadow_membership_mask[~canary_mask_arr] = True

# Run audit with both C values
print("=" * 60)
for C_val, label in [(10.0, "Tuned C (10.0)"), (1.0, "Default C (1.0)")]:
    result = audit_multi_canary(
        correctness_full, shadow_membership_mask, canary_indices,
        C_val, seed, alpha, delta
    )
    ba = result["balanced_accuracy"]
    ee = result["emp_eps"]
    print(f"{label}:")
    print(f"  Balanced Accuracy : {ba:.4f}")
    print(f"  Empirical Epsilon : {ee:.6f}")
    for fpr_val, tpr_val in result["tpr_at_fpr"].items():
        print(f"  TPR at FPR {fpr_val*100:.1f}% : {tpr_val*100:.4f}%")
    print("-" * 40)
print("=" * 60)
