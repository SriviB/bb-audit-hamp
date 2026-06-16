"""
Shared argparse builder for DP-SGD audit entry-points.

Centralises the ~50 arguments that were previously copy-pasted into every
audit script. Entry-points call `build_parser()` and may add their own
script-specific arguments on top.
"""
import argparse
from models import Models


def build_parser() -> argparse.ArgumentParser:
    """Return a fully configured ArgumentParser for parallel audit runs."""
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # ------------------------------------------------------------------
    # Distributed / system
    # ------------------------------------------------------------------
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for torchrun distributed training')

    # ------------------------------------------------------------------
    # Data and model
    # ------------------------------------------------------------------
    parser.add_argument('--data_name', type=str, default='mnist',
                        help='Dataset: mnist, cifar10, cifar100, purchase, tiny_shakespeare')
    parser.add_argument('--model_name', type=str, default='lr',
                        choices=list(Models.keys()), help='Model architecture')
    parser.add_argument('--n_df', type=int, default=0,
                        help='Dataset size |D| (0 = full dataset)')

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    parser.add_argument('--n_reps', type=int, default=200,
                        help='Number of shadow models to train')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Training epochs per model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--block_size', type=int,
                        help='Process batch in sub-blocks to save GPU memory')
    parser.add_argument('--aug_mult', type=int, default=1,
                        help='Augmentation multiplicity')
    parser.add_argument('--sampling', type=str, default='poisson',
                        choices=['poisson', 'shuffle'],
                        help='Minibatch sampling strategy. '
                             '"poisson": each sample included independently with prob q (default, '
                             'matches DP-SGD privacy analysis). '
                             '"shuffle": standard random-sampler DataLoader.')

    # ------------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------------
    parser.add_argument('--epsilon', type=float, default=None,
                        help='DP ε budget (None = non-private)')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='DP δ budget')
    parser.add_argument('--max_grad_norm', type=float, default=1,
                        help='Per-sample gradient clipping norm')

    # ------------------------------------------------------------------
    # Canary / target sample
    # ------------------------------------------------------------------
    parser.add_argument('--target_type', type=str, default='blank',
                        help='Canary type: blank, mislabeled, clipbkd, badnets, fgsm, '
                             'gradient_space_canary, empty_sequence, or path to .npy file')
    parser.add_argument('--canary_pt', type=str, default=None,
                        help='Path to a .pt canary file; overrides --target_type')
    parser.add_argument('--gradient_space_canary_pt', type=str, default=None,
                        help='Path to a pre-crafted gradient-space canary dict '
                             '(used with --target_type gradient_space_canary)')
    parser.add_argument('--mislabeled_target_class', type=int, default=1,
                        help='Target class for mislabeled canary')
    parser.add_argument('--blank_alpha', type=float, default=0.0,
                        help='Blank canary interpolation: 0 = all-zeros, 1 = label-9 image')
    parser.add_argument('--badnets_label', type=int, default=-1,
                        help='Label assigned to badnets canary')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class for gradient-space audit')

    # ------------------------------------------------------------------
    # Audit configuration
    # ------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=0,
                        help='Global random seed')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='',
                        help='Fix model init across reps (optionally supply a weight path)')
    parser.add_argument('--fit_world_only', type=str, default=None,
                        choices=['in', 'out'],
                        help='Train only "in" or only "out" models')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for empirical ε estimation')
    parser.add_argument('--holdout_audit', action='store_true',
                        help='Hold out half the reps for threshold selection')
    parser.add_argument('--store_all_losses', action='store_true',
                        help='Save per-sample training losses for every rep')
    parser.add_argument('--store_canary_rank', action='store_true')
    parser.add_argument('--view_badnets', action='store_true')
    parser.add_argument('--out', type=str, default='exp_data/',
                        help='Output directory')

    # ------------------------------------------------------------------
    # Defense (gradient-norm filtering)
    # ------------------------------------------------------------------
    parser.add_argument('--defense', action='store_true',
                        help='Enable per-class gradient-norm defense')
    parser.add_argument('--defense_k', type=int, default=5,
                        help='Samples dropped per class per filter epoch')
    parser.add_argument('--defense_apply_ascent', action='store_true', default=False,
                        help='Apply gradient ascent instead of dropping flagged samples')
    parser.add_argument('--defense_filter_every', type=int, default=1,
                        help='Apply defense every N epochs')
    parser.add_argument('--defense_score_norm', type=str, default='linf',
                        choices=['linf', 'l2', 'l1'],
                        help='Norm used to score per-sample gradients')
    parser.add_argument('--defense_score_fn', type=str, default='grad_norm',
                        choices=[
                            'grad_norm', 'grad_norm_unclipped', 'grad_norm_percentile',
                            'grad_dir_volatility', 'grad_dir_volatility_unclipped',
                            'rand_proj_var', 'rand_proj_var_unclipped',
                            'maxmin_proj_ratio', 'maxmin_proj_ratio_unclipped',
                            'gradient_rank', 'gradient_rank_unclipped',
                            'grad_accel', 'grad_accel_unclipped',
                            'grad_jerk', 'grad_jerk_unclipped',
                            'norm_x_dir_uniqueness', 'norm_x_dir_uniqueness_unclipped',
                            'alignment_with_rand_proj', 'alignment_with_rand_proj_unclipped',
                            'gradient_sparsity', 'gradient_sparsity_unclipped',
                            'gradient_kurtosis', 'gradient_kurtosis_unclipped',
                            'grad_dir_change_rate', 'grad_dir_change_rate_unclipped',
                            'norm_x_trajectory_orth', 'norm_x_trajectory_orth_unclipped',
                            'gradient_scatter', 'gradient_scatter_unclipped',
                            'fisher', 'fisher_unclipped',
                            'grad_norm_x_loss', 'grad_norm_x_loss_unclipped',
                            'cos_update', 'cos_update_unclipped',
                            'cos_theta0', 'cos_theta0_unclipped',
                            'inv_confidence', 'prediction_margin', 'pred_entropy',
                            'loss', 'loss_momentum', 'loss_volatility',
                        ],
                        help='Scoring function used by the defense filter')

    # Defense scoring hyperparameters
    parser.add_argument('--grad_norm_percentile_k', type=int, default=20)
    parser.add_argument('--grad_dir_volatility_k', type=int, default=5)
    parser.add_argument('--grad_dir_proj_dim', type=int, default=64)
    parser.add_argument('--grad_dir_proj_seed', type=int, default=0)
    parser.add_argument('--dir_unique_k', type=int, default=5)
    parser.add_argument('--rand_proj_var_m', type=int, default=10)
    parser.add_argument('--rand_proj_var_seed', type=int, default=0)
    parser.add_argument('--maxmin_proj_k', type=int, default=10)
    parser.add_argument('--maxmin_proj_seed', type=int, default=0)
    parser.add_argument('--grad_rank_mode', type=str, default='effdim',
                        choices=['effdim', 'entropy'])
    parser.add_argument('--grad_rank_eps', type=float, default=1e-12)
    parser.add_argument('--grad_accel_proj_dim', type=int, default=64)
    parser.add_argument('--grad_accel_proj_seed', type=int, default=0)
    parser.add_argument('--grad_jerk_proj_dim', type=int, default=64)
    parser.add_argument('--grad_jerk_proj_seed', type=int, default=0)
    parser.add_argument('--alignment_proj_k', type=int, default=10)
    parser.add_argument('--alignment_proj_seed', type=int, default=0)
    parser.add_argument('--grad_scatter_k', type=int, default=5)
    parser.add_argument('--loss_volatility_k', type=int, default=5)

    return parser
