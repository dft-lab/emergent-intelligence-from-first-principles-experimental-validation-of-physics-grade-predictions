"""
Experimental Validation of Data Field Theory Predictions
Author: Mohammadreza Nehzati
Repository: https://github.com/dft-labs/emergent-intelligence-from-first-principles-experimental-validation-of-physics-grade-predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import seaborn as sns
from dft_core import DataFieldTheory, generate_hierarchical_data
import json
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'figure.autolayout': True,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.fontsize': 10,
})


def experiment_p1_critical_phenomena(n_seeds: int = 20, save_path: str = 'results_p1.json'):
    """
    Experiment P1: Critical phenomena during concept formation.

    Measures:
    1. Correlation length divergence xi(t) ~ |t-t_c|^{-nu}
    2. 1/f noise spectrum S(f) ~ f^{-alpha}
    3. Finite-size scaling and data collapse
    """
    print("=" * 60)
    print("Experiment P1: Critical Phenomena")
    print("=" * 60)

    results = {
        'n_seeds': n_seeds,
        'critical_exponents': [],
        'psd_exponents': [],
        'finite_size_scaling': {}
    }

    N_values = [642, 2562, 10242]

    for N in N_values:
        print(f"\nRunning finite-size scaling for N={N}")
        xi_peaks = []
        t_c_values = []

        for seed in tqdm(range(n_seeds)):
            # alpha > lambda_1 = 2 ensures spatial (ell>=1) modes are
            # linearly unstable, producing genuine critical dynamics.
            model = DataFieldTheory(N=N, alpha=3.0, seed=seed)
            model.initialize_field(k=8)

            X, y, y_onehot, _ = generate_hierarchical_data(
                N_samples=500, n_categories=8, manifold='sphere', seed=seed
            )

            rng = np.random.default_rng(seed)
            n_steps = 200
            xi_history = []

            for step in range(n_steps):
                batch_size = 32
                idx = rng.choice(len(X), batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y_onehot[idx]

                model.step(X_batch, y_batch)

                if step % 5 == 0:
                    xi = model.correlation_length()
                    xi_history.append((step, xi))

            steps, xi_vals = zip(*xi_history)
            t_c_idx = np.argmax(xi_vals)
            t_c = steps[t_c_idx]
            xi_peak = xi_vals[t_c_idx]

            xi_peaks.append(xi_peak)
            t_c_values.append(t_c)

            # Fit power-law divergence around t_c
            if t_c_idx >= 2 and t_c_idx < len(steps) - 2:
                window = min(10, t_c_idx, len(steps) - 1 - t_c_idx)
                start = max(0, t_c_idx - window)
                end = min(len(steps), t_c_idx + window + 1)

                t_window = np.array(steps[start:end])
                xi_window = np.array(xi_vals[start:end])

                # Exclude the singular point at t_c from the fit
                mask = t_window != t_c
                t_fit = t_window[mask]
                xi_fit = xi_window[mask]

                def power_law(t, A, nu):
                    return A * np.abs(t - t_c) ** (-nu)

                if len(t_fit) > 2:
                    try:
                        popt, _ = optimize.curve_fit(
                            power_law, t_fit, xi_fit,
                            p0=[xi_peak, 0.63], bounds=(0, [10, 2]),
                            maxfev=5000
                        )
                        results['critical_exponents'].append({
                            'N': N, 'seed': seed, 'nu': float(popt[1]),
                            't_c': float(t_c), 'xi_peak': float(xi_peak)
                        })
                    except (RuntimeError, ValueError) as e:
                        logger.debug(f"Power-law fit failed for N={N}, seed={seed}: {e}")

            # Compute PSD for 1/f analysis
            if seed < 5:
                xi_detrended = signal.detrend(xi_vals)

                f, Pxx = signal.welch(
                    xi_detrended, fs=0.2, nperseg=min(128, len(xi_detrended)),
                    window='hann'
                )

                mask = (f > 0.01) & (f < 0.5)
                if np.sum(mask) > 3:
                    f_sel = f[mask]
                    Pxx_sel = Pxx[mask]
                    # Filter out zero/negative PSD values before log
                    pos = Pxx_sel > 0
                    if np.sum(pos) > 3:
                        coeffs = np.polyfit(
                            np.log(f_sel[pos]), np.log(Pxx_sel[pos]), 1
                        )
                        alpha = -coeffs[0]
                        if np.isfinite(alpha):
                            results['psd_exponents'].append({
                                'N': N, 'seed': seed, 'alpha': float(alpha)
                            })

        # Store finite-size scaling results using string keys for JSON compatibility
        results['finite_size_scaling'][str(N)] = {
            'mean_xi_peak': float(np.mean(xi_peaks)),
            'std_xi_peak': float(np.std(xi_peaks)),
            'mean_t_c': float(np.mean(t_c_values)),
            'std_t_c': float(np.std(t_c_values))
        }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_path}")

    nu_vals = [exp['nu'] for exp in results['critical_exponents']]
    alpha_vals = [exp['alpha'] for exp in results['psd_exponents']]

    if nu_vals:
        print(f"\nSummary Statistics:")
        print(f"Critical exponent nu: {np.mean(nu_vals):.3f} +/- {np.std(nu_vals):.3f}")
    if alpha_vals:
        print(f"PSD exponent alpha: {np.mean(alpha_vals):.3f} +/- {np.std(alpha_vals):.3f}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    if nu_vals:
        ax.hist(nu_vals, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(nu_vals), color='red', linestyle='--',
                   label=f'Mean: {np.mean(nu_vals):.3f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
    ax.set_xlabel(r'Critical exponent $\nu$')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of $\\nu$ (n={len(nu_vals)})')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if alpha_vals:
        ax.hist(alpha_vals, bins=15, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(1.0, color='black', linestyle='--', label='1/f reference')
        ax.axvline(np.mean(alpha_vals), color='red', linestyle='-',
                   label=f'Mean: {np.mean(alpha_vals):.3f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
    ax.set_xlabel(r'PSD exponent $\alpha$')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of $\\alpha$ (n={len(alpha_vals)})')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    Ns = sorted(results['finite_size_scaling'].keys(), key=int)
    xi_means = [results['finite_size_scaling'][N_key]['mean_xi_peak'] for N_key in Ns]
    xi_stds = [results['finite_size_scaling'][N_key]['std_xi_peak'] for N_key in Ns]
    Ns_int = [int(n) for n in Ns]

    ax.errorbar(Ns_int, xi_means, yerr=xi_stds, fmt='o-', capsize=5, linewidth=2)
    ax.set_xlabel('System size N')
    ax.set_ylabel(r'Peak correlation length $\xi_{\mathrm{peak}}$')
    ax.set_title('Finite-size scaling')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_p1_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results


def experiment_p2_mass_robustness(n_configs: int = 50, save_path: str = 'results_p2.json'):
    """
    Experiment P2: Mass-robustness law.

    Tests: epsilon_OOD proportional to m_gap^{-2}
    """
    print("=" * 60)
    print("Experiment P2: Mass-Robustness Law")
    print("=" * 60)

    results = {
        'n_configs': n_configs,
        'mass_gap_vals': [],
        'ood_errors': [],
        'correlations': []
    }

    alphas = np.linspace(0.1, 1.9, n_configs)

    # Fixed training data across all configs so the only variable is alpha.
    X_train, y_train, y_train_onehot, centers = generate_hierarchical_data(
        N_samples=500, n_categories=8, manifold='sphere', seed=42
    )

    # OOD data: Gaussian noise added to training features, projected
    # back to the sphere.  Models with higher mass gap have smoother
    # fields and interpolate better under feature noise.
    rng_ood = np.random.default_rng(12345)
    noise_level = 0.3
    X_ood = X_train + noise_level * rng_ood.standard_normal(X_train.shape)
    X_ood = X_ood / np.linalg.norm(X_ood, axis=1, keepdims=True)
    y_ood = y_train

    for i, alpha in enumerate(tqdm(alphas)):
        model = DataFieldTheory(N=2562, alpha=alpha, seed=i)
        model.initialize_field(k=8)

        rng_train = np.random.default_rng(i)
        n_steps = 150
        for step in range(n_steps):
            batch_size = 32
            idx = rng_train.choice(len(X_train), batch_size, replace=False)
            X_batch = X_train[idx]
            y_batch = y_train_onehot[idx]
            model.step(X_batch, y_batch)

        m_gap = model.physical_mass_gap()
        y_pred = model.predict(X_ood)
        ood_error = 1 - np.mean(y_pred == y_ood)

        results['mass_gap_vals'].append(float(m_gap))
        results['ood_errors'].append(float(ood_error))

    m_gap_vals = np.array(results['mass_gap_vals'])
    ood_errors = np.array(results['ood_errors'])

    # Filter out zero/negative mass gaps for fitting
    valid = m_gap_vals > 1e-10
    m_valid = m_gap_vals[valid]
    e_valid = ood_errors[valid]

    def model_inverse_square(m, A):
        return A / m**2

    def model_inverse(m, A):
        return A / m

    def model_exponential(m, A, kappa):
        return A * np.exp(-kappa * m)

    try:
        popt1, _ = optimize.curve_fit(model_inverse_square, m_valid, e_valid, p0=[0.5])
        popt2, _ = optimize.curve_fit(model_inverse, m_valid, e_valid, p0=[0.5])
        popt3, _ = optimize.curve_fit(model_exponential, m_valid, e_valid, p0=[1.0, 1.0])

        n = len(e_valid)
        res1 = e_valid - model_inverse_square(m_valid, *popt1)
        res2 = e_valid - model_inverse(m_valid, *popt2)
        res3 = e_valid - model_exponential(m_valid, *popt3)

        # AIC = n*log(RSS/n) + 2k
        rss1 = np.sum(res1**2)
        rss2 = np.sum(res2**2)
        rss3 = np.sum(res3**2)

        aic1 = n * np.log(rss1 / n) + 2 * 1
        aic2 = n * np.log(rss2 / n) + 2 * 1
        aic3 = n * np.log(rss3 / n) + 2 * 2

        ss_total = np.sum((e_valid - np.mean(e_valid))**2)

        results['fits'] = {
            'inverse_square': {
                'params': popt1.tolist(),
                'AIC': float(aic1),
                'R2': float(1 - rss1 / ss_total)
            },
            'inverse': {
                'params': popt2.tolist(),
                'AIC': float(aic2),
                'R2': float(1 - rss2 / ss_total)
            },
            'exponential': {
                'params': popt3.tolist(),
                'AIC': float(aic3),
                'R2': float(1 - rss3 / ss_total)
            }
        }

        r, p = stats.pearsonr(m_valid, e_valid)
        results['correlations'].append({
            'pearson_r': float(r),
            'pearson_p': float(p)
        })

    except (RuntimeError, ValueError) as e:
        logger.warning(f"Model fitting error: {e}")

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_path}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(m_gap_vals, ood_errors, alpha=0.6, s=30)
    ax.set_xlabel(r'Mass gap $m_{\mathrm{gap}}$')
    ax.set_ylabel(r'OOD error $\epsilon_{\mathrm{OOD}}$')
    ax.set_title(f'Mass-robustness relationship\n(n={n_configs})')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sort_idx = np.argsort(m_valid)
    m_sorted = m_valid[sort_idx]

    if 'fits' in results:
        ax.scatter(m_valid, e_valid, alpha=0.4, s=20, label='Data')
        ax.plot(m_sorted, model_inverse_square(m_sorted, *popt1),
                'r-', linewidth=2,
                label=r'$\epsilon \propto m^{{-2}}$ (AIC={:.1f})'.format(aic1))
        ax.plot(m_sorted, model_inverse(m_sorted, *popt2),
                'g--', linewidth=2,
                label=r'$\epsilon \propto m^{{-1}}$ (AIC={:.1f})'.format(aic2))
        ax.plot(m_sorted, model_exponential(m_sorted, *popt3),
                'b:', linewidth=2,
                label=r'$\epsilon \propto e^{{-\kappa m}}$ (AIC={:.1f})'.format(aic3))
        ax.set_xlabel(r'Mass gap $m_{\mathrm{gap}}$')
        ax.set_ylabel(r'OOD error $\epsilon_{\mathrm{OOD}}$')
        ax.set_title('Model comparison')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    if 'fits' in results:
        residuals = e_valid - model_inverse_square(m_valid, *popt1)
        ax.hist(residuals, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel(r'Residuals ($\epsilon - \epsilon_{\mathrm{pred}}$)')
        ax.set_ylabel('Count')
        ax.set_title(r'Residual distribution ($\epsilon \propto m^{-2}$ model)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_p2_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    if 'fits' in results:
        print(f"\nModel Comparison (AIC):")
        print(f"  epsilon ~ m^(-2): {results['fits']['inverse_square']['AIC']:.1f} "
              f"(R2={results['fits']['inverse_square']['R2']:.3f})")
        print(f"  epsilon ~ m^(-1): {results['fits']['inverse']['AIC']:.1f} "
              f"(R2={results['fits']['inverse']['R2']:.3f})")
        print(f"  epsilon ~ exp(-kappa*m): {results['fits']['exponential']['AIC']:.1f} "
              f"(R2={results['fits']['exponential']['R2']:.3f})")

    if results['correlations']:
        r = results['correlations'][0]['pearson_r']
        p = results['correlations'][0]['pearson_p']
        print(f"\nCorrelation: rho = {r:.3f} (p = {p:.2e})")
        print(f"Cohen's d: {np.abs(r) / np.sqrt(1 - r**2):.2f}")

    return results


def experiment_p3_causal_propagation(n_realizations: int = 100, save_path: str = 'results_p3.json'):
    """
    Experiment P3: Emergent causal propagation.

    Measures: Finite propagation speed c_eff from hyperbolic regularization.
    Theoretical prediction: c_eff = sqrt(gamma/tau).
    """
    print("=" * 60)
    print("Experiment P3: Causal Propagation")
    print("=" * 60)

    tau_val = 1.0     # Large enough for underdamped ℓ=1 mode (4τλ₁=8>1)
    gamma_val = 1.0

    results = {
        'n_realizations': n_realizations,
        'propagation_speeds': [],
        'arrival_times': [],
        'tau': tau_val,
        'gamma': gamma_val,
    }

    c_theoretical = np.sqrt(gamma_val / tau_val)

    # Pure damped wave: alpha=0, beta=0 (no GL potential), tiny noise
    model = DataFieldTheory(N=2562, tau=tau_val, gamma=gamma_val,
                            alpha=0.0, beta=0.0, sigma=0.001, seed=42)
    model.initialize_field(k=1)

    vertices = model.vertices
    origin_idx = np.argmax(vertices[:, 0])
    # Geodesic distance on unit sphere: arccos(v · v_origin)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        dot_products = np.clip(vertices @ vertices[origin_idx], -1.0, 1.0)
    distances_from_origin = np.arccos(dot_products)

    for realization in tqdm(range(n_realizations)):
        model.phi = np.zeros((model.N, 1))
        model.phi_dot = np.zeros((model.N, 1))
        model.history = {'t': [], 'energy': []}
        model.t = 0

        # Localized Gaussian perturbation at origin vertex
        sigma_pert = 0.15
        perturbation = np.exp(
            -distances_from_origin**2 / (2 * sigma_pert**2)
        )
        model.phi[:, 0] = 1.0 * perturbation

        max_steps = 120
        threshold = 0.005
        eta = model.eta  # physical time per step call

        arrival_times = np.full(model.N, np.inf)

        for t_step in range(1, max_steps + 1):
            model.step()

            signal_strength = np.abs(model.phi[:, 0])
            new_arrivals = (
                (signal_strength > threshold) & (arrival_times == np.inf)
            )
            arrival_times[new_arrivals] = t_step * eta  # physical time

        # Exclude vertices near the source (within perturbation width)
        far_mask = distances_from_origin > 2 * sigma_pert
        valid_mask = (arrival_times < np.inf) & far_mask
        if np.sum(valid_mask) > 10:
            d_valid = distances_from_origin[valid_mask]
            t_valid = arrival_times[valid_mask]

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                d_valid, t_valid
            )

            c_eff = 1.0 / slope if slope > 0 else 0.0

            results['propagation_speeds'].append(float(c_eff))
            results['arrival_times'].append({
                'distances': d_valid.tolist(),
                'times': t_valid.tolist(),
                'r_squared': float(r_value**2)
            })

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_path}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    speeds = np.array(results['propagation_speeds'])
    ax.hist(speeds, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(speeds), color='red', linestyle='--',
               label=f'Mean: {np.mean(speeds):.3f} +/- {np.std(speeds):.3f}')
    ax.axvline(c_theoretical, color='green', linestyle=':',
               label=f'Theoretical: $\\sqrt{{\\Gamma/\\tau}}$ = {c_theoretical:.3f}')
    ax.set_xlabel(r'Propagation speed $c_{\mathrm{eff}}$')
    ax.set_ylabel('Count')
    ax.set_title(f'Propagation speed distribution\n(n={len(speeds)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if results['arrival_times']:
        example = results['arrival_times'][0]
        d = np.array(example['distances'])
        t = np.array(example['times'])

        ax.scatter(d, t, alpha=0.6, s=20)

        slope, intercept, r_value, _, _ = stats.linregress(d, t)
        d_fit = np.linspace(0, max(d), 100)
        t_fit = slope * d_fit + intercept

        c_label = f'{1.0 / slope:.3f}' if slope > 0 else 'inf'
        ax.plot(d_fit, t_fit, 'r-', linewidth=2,
                label=f'$c_{{\\mathrm{{eff}}}}$ = {c_label}, $R^2$ = {r_value**2:.3f}')

        ax.set_xlabel('Distance from source')
        ax.set_ylabel('Arrival time')
        ax.set_title('Signal propagation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    theta = np.linspace(0, 2 * np.pi, 100)
    c_mean = np.mean(speeds) if len(speeds) > 0 else c_theoretical
    for t_lc in [2, 5, 10]:
        r = c_mean * t_lc
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, '--', alpha=0.5, label=f't={t_lc}' if t_lc == 2 else None)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Theoretical light cones')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_p3_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    speeds = np.array(results['propagation_speeds'])
    print(f"\nPropagation speed: {np.mean(speeds):.3f} +/- {np.std(speeds):.3f}")
    print(f"Theoretical prediction: sqrt(Gamma/tau) = {c_theoretical:.3f}")
    print(f"Relative error: {np.abs(np.mean(speeds) - c_theoretical) / c_theoretical * 100:.1f}%")

    return results


def experiment_p4_equivariance(save_path: str = 'results_p4.json', **kwargs):
    """
    Experiment P4: Rotational equivariance.

    Measures: Ward identity residual via eigenvalue multiplet splitting.

    On S², the Laplacian eigenvalues are l(l+1) with multiplicity 2l+1.
    If the discrete Laplacian is rotationally equivariant, eigenvalues within
    each l-multiplet should be exactly degenerate.  The relative splitting
    (max-min)/mean within each multiplet quantifies the residual and should
    converge to zero with mesh refinement.
    """
    print("=" * 60)
    print("Experiment P4: Rotational Equivariance")
    print("=" * 60)

    results = {
        'ward_residuals': [],
        'mesh_convergence': {}
    }

    N_values = [162, 642, 2562, 10242]

    # Theoretical: eigenvalue = l(l+1), multiplicity = 2l+1
    ell_max = 4
    multiplet_info = [(ell, ell * (ell + 1), 2 * ell + 1)
                      for ell in range(ell_max + 1)]
    n_eig_needed = sum(2 * ell + 1 for ell in range(ell_max + 1))  # 25

    for N in N_values:
        print(f"\nTesting mesh resolution N={N}")

        model = DataFieldTheory(N=N, seed=42)
        model.initialize_field(k=1)

        # Solve generalised eigenvalue problem: L u = lambda M u
        M_diag = sp.diags(model.M_lumped)
        n_eig = min(n_eig_needed, N - 1)
        try:
            eigvals = spla.eigsh(model.L, k=n_eig, M=M_diag,
                                 which='SM', return_eigenvectors=False)
        except Exception:
            M_sqrt_inv = sp.diags(np.where(
                model.M_lumped > 1e-15,
                1.0 / np.sqrt(model.M_lumped), 0.0))
            L_sym = M_sqrt_inv @ model.L @ M_sqrt_inv
            eigvals = np.sort(
                np.linalg.eigvalsh(L_sym.toarray())
            )[:n_eig]
        eigvals = np.sort(eigvals)

        idx = 0
        multiplet_splits = []
        for ell, lam_theory, mult in multiplet_info:
            if idx + mult > len(eigvals):
                break
            group = eigvals[idx:idx + mult]
            spread = float(np.max(group) - np.min(group))
            mean_val = float(np.mean(group))
            rel_split = spread / mean_val if mean_val > 1e-10 else spread
            multiplet_splits.append({
                'ell': ell,
                'theoretical': lam_theory,
                'mean': mean_val,
                'spread': spread,
                'relative_split': float(rel_split)
            })
            if ell > 0:
                results['ward_residuals'].append(float(rel_split))
            idx += mult

        # Ward residual = mean relative splitting for l >= 1
        splits_l1_plus = [s['relative_split'] for s in multiplet_splits
                          if s['ell'] > 0]
        mean_residual = float(np.mean(splits_l1_plus))
        std_residual = float(np.std(splits_l1_plus))

        results['mesh_convergence'][str(N)] = {
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'characteristic_length': float(1 / np.sqrt(N)),
            'multiplet_splits': multiplet_splits
        }
        print(f"  Ward residual R = {mean_residual:.6f}")

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_path}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    residuals = np.array(results['ward_residuals'])
    ax.hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(residuals), color='red', linestyle='--',
               label=f'Mean: {np.mean(residuals):.6f}')
    ax.axvline(0.05, color='black', linestyle=':', label='Threshold: 0.05')
    ax.set_xlabel('Multiplet splitting (relative)')
    ax.set_ylabel('Count')
    ax.set_title(f'Eigenvalue splitting distribution\n(n={len(residuals)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    Ns = sorted(results['mesh_convergence'].keys(), key=int)
    residuals_mean = [results['mesh_convergence'][N_key]['mean_residual']
                      for N_key in Ns]
    residuals_std = [results['mesh_convergence'][N_key]['std_residual']
                     for N_key in Ns]
    h_vals = [1 / np.sqrt(int(N_key)) for N_key in Ns]

    ax.errorbar(h_vals, residuals_mean, yerr=residuals_std, fmt='o-', capsize=5)
    ax.set_xlabel(r'Characteristic mesh size $h \propto 1/\sqrt{N}$')
    ax.set_ylabel('Ward residual R')
    ax.set_title('Mesh refinement convergence')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if len(h_vals) > 2:
        log_h = np.log(h_vals)
        log_R = np.log(residuals_mean)

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_h, log_R)

        ax.loglog(h_vals, residuals_mean, 'o-', label='Data')

        h_fit = np.logspace(np.log10(min(h_vals)),
                            np.log10(max(h_vals)), 100)
        R_fit = np.exp(intercept) * h_fit**slope
        ax.loglog(h_fit, R_fit, 'r--',
                  label=f'$R \\propto h^{{{slope:.2f}}}$\n'
                        f'$R^2 = {r_value**2:.3f}$')

        ax.set_xlabel('Mesh size h')
        ax.set_ylabel('Residual R')
        ax.set_title('Convergence rate analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_p4_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    residuals = np.array(results['ward_residuals'])
    print(f"\nWard identity residual: {np.mean(residuals):.6f} +/- "
          f"{np.std(residuals):.6f}")
    print(f"Below threshold (0.05): "
          f"{(residuals < 0.05).sum() / len(residuals) * 100:.1f}%")

    if len(h_vals) > 2:
        print(f"Convergence rate: R ~ h^{slope:.2f}")
        print(f"R^2 = {r_value**2:.3f}")

    return results


def run_all_experiments():
    """Run all four experiments."""
    print("=" * 70)
    print("COMPREHENSIVE EXPERIMENTAL VALIDATION OF DATA FIELD THEORY")
    print("=" * 70)

    import os
    os.makedirs('results', exist_ok=True)

    print("\n1. Running Experiment P1: Critical Phenomena...")
    results_p1 = experiment_p1_critical_phenomena(
        n_seeds=10, save_path='results/experiment_p1.json')

    print("\n2. Running Experiment P2: Mass-Robustness Law...")
    results_p2 = experiment_p2_mass_robustness(
        n_configs=30, save_path='results/experiment_p2.json')

    print("\n3. Running Experiment P3: Causal Propagation...")
    results_p3 = experiment_p3_causal_propagation(
        n_realizations=50, save_path='results/experiment_p3.json')

    print("\n4. Running Experiment P4: Equivariance...")
    results_p4 = experiment_p4_equivariance(
        save_path='results/experiment_p4.json')

    print("\n" + "=" * 70)
    print("EXPERIMENTAL SUMMARY")
    print("=" * 70)

    if results_p1 and 'critical_exponents' in results_p1:
        nu_vals = [exp['nu'] for exp in results_p1['critical_exponents']]
        if nu_vals:
            print(f"P1 - Critical exponent nu: {np.mean(nu_vals):.3f} +/- {np.std(nu_vals):.3f}")

    if results_p2 and 'correlations' in results_p2 and results_p2['correlations']:
        r = results_p2['correlations'][0]['pearson_r']
        p = results_p2['correlations'][0]['pearson_p']
        print(f"P2 - Mass-robustness correlation: rho = {r:.3f} (p = {p:.2e})")

    if results_p3 and 'propagation_speeds' in results_p3:
        speeds = np.array(results_p3['propagation_speeds'])
        if len(speeds) > 0:
            print(f"P3 - Propagation speed: c_eff = {np.mean(speeds):.3f} +/- {np.std(speeds):.3f}")

    if results_p4 and 'ward_residuals' in results_p4:
        residuals = np.array(results_p4['ward_residuals'])
        if len(residuals) > 0:
            print(f"P4 - Ward identity residual: R = {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}")
            print(f"    Below threshold (0.05): {(residuals < 0.05).sum() / len(residuals) * 100:.1f}%")

    print("\nAll experiments completed!")
    print("Results saved to 'results/' directory")
    print("Figures saved as PNG files")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
