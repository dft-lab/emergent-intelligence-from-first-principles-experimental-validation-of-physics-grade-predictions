"""
Experimental Validation of Data Field Theory Predictions
Author: Mohammadreza Nehzati
Repository: https://github.com/dft-labs/emergent-intelligence-from-first-principles-experimental-validation-of-physics-grade-predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize
import seaborn as sns
from dft_core import DataFieldTheory, generate_hierarchical_data
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
    1. Correlation length divergence ξ(t) ~ |t-t_c|^{-ν}
    2. 1/f noise spectrum S(f) ~ f^{-α}
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
    
    # Different system sizes for finite-size scaling
    N_values = [642, 2562, 10242]
    
    for N in N_values:
        print(f"\nRunning finite-size scaling for N={N}")
        xi_peaks = []
        t_c_values = []
        
        for seed in tqdm(range(n_seeds)):
            # Initialize model
            model = DataFieldTheory(N=N, seed=seed)
            model.initialize_field(k=8)
            
            # Generate data
            X, y, y_onehot, _ = generate_hierarchical_data(
                N_samples=500, n_categories=8, manifold='sphere', seed=seed
            )
            
            # Train and track correlation length
            n_steps = 200
            xi_history = []
            
            for step in range(n_steps):
                batch_size = 32
                idx = np.random.choice(len(X), batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y_onehot[idx]
                
                model.step(X_batch, y_batch)
                
                if step % 5 == 0:
                    xi = model.correlation_length()
                    xi_history.append((step, xi))
            
            # Find critical time (peak correlation length)
            steps, xi_vals = zip(*xi_history)
            t_c_idx = np.argmax(xi_vals)
            t_c = steps[t_c_idx]
            xi_peak = xi_vals[t_c_idx]
            
            xi_peaks.append(xi_peak)
            t_c_values.append(t_c)
            
            # Fit power-law divergence around t_c
            if t_c_idx > 5 and t_c_idx < len(steps) - 5:
                # Use ±10 steps around critical point
                window = 10
                start = max(0, t_c_idx - window)
                end = min(len(steps), t_c_idx + window)
                
                t_window = np.array(steps[start:end])
                xi_window = np.array(xi_vals[start:end])
                
                # Fit ξ ~ |t-t_c|^{-ν}
                def power_law(t, A, nu):
                    return A * np.abs(t - t_c) ** (-nu)
                
                try:
                    popt, _ = optimize.curve_fit(
                        power_law, t_window, xi_window,
                        p0=[xi_peak, 0.63], bounds=(0, [10, 2])
                    )
                    results['critical_exponents'].append({
                        'N': N, 'seed': seed, 'nu': float(popt[1]),
                        't_c': float(t_c), 'xi_peak': float(xi_peak)
                    })
                except:
                    pass
            
            # Compute PSD for 1/f analysis
            if seed < 5:  # Only for first few seeds to save time
                # Detrend
                xi_detrended = signal.detrend(xi_vals)
                
                # Compute PSD using Welch's method
                f, Pxx = signal.welch(
                    xi_detrended, fs=0.2, nperseg=min(128, len(xi_detrended)),
                    window='hann'
                )
                
                # Fit power law in log-log space
                mask = (f > 0.01) & (f < 0.5)  # Intermediate frequencies
                if np.sum(mask) > 3:
                    coeffs = np.polyfit(np.log(f[mask]), np.log(Pxx[mask]), 1)
                    alpha = -coeffs[0]  # S(f) ~ f^{-α}
                    
                    results['psd_exponents'].append({
                        'N': N, 'seed': seed, 'alpha': float(alpha)
                    })
        
        # Store finite-size scaling results
        results['finite_size_scaling'][N] = {
            'mean_xi_peak': float(np.mean(xi_peaks)),
            'std_xi_peak': float(np.std(xi_peaks)),
            'mean_t_c': float(np.mean(t_c_values)),
            'std_t_c': float(np.std(t_c_values))
        }
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    
    # Generate summary statistics
    nu_vals = [exp['nu'] for exp in results['critical_exponents']]
    alpha_vals = [exp['alpha'] for exp in results['psd_exponents']]
    
    print(f"\nSummary Statistics:")
    print(f"Critical exponent ν: {np.mean(nu_vals):.3f} ± {np.std(nu_vals):.3f}")
    print(f"PSD exponent α: {np.mean(alpha_vals):.3f} ± {np.std(alpha_vals):.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Critical exponent distribution
    ax = axes[0]
    ax.hist(nu_vals, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(nu_vals), color='red', linestyle='--', label=f'Mean: {np.mean(nu_vals):.3f}')
    ax.set_xlabel('Critical exponent ν')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of ν (n={})'.format(len(nu_vals)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: PSD exponent distribution
    ax = axes[1]
    ax.hist(alpha_vals, bins=15, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(1.0, color='black', linestyle='--', label='1/f reference')
    ax.axvline(np.mean(alpha_vals), color='red', linestyle='-', label=f'Mean: {np.mean(alpha_vals):.3f}')
    ax.set_xlabel('PSD exponent α')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of α (n={})'.format(len(alpha_vals)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel C: Finite-size scaling
    ax = axes[2]
    Ns = list(results['finite_size_scaling'].keys())
    xi_means = [results['finite_size_scaling'][N]['mean_xi_peak'] for N in Ns]
    xi_stds = [results['finite_size_scaling'][N]['std_xi_peak'] for N in Ns]
    
    ax.errorbar(Ns, xi_means, yerr=xi_stds, fmt='o-', capsize=5, linewidth=2)
    ax.set_xlabel('System size N')
    ax.set_ylabel('Peak correlation length ξ_peak')
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
    
    Tests: ε_OOD ∝ m_gap^{-2}
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
    
    # Vary parameters to get different mass gaps
    alphas = np.linspace(0.5, 2.0, n_configs)
    
    for i, alpha in enumerate(tqdm(alphas)):
        # Initialize model with different alpha
        model = DataFieldTheory(N=2562, alpha=alpha, seed=i)
        model.initialize_field(k=8)
        
        # Generate training and OOD test data
        # In-distribution: balanced 8 categories
        X_train, y_train, y_train_onehot, centers = generate_hierarchical_data(
            N_samples=500, n_categories=8, manifold='sphere', seed=i
        )
        
        # Out-of-distribution: novel category combinations
        # Create shifted distribution by rotating prototypes
        from scipy.spatial.transform import Rotation
        rot = Rotation.random(random_state=i).as_matrix()
        centers_ood = centers @ rot.T
        
        # Generate OOD samples
        X_ood = []
        y_ood = []
        kappa = 8.0
        
        for c in range(8):
            center = centers_ood[c]
            for _ in range(200):  # Fewer OOD samples
                u = np.random.normal(0, 1, 3)
                u = u / np.linalg.norm(u)
                direction = center + 0.1 * np.random.normal(0, 1, 3)
                point = u + kappa * direction
                point = point / np.linalg.norm(point)
                
                X_ood.append(point)
                y_ood.append(c)
        
        X_ood = np.array(X_ood)
        y_ood = np.array(y_ood)
        
        # Train model
        n_steps = 150
        for step in range(n_steps):
            batch_size = 32
            idx = np.random.choice(len(X_train), batch_size, replace=False)
            X_batch = X_train[idx]
            y_batch = y_train_onehot[idx]
            
            model.step(X_batch, y_batch)
        
        # Compute mass gap
        m_gap = model.spectral_gap()
        
        # Compute OOD error
        y_pred = model.predict(X_ood)
        ood_error = 1 - np.mean(y_pred == y_ood)
        
        results['mass_gap_vals'].append(float(m_gap))
        results['ood_errors'].append(float(ood_error))
    
    # Fit models
    m_gap_vals = np.array(results['mass_gap_vals'])
    ood_errors = np.array(results['ood_errors'])
    
    # Model 1: ε ∝ m^{-2}
    def model_inverse_square(m, A):
        return A / m**2
    
    # Model 2: ε ∝ m^{-1}
    def model_inverse(m, A):
        return A / m
    
    # Model 3: ε ∝ exp(-κm)
    def model_exponential(m, A, kappa):
        return A * np.exp(-kappa * m)
    
    try:
        # Fit models
        popt1, _ = optimize.curve_fit(model_inverse_square, m_gap_vals, ood_errors, p0=[0.5])
        popt2, _ = optimize.curve_fit(model_inverse, m_gap_vals, ood_errors, p0=[0.5])
        popt3, _ = optimize.curve_fit(model_exponential, m_gap_vals, ood_errors, p0=[1.0, 1.0])
        
        # Compute AIC
        n = len(ood_errors)
        
        # Residuals
        res1 = ood_errors - model_inverse_square(m_gap_vals, *popt1)
        res2 = ood_errors - model_inverse(m_gap_vals, *popt2)
        res3 = ood_errors - model_exponential(m_gap_vals, *popt3)
        
        # AIC = n*log(RSS/n) + 2k
        k1 = 1  # one parameter
        k2 = 1
        k3 = 2  # two parameters
        
        aic1 = n * np.log(np.sum(res1**2)/n) + 2*k1
        aic2 = n * np.log(np.sum(res2**2)/n) + 2*k2
        aic3 = n * np.log(np.sum(res3**2)/n) + 2*k3
        
        # Store results
        results['fits'] = {
            'inverse_square': {
                'params': popt1.tolist(),
                'AIC': float(aic1),
                'R2': float(1 - np.sum(res1**2)/np.sum((ood_errors - np.mean(ood_errors))**2))
            },
            'inverse': {
                'params': popt2.tolist(),
                'AIC': float(aic2),
                'R2': float(1 - np.sum(res2**2)/np.sum((ood_errors - np.mean(ood_errors))**2))
            },
            'exponential': {
                'params': popt3.tolist(),
                'AIC': float(aic3),
                'R2': float(1 - np.sum(res3**2)/np.sum((ood_errors - np.mean(ood_errors))**2))
            }
        }
        
        # Compute correlation
        r, p = stats.pearsonr(m_gap_vals, ood_errors)
        results['correlations'].append({
            'pearson_r': float(r),
            'pearson_p': float(p)
        })
        
    except Exception as e:
        print(f"Fitting error: {e}")
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Raw data
    ax = axes[0]
    ax.scatter(m_gap_vals, ood_errors, alpha=0.6, s=30)
    ax.set_xlabel('Mass gap m_gap')
    ax.set_ylabel('OOD error ε_OOD')
    ax.set_title('Mass-robustness relationship\n(n={})'.format(n_configs))
    ax.grid(True, alpha=0.3)
    
    # Panel B: Model fits
    ax = axes[1]
    # Sort for smooth curves
    sort_idx = np.argsort(m_gap_vals)
    m_sorted = m_gap_vals[sort_idx]
    
    if 'fits' in results:
        ax.scatter(m_gap_vals, ood_errors, alpha=0.4, s=20, label='Data')
        
        # Plot fits
        ax.plot(m_sorted, model_inverse_square(m_sorted, *popt1), 
                'r-', linewidth=2, label='ε ∝ m$^{-2}$ (AIC={:.1f})'.format(aic1))
        ax.plot(m_sorted, model_inverse(m_sorted, *popt2),
                'g--', linewidth=2, label='ε ∝ m$^{-1}$ (AIC={:.1f})'.format(aic2))
        ax.plot(m_sorted, model_exponential(m_sorted, *popt3),
                'b:', linewidth=2, label='ε ∝ e$^{-κm}$ (AIC={:.1f})'.format(aic3))
        
        ax.set_xlabel('Mass gap m_gap')
        ax.set_ylabel('OOD error ε_OOD')
        ax.set_title('Model comparison')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Panel C: Residuals
    ax = axes[2]
    if 'fits' in results:
        residuals = ood_errors - model_inverse_square(m_gap_vals, *popt1)
        ax.hist(residuals, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel('Residuals (ε - ε_pred)')
        ax.set_ylabel('Count')
        ax.set_title('Residual distribution\n(ε ∝ m$^{-2}$ model)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_p2_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    if 'fits' in results:
        print(f"\nModel Comparison (AIC):")
        print(f"  ε ∝ m^(-2): {results['fits']['inverse_square']['AIC']:.1f} (R²={results['fits']['inverse_square']['R2']:.3f})")
        print(f"  ε ∝ m^(-1): {results['fits']['inverse']['AIC']:.1f} (R²={results['fits']['inverse']['R2']:.3f})")
        print(f"  ε ∝ exp(-κm): {results['fits']['exponential']['AIC']:.1f} (R²={results['fits']['exponential']['R2']:.3f})")
        
    if results['correlations']:
        r = results['correlations'][0]['pearson_r']
        p = results['correlations'][0]['pearson_p']
        print(f"\nCorrelation: ρ = {r:.3f} (p = {p:.2e})")
        print(f"Cohen's d: {np.abs(r)/np.sqrt(1-r**2):.2f}")
    
    return results


def experiment_p3_causal_propagation(n_realizations: int = 100, save_path: str = 'results_p3.json'):
    """
    Experiment P3: Emergent causal propagation.
    
    Measures: Finite propagation speed c_eff from hyperbolic regularization.
    """
    print("=" * 60)
    print("Experiment P3: Causal Propagation")
    print("=" * 60)
    
    results = {
        'n_realizations': n_realizations,
        'propagation_speeds': [],
        'arrival_times': []
    }
    
    # Initialize model with hyperbolic regularization
    model = DataFieldTheory(N=2562, tau=0.05, gamma=1.0, seed=42)
    model.initialize_field(k=1)  # Scalar field for simplicity
    
    # Create initial perturbation at origin
    # Find vertex closest to (1,0,0) on sphere
    vertices = model.vertices
    origin_idx = np.argmax(vertices[:, 0])  # Max x-coordinate
    
    for realization in tqdm(range(n_realizations)):
        # Reset field
        model.phi = np.zeros((model.N, 1))
        model.phi_dot = np.zeros((model.N, 1))
        
        # Add Gaussian perturbation at origin
        sigma_pert = 0.1
        distances = np.linalg.norm(vertices - vertices[origin_idx], axis=1)
        perturbation = np.exp(-distances**2 / (2*sigma_pert**2))
        model.phi[:, 0] = 0.1 * perturbation
        
        # Track propagation
        max_time = 50
        threshold = 0.03  # Detection threshold
        
        arrival_times = np.full(model.N, np.inf)
        
        for t in range(max_time):
            # Evolve without source term
            model.step()
            
            # Check which vertices have exceeded threshold
            signal_strength = np.abs(model.phi[:, 0])
            new_arrivals = (signal_strength > threshold) & (arrival_times == np.inf)
            arrival_times[new_arrivals] = t
        
        # Convert to distances and arrival times
        distances = np.linalg.norm(vertices - vertices[origin_idx], axis=1)
        
        # Only consider vertices that actually received signal
        valid_mask = arrival_times < np.inf
        if np.sum(valid_mask) > 10:
            d_valid = distances[valid_mask]
            t_valid = arrival_times[valid_mask]
            
            # Fit linear: t = d/c
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                d_valid, t_valid
            )
            
            c_eff = 1/slope if slope > 0 else 0
            
            results['propagation_speeds'].append(float(c_eff))
            results['arrival_times'].append({
                'distances': d_valid.tolist(),
                'times': t_valid.tolist(),
                'r_squared': float(r_value**2)
            })
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Propagation speed distribution
    ax = axes[0]
    speeds = np.array(results['propagation_speeds'])
    ax.hist(speeds, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(speeds), color='red', linestyle='--', 
              label=f'Mean: {np.mean(speeds):.3f} ± {np.std(speeds):.3f}')
    ax.axvline(np.sqrt(1/0.05), color='green', linestyle=':', 
              label='Theoretical: √(Γ/τ) = 0.447')
    ax.set_xlabel('Propagation speed c_eff')
    ax.set_ylabel('Count')
    ax.set_title('Propagation speed distribution\n(n={})'.format(len(speeds)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Example distance-time relationship
    ax = axes[1]
    if results['arrival_times']:
        # Use first realization as example
        example = results['arrival_times'][0]
        d = np.array(example['distances'])
        t = np.array(example['times'])
        
        ax.scatter(d, t, alpha=0.6, s=20)
        
        # Fit line
        slope, intercept, r_value, _, _ = stats.linregress(d, t)
        d_fit = np.linspace(0, max(d), 100)
        t_fit = slope * d_fit + intercept
        
        ax.plot(d_fit, t_fit, 'r-', linewidth=2,
               label=f'c_eff = {1/slope:.3f}, R² = {r_value**2:.3f}')
        
        ax.set_xlabel('Distance from source')
        ax.set_ylabel('Arrival time')
        ax.set_title('Signal propagation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Panel C: Wavefront visualization
    ax = axes[2]
    # Simple visualization of expanding wave
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Theoretical light cone
    c_mean = np.mean(speeds)
    for t in [10, 20, 30]:
        r = c_mean * t
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, '--', alpha=0.5, label=f't={t}' if t==10 else None)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Theoretical light cones')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_p3_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    speeds = np.array(results['propagation_speeds'])
    print(f"\nPropagation speed: {np.mean(speeds):.3f} ± {np.std(speeds):.3f}")
    print(f"Theoretical prediction: √(Γ/τ) = {np.sqrt(1/0.05):.3f}")
    print(f"Relative error: {np.abs(np.mean(speeds) - np.sqrt(1/0.05))/np.sqrt(1/0.05)*100:.1f}%")
    
    return results


def experiment_p4_equivariance(n_configs: int = 50, save_path: str = 'results_p4.json'):
    """
    Experiment P4: Rotational equivariance.
    
    Measures: Ward identity residual R and convergence with mesh refinement.
    """
    print("=" * 60)
    print("Experiment P4: Rotational Equivariance")
    print("=" * 60)
    
    results = {
        'n_configs': n_configs,
        'ward_residuals': [],
        'mesh_convergence': {}
    }
    
    # Test different mesh resolutions
    N_values = [162, 642, 2562, 10242]
    
    for N in N_values:
        print(f"\nTesting mesh resolution N={N}")
        residuals = []
        
        for config in tqdm(range(min(n_configs, 20))):  # Fewer for larger N
            # Initialize model
            model = DataFieldTheory(N=N, seed=config)
            model.initialize_field(k=3)
            
            # Generate random rotation
            from scipy.spatial.transform import Rotation
            R = Rotation.random(random_state=config).as_matrix()
            
            # Apply rotation to vertices
            vertices_rotated = model.vertices @ R.T
            
            # Need to recompute Laplacian for rotated vertices
            # For icosahedral mesh, rotation preserves connectivity
            # So we can use the same Laplacian but need to interpolate field
            
            # Create field with some structure
            phi_original = np.random.normal(0, 1, (N, 3))
            
            # Interpolate rotated field (simplified)
            # In practice, need proper interpolation on sphere
            # Here we use nearest neighbor for demonstration
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1).fit(vertices_rotated)
            _, indices = nn.kneighbors(model.vertices)
            
            phi_rotated = phi_original[indices.flatten()]
            
            # Apply Laplacian to both
            if sp.issparse(model.L):
                L_phi_original = model.L @ phi_original
                L_phi_rotated = model.L @ phi_rotated
            else:
                L_phi_original = model.L.dot(phi_original)
                L_phi_rotated = model.L.dot(phi_rotated)
            
            # Interpolate L(phi_rotated) back to original coordinates
            _, indices_back = nn.kneighbors(vertices_rotated)
            L_phi_rotated_back = L_phi_rotated[indices_back.flatten()]
            
            # Compute Ward residual
            numerator = np.linalg.norm(L_phi_rotated_back - L_phi_original)
            denominator = np.linalg.norm(L_phi_original)
            
            if denominator > 1e-10:
                residual = numerator / denominator
                residuals.append(float(residual))
        
        if residuals:
            results['mesh_convergence'][N] = {
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'characteristic_length': float(1/np.sqrt(N))  # Approximate mesh size
            }
            results['ward_residuals'].extend(residuals)
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Residual distribution
    ax = axes[0]
    residuals = np.array(results['ward_residuals'])
    ax.hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(residuals), color='red', linestyle='--',
              label=f'Mean: {np.mean(residuals):.4f} ± {np.std(residuals):.4f}')
    ax.axvline(0.05, color='black', linestyle=':', label='Threshold: 0.05')
    ax.set_xlabel('Ward identity residual R')
    ax.set_ylabel('Count')
    ax.set_title('Residual distribution\n(n={})'.format(len(residuals)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Mesh convergence
    ax = axes[1]
    Ns = sorted(results['mesh_convergence'].keys())
    residuals_mean = [results['mesh_convergence'][N]['mean_residual'] for N in Ns]
    residuals_std = [results['mesh_convergence'][N]['std_residual'] for N in Ns]
    h_vals = [1/np.sqrt(N) for N in Ns]  # Characteristic mesh size
    
    ax.errorbar(h_vals, residuals_mean, yerr=residuals_std, fmt='o-', capsize=5)
    ax.set_xlabel('Characteristic mesh size h ∝ 1/√N')
    ax.set_ylabel('Ward residual R')
    ax.set_title('Mesh refinement convergence')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Panel C: Fit convergence rate
    ax = axes[2]
    if len(h_vals) > 2:
        # Fit power law: R ∝ h^p
        log_h = np.log(h_vals)
        log_R = np.log(residuals_mean)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_h, log_R)
        
        ax.loglog(h_vals, residuals_mean, 'o-', label='Data')
        
        # Plot fit
        h_fit = np.logspace(np.log10(min(h_vals)), np.log10(max(h_vals)), 100)
        R_fit = np.exp(intercept) * h_fit**slope
        ax.loglog(h_fit, R_fit, 'r--', 
                 label=f'R ∝ h$^{{{slope:.2f}}}$\nR² = {r_value**2:.3f}')
        
        ax.set_xlabel('Mesh size h')
        ax.set_ylabel('Residual R')
        ax.set_title('Convergence rate analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_p4_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    residuals = np.array(results['ward_residuals'])
    print(f"\nWard identity residual: {np.mean(residuals):.4f} ± {np.std(residuals):.4f}")
    print(f"Below threshold (0.05): {(residuals < 0.05).sum()/len(residuals)*100:.1f}%")
    
    if len(h_vals) > 2:
        print(f"Convergence rate: R ∝ h^{slope:.2f}")
        print(f"R² = {r_value**2:.3f}")
    
    return results


def run_all_experiments():
    """Run all four experiments."""
    print("=" * 70)
    print("COMPREHENSIVE EXPERIMENTAL VALIDATION OF DATA FIELD THEORY")
    print("=" * 70)
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run experiments (with reduced size for demonstration)
    print("\n1. Running Experiment P1: Critical Phenomena...")
    results_p1 = experiment_p1_critical_phenomena(n_seeds=10, 
                                                 save_path='results/experiment_p1.json')
    
    print("\n2. Running Experiment P2: Mass-Robustness Law...")
    results_p2 = experiment_p2_mass_robustness(n_configs=30,
                                              save_path='results/experiment_p2.json')
    
    print("\n3. Running Experiment P3: Causal Propagation...")
    results_p3 = experiment_p3_causal_propagation(n_realizations=50,
                                                 save_path='results/experiment_p3.json')
    
    print("\n4. Running Experiment P4: Equivariance...")
    results_p4 = experiment_p4_equivariance(n_configs=20,
                                           save_path='results/experiment_p4.json')
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("EXPERIMENTAL SUMMARY")
    print("=" * 70)
    
    if results_p1 and 'critical_exponents' in results_p1:
        nu_vals = [exp['nu'] for exp in results_p1['critical_exponents']]
        print(f"P1 - Critical exponent ν: {np.mean(nu_vals):.3f} ± {np.std(nu_vals):.3f}")
    
    if results_p2 and 'correlations' in results_p2:
        r = results_p2['correlations'][0]['pearson_r']
        p = results_p2['correlations'][0]['pearson_p']
        print(f"P2 - Mass-robustness correlation: ρ = {r:.3f} (p = {p:.2e})")
    
    if results_p3 and 'propagation_speeds' in results_p3:
        speeds = np.array(results_p3['propagation_speeds'])
        print(f"P3 - Propagation speed: c_eff = {np.mean(speeds):.3f} ± {np.std(speeds):.3f}")
    
    if results_p4 and 'ward_residuals' in results_p4:
        residuals = np.array(results_p4['ward_residuals'])
        print(f"P4 - Ward identity residual: R = {np.mean(residuals):.4f} ± {np.std(residuals):.4f}")
        print(f"    Below threshold (0.05): {(residuals < 0.05).sum()/len(residuals)*100:.1f}%")
    
    print("\nAll experiments completed!")
    print("Results saved to 'results/' directory")
    print("Figures saved as PNG files")
    print("=" * 70)


if __name__ == "__main__":
    # Run demonstration
    run_all_experiments()
