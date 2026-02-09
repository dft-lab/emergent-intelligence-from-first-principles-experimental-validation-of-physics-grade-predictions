"""
Data Field Theory Core Implementation
Author: Mohammadreza Nehzati
Repository: https://github.com/dft-labs/emergent-intelligence-from-first-principles-experimental-validation-of-physics-grade-predictions
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import signal
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
import trimesh
import matplotlib.pyplot as plt
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Icosphere subdivision levels: level -> vertex count
# 0->12, 1->42, 2->162, 3->642, 4->2562, 5->10242, 6->40962
_ICOSPHERE_LEVELS = {12: 0, 42: 1, 162: 2, 642: 3, 2562: 4, 10242: 5, 40962: 6}


class DataFieldTheory:
    """Implementation of Data Field Theory on Riemannian manifolds."""

    def __init__(self,
                 manifold_type: str = 'sphere',
                 N: int = 2562,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 eta: float = 0.1,
                 sigma: float = 0.01,
                 tau: float = 0.05,
                 gamma: float = 1.0,
                 sigma_k: float = 0.1,
                 seed: int = 42):
        """
        Initialize DFT model.

        Parameters
        ----------
        manifold_type : str
            Type of manifold ('sphere' or 'hyperbolic')
        N : int
            Number of vertices in discretization. For 'sphere', must be one of
            {12, 42, 162, 642, 2562, 10242, 40962}.
        alpha, beta : float
            Ginzburg-Landau potential parameters
        eta : float
            Learning rate / time step
        sigma : float
            Noise amplitude
        tau : float
            Hyperbolic regularization parameter
        gamma : float
            Gradient coupling (spatial stiffness) coefficient
        sigma_k : float
            Kernel bandwidth for data source term
        seed : int
            Random seed for reproducibility
        """
        self.manifold_type = manifold_type
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        self.sigma_k = sigma_k
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Initialize manifold geometry
        self.vertices, self.faces = self._create_manifold(manifold_type, N)
        self.N = len(self.vertices)  # Update N to actual vertex count
        self.L = self._compute_laplacian()

        # Field state
        self.phi = None
        self.phi_dot = None
        self.t = 0

        # Statistics tracking (phi snapshots stored every record_interval steps)
        self.history = {
            't': [],
            'energy': [],
            'correlation_length': [],
            'accuracy': []
        }

    def _create_manifold(self, manifold_type: str, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create manifold discretization."""
        if manifold_type == 'sphere':
            return self._create_icosphere(N)
        elif manifold_type == 'hyperbolic':
            return self._create_hyperbolic_mesh(N)
        else:
            raise ValueError(f"Unknown manifold type: {manifold_type}")

    def _create_icosphere(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create icosahedral discretization of sphere.

        Parameters
        ----------
        N : int
            Requested number of vertices. Snapped to nearest valid icosphere
            subdivision level.

        Returns
        -------
        vertices : np.ndarray, shape (N_actual, 3)
        faces : np.ndarray, shape (F, 3)
        """
        # Find closest valid subdivision level
        valid_counts = sorted(_ICOSPHERE_LEVELS.keys())
        closest_N = min(valid_counts, key=lambda v: abs(v - N))
        subdivisions = _ICOSPHERE_LEVELS[closest_N]

        if closest_N != N:
            logger.warning(
                f"Requested N={N} is not a valid icosphere vertex count. "
                f"Using N={closest_N} (subdivisions={subdivisions})."
            )

        mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int64)

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        return vertices, faces

    def _create_hyperbolic_mesh(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create mesh in Poincare disk."""
        r = np.sqrt(self.rng.uniform(0, 0.95**2, N))
        theta = self.rng.uniform(0, 2 * np.pi, N)
        vertices = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

        tri = Delaunay(vertices)
        faces = tri.simplices

        return vertices, faces

    def _compute_laplacian(self) -> sp.csr_matrix:
        """Compute Laplacian for mesh."""
        if self.manifold_type == 'sphere':
            return self._compute_cotangent_laplacian()
        else:
            return self._compute_graph_laplacian()

    def _compute_cotangent_laplacian(self) -> sp.csr_matrix:
        """Compute cotangent (stiffness) Laplacian and lumped mass matrix.

        The stiffness matrix ``L`` and lumped mass ``M`` satisfy the
        generalized eigenvalue relation ``L u = lambda M u``, where the
        eigenvalues ``lambda`` converge to the Laplace-Beltrami spectrum
        as *N* grows.  The dynamics use ``M^{-1} L`` so that the effective
        Laplacian coupling is resolution-independent.
        """
        n_vertices = len(self.vertices)
        adj = sp.lil_matrix((n_vertices, n_vertices))
        vertex_area = np.zeros(n_vertices)

        n_degenerate = 0
        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]

            # Face area — used for the lumped mass matrix
            face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            for idx in face:
                vertex_area[idx] += face_area / 3.0

            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3

                vi = self.vertices[face[i]]
                vj = self.vertices[face[j]]
                vk = self.vertices[face[k]]

                e1 = vi - vk
                e2 = vj - vk
                cross_norm = np.linalg.norm(np.cross(e1, e2))

                if cross_norm < 1e-12:
                    n_degenerate += 1
                    continue

                cot_angle = 0.5 * np.dot(e1, e2) / cross_norm

                adj[face[i], face[j]] += cot_angle
                adj[face[j], face[i]] += cot_angle

        if n_degenerate > 0:
            logger.warning(
                f"Skipped {n_degenerate} degenerate triangle half-edges "
                f"in cotangent Laplacian computation."
            )

        # Lumped mass matrix (diagonal) — needed by step() and spectral_gap()
        self.M_lumped = vertex_area
        self.M_inv = np.where(vertex_area > 1e-15, 1.0 / vertex_area, 0.0)

        degree = np.array(adj.sum(axis=1)).flatten()
        L = sp.diags(degree) - adj
        L = sp.csr_matrix(L)

        # Gershgorin estimate of max eigenvalue of M^{-1} L
        self.lambda_max = float(np.max(degree * self.M_inv))

        return L

    def _compute_graph_laplacian(self) -> sp.csr_matrix:
        """Compute graph Laplacian for hyperbolic mesh."""
        adj = kneighbors_graph(self.vertices, n_neighbors=6, mode='connectivity')

        # Symmetrize
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

        degree = np.array(adj.sum(axis=1)).flatten()
        L = sp.diags(degree) - adj

        # Uniform lumped mass (no natural area element for knn graph)
        self.M_lumped = np.ones(self.N)
        self.M_inv = np.ones(self.N)
        self.lambda_max = float(np.max(degree))

        return sp.csr_matrix(L)

    def initialize_field(self, k: int = 8):
        """Initialize field with random values."""
        self.phi = self.rng.normal(0, 0.1, (self.N, k))
        self.phi_dot = np.zeros_like(self.phi)

    def free_energy(self, phi: np.ndarray = None) -> float:
        """Compute Ginzburg-Landau free energy.

        F[phi] = (1/2) phi^T L phi  +  sum_i [ -(alpha/2)|phi_i|^2 + (beta/4)|phi_i|^4 ]

        The force is -dF/dphi:
          force_i = -L phi_i + alpha phi_i - beta |phi_i|^2 phi_i
        """
        if phi is None:
            phi = self.phi

        # Gradient term: (1/2) sum phi^T L phi
        grad_term = np.sum(phi * (self.L @ phi))

        # Potential term (area-weighted for correct continuous limit)
        phi_norm_sq = np.sum(phi**2, axis=1)
        potential = -0.5 * self.alpha * phi_norm_sq + 0.25 * self.beta * phi_norm_sq**2

        return 0.5 * grad_term + np.sum(self.M_lumped * potential)

    def source_term(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute external source term from data.

        Parameters
        ----------
        X : np.ndarray, shape (batch_size, 3) for sphere or (batch_size, 2) for hyperbolic
            Data points on manifold
        y : np.ndarray, shape (batch_size, k)
            One-hot encoded labels

        Returns
        -------
        J : np.ndarray, shape (N, k)
            Source term at each vertex
        """
        batch_size = len(X)

        # Suppress spurious BLAS-level matmul warnings (data is verified
        # finite; warnings arise from denormalized intermediates in BLAS).
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            # Raw Gaussian kernel (no per-row softmax normalisation).
            # This makes the source coupling resolution-independent:
            # K(x_j, x_i) depends only on physical distance, not on how
            # many mesh vertices fall inside the kernel width.
            distances = euclidean_distances(X, self.vertices)
            weights = np.exp(-distances**2 / (2 * self.sigma_k**2))

            # J_j = sum_i K_ij * (y_i - phi_j)
            weight_sums = weights.sum(axis=0)  # (N,)
            J = weights.T @ y - weight_sums[:, np.newaxis] * self.phi

        return J / batch_size

    def step(self, X_batch: np.ndarray = None, y_batch: np.ndarray = None):
        """
        Perform one time step of field evolution.

        Parameters
        ----------
        X_batch, y_batch : np.ndarray or None
            Batch of data for source term. If None, J=0.
        """
        if X_batch is not None and y_batch is not None:
            J = self.source_term(X_batch, y_batch)
        else:
            J = np.zeros_like(self.phi)

        if self.tau > 0:
            # Telegraph equation: tau * phi_tt + phi_t = F(phi) + noise
            # CFL for the damped wave equation with max eigenvalue lambda_max:
            #   dt < min(tau, 2*sqrt(tau/lambda_max))
            # Apply a safety factor of 0.5.
            dt_cfl = min(0.5 * self.tau,
                         np.sqrt(self.tau / max(self.gamma * self.lambda_max, 1.0)))
            n_substeps = max(1, int(np.ceil(self.eta / dt_cfl)))
            dt = self.eta / n_substeps

            for _ in range(n_substeps):
                phi_norm_sq = np.sum(self.phi**2, axis=1)
                Lphi = self.M_inv[:, np.newaxis] * (self.L @ self.phi)
                force = -self.gamma * Lphi + self.alpha * self.phi \
                    - self.beta * phi_norm_sq[:, np.newaxis] * self.phi + J
                # Scale noise to preserve total diffusion across sub-steps
                noise = self.sigma * np.sqrt(self.eta / dt) \
                    * self.rng.standard_normal(self.phi.shape)

                self.phi_dot = (1 - dt / self.tau) * self.phi_dot \
                    + (dt / self.tau) * (force + noise)
                self.phi = self.phi + dt * self.phi_dot
        else:
            # Parabolic (gradient flow) equation: phi_t = F(phi) + noise
            # CFL: dt < 2/lambda_max; use safety factor 0.5.
            dt_cfl = 1.0 / max(self.gamma * self.lambda_max, 1.0)
            n_substeps = max(1, int(np.ceil(self.eta / dt_cfl)))
            dt = self.eta / n_substeps
            for _ in range(n_substeps):
                phi_norm_sq = np.sum(self.phi**2, axis=1)
                Lphi = self.M_inv[:, np.newaxis] * (self.L @ self.phi)
                force = -self.gamma * Lphi + self.alpha * self.phi \
                    - self.beta * phi_norm_sq[:, np.newaxis] * self.phi + J
                noise = self.sigma * np.sqrt(self.eta / dt) \
                    * self.rng.standard_normal(self.phi.shape)
                self.phi = self.phi + dt * (force + noise)

        self.t += 1

        # Detect numerical divergence early
        phi_max = np.max(np.abs(self.phi))
        if not np.isfinite(phi_max):
            logger.error(
                "Field diverged to inf/NaN at step %d. "
                "Consider reducing eta or sigma.", self.t
            )
        elif phi_max > 1e6:
            logger.warning(
                "Large field norm (max |phi| = %.2e) at step %d. "
                "Possible numerical instability.", phi_max, self.t
            )

        # Record scalar history (no full phi snapshots to save memory)
        self.history['t'].append(self.t)
        self.history['energy'].append(self.free_energy())

    def correlation_length(self, phi: np.ndarray = None) -> float:
        """Compute correlation length from binned spatial correlation function.

        Uses the connected two-point function G(r), binned by geodesic
        distance.  The second-moment correlation length is then

            xi^2 = sum_r  r^2 G(r)  /  (2 d sum_r G(r))

        where the sums run only over bins with G(r) > 0 (short-range
        positive correlations).  This avoids the degenerate denominator
        that arises from sum_{ij} C_{ij} = 0 on compact manifolds.
        """
        if phi is None:
            phi = self.phi

        if not np.all(np.isfinite(phi)):
            return 0.0

        N = len(phi)
        dim = self.vertices.shape[1]

        phi_c = phi - phi.mean(axis=0, keepdims=True)

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            D = euclidean_distances(self.vertices)
            C = phi_c @ phi_c.T / phi.shape[1]

        # Upper triangle (exclude self-correlations on the diagonal)
        triu = np.triu_indices(N, k=1)
        d_pairs = D[triu]
        c_pairs = C[triu]

        # Bin the pair correlation by distance
        n_bins = 30
        bin_edges = np.linspace(0, d_pairs.max() + 1e-10, n_bins + 1)
        bin_idx = np.digitize(d_pairs, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        G_sum = np.bincount(bin_idx, weights=c_pairs, minlength=n_bins)

        valid = counts > 0
        G = np.zeros(n_bins)
        G[valid] = G_sum[valid] / counts[valid]

        r = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Second-moment from the positive part of G(r)
        pos = valid & (G > 0)
        if not np.any(pos):
            return 0.0

        numerator = np.sum(r[pos] ** 2 * G[pos])
        denominator = 2 * dim * np.sum(G[pos])

        if denominator > 0:
            return float(np.sqrt(numerator / denominator))
        return 0.0

    def spectral_gap(self) -> float:
        """Compute spectral gap of the Laplace-Beltrami operator.

        Solves the generalised eigenvalue problem ``L u = lambda M u``
        so that the returned eigenvalue is resolution-independent
        (converges to the continuous spectrum as *N* grows).
        """
        n_eig = min(10, self.N - 1)
        M_diag = sp.diags(self.M_lumped)
        try:
            eigvals = spla.eigsh(self.L, k=n_eig, M=M_diag,
                                 which='SM', return_eigenvectors=False)
            positive = eigvals[eigvals > 1e-10]
            if len(positive) == 0:
                logger.warning("No positive eigenvalues found; returning 0.")
                return 0.0
            return float(np.min(positive))
        except (spla.ArpackNoConvergence, spla.ArpackError) as e:
            logger.warning(f"ARPACK did not converge: {e}. Using dense fallback.")
            # Symmetrise via M^{-1/2}: M^{-1/2} L M^{-1/2} z = lambda z
            L_dense = self.L.toarray()
            M_sqrt_inv = np.sqrt(self.M_inv)
            L_sym = M_sqrt_inv[:, np.newaxis] * L_dense * M_sqrt_inv[np.newaxis, :]
            eigvals = np.linalg.eigvalsh(L_sym)
            positive = eigvals[eigvals > 1e-10]
            if len(positive) == 0:
                return 0.0
            return float(np.min(positive))

    def physical_mass_gap(self) -> float:
        """Compute physical mass gap of the linearised theory.

        In the subcritical regime (alpha < lambda_1), the mass gap is
        m^2 = lambda_1 - alpha, which controls how strongly spatial
        fluctuations are suppressed.  Near the critical point alpha -> lambda_1,
        the mass vanishes and the field becomes susceptible to perturbations.
        """
        lambda_1 = self.spectral_gap()
        m_sq = lambda_1 - self.alpha
        return float(np.sqrt(max(m_sq, 0.0)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for data points.

        The spatial mean of phi is subtracted before taking argmax so
        that the prediction reflects the source-driven spatial variation
        rather than the uniform (ell=0) component of the GL dynamics.
        """
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            distances = euclidean_distances(X, self.vertices)
        nearest = np.argmin(distances, axis=1)
        phi_centered = self.phi - self.phi.mean(axis=0, keepdims=True)
        predictions = phi_centered[nearest]
        return np.argmax(predictions, axis=1)

    def save(self, filename: str):
        """Save model state using numpy format."""
        np.savez(
            filename,
            phi=self.phi,
            phi_dot=self.phi_dot,
            t=self.t,
            history_t=np.array(self.history['t']),
            history_energy=np.array(self.history['energy']),
            history_correlation_length=np.array(
                self.history.get('correlation_length', [])
            ),
            history_accuracy=np.array(self.history.get('accuracy', [])),
            manifold_type=self.manifold_type,
            N=self.N,
            alpha=self.alpha,
            beta=self.beta,
            eta=self.eta,
            sigma=self.sigma,
            tau=self.tau,
            gamma=self.gamma,
            sigma_k=self.sigma_k,
            seed=self.seed,
        )

    @classmethod
    def load(cls, filename: str):
        """Load model state from numpy format."""
        data = np.load(filename, allow_pickle=False)

        model = cls(
            manifold_type=str(data['manifold_type']),
            N=int(data['N']),
            alpha=float(data['alpha']),
            beta=float(data['beta']),
            eta=float(data['eta']),
            sigma=float(data['sigma']),
            tau=float(data['tau']),
            gamma=float(data['gamma']),
            sigma_k=float(data['sigma_k']),
            seed=int(data['seed']),
        )

        model.phi = data['phi']
        model.phi_dot = data['phi_dot']
        model.t = int(data['t'])
        model.history = {
            't': data['history_t'].tolist(),
            'energy': data['history_energy'].tolist(),
            'correlation_length': data['history_correlation_length'].tolist(),
            'accuracy': data['history_accuracy'].tolist(),
        }
        return model


def generate_hierarchical_data(N_samples: int = 1000,
                               n_categories: int = 8,
                               manifold: str = 'sphere',
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate hierarchical classification data on manifold.

    Parameters
    ----------
    N_samples : int
        Number of samples per category
    n_categories : int
        Number of categories (should be power of 2 for hierarchical structure)
    manifold : str
        'sphere' or 'hyperbolic'
    seed : int
        Random seed

    Returns
    -------
    X : np.ndarray, shape (N_samples * n_categories, dim)
        Data points
    y : np.ndarray, shape (N_samples * n_categories,)
        Category labels (0 to n_categories-1)
    y_onehot : np.ndarray, shape (N_samples * n_categories, n_categories)
        One-hot encoded labels
    centers : np.ndarray, shape (n_categories, dim)
        Category prototype centers
    """
    rng = np.random.default_rng(seed)

    if manifold == 'sphere':
        # Generate category centers (evenly spaced on sphere via Fibonacci spiral)
        centers = []
        for i in range(n_categories):
            phi_angle = i * np.pi * (3 - np.sqrt(5))
            z = 1 - (i / (n_categories - 1)) * 2
            radius = np.sqrt(1 - z**2)
            x = np.cos(phi_angle) * radius
            y_coord = np.sin(phi_angle) * radius
            centers.append([x, y_coord, z])
        centers = np.array(centers)

        # Generate samples from approximate von Mises-Fisher distributions
        X = []
        y = []
        kappa = 10.0

        for c in range(n_categories):
            center = centers[c]
            for _ in range(N_samples):
                u = rng.standard_normal(3)
                u = u / np.linalg.norm(u)
                direction = center + 0.1 * rng.standard_normal(3)
                point = u + kappa * direction
                point = point / np.linalg.norm(point)
                X.append(point)
                y.append(c)

    elif manifold == 'hyperbolic':
        # Generate category centers in Poincare disk
        max_r = 0.8
        centers = []
        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False)
        radii = np.linspace(0.1, max_r, int(np.ceil(np.sqrt(n_categories))))

        idx = 0
        for r in radii:
            for angle in angles[:len(angles) // len(radii) + 1]:
                if idx >= n_categories:
                    break
                centers.append([r * np.cos(angle), r * np.sin(angle)])
                idx += 1

        centers = np.array(centers[:n_categories])

        X = []
        y = []
        spread = 0.1

        for c in range(n_categories):
            center = centers[c]
            for _ in range(N_samples):
                point = center + spread * rng.standard_normal(2)
                norm = np.linalg.norm(point)
                if norm >= 0.95:
                    point = point * 0.95 / norm
                X.append(point)
                y.append(c)

    else:
        raise ValueError(f"Unknown manifold: {manifold}")

    X = np.array(X)
    y = np.array(y)

    y_onehot = np.zeros((len(y), n_categories))
    y_onehot[np.arange(len(y)), y] = 1

    return X, y, y_onehot, centers


def finite_size_scaling_analysis():
    """Perform finite-size scaling analysis."""
    N_values = [642, 2562, 10242]
    results = {}
    rng = np.random.default_rng(42)

    for N in N_values:
        print(f"Running analysis for N={N}")

        model = DataFieldTheory(N=N, seed=42)
        model.initialize_field(k=8)

        X, _, y_onehot, _ = generate_hierarchical_data(
            N_samples=500, n_categories=8, manifold='sphere'
        )

        n_steps = 200
        xi_history = []

        for step in range(n_steps):
            batch_size = 32
            idx = rng.choice(len(X), batch_size, replace=False)
            X_batch = X[idx]
            y_batch = y_onehot[idx]

            model.step(X_batch, y_batch)

            if step % 10 == 0:
                xi = model.correlation_length()
                xi_history.append((step, xi))

        results[N] = {
            'xi_history': xi_history,
            'final_phi': model.phi.copy()
        }

    return results


def plot_critical_phenomena(results: dict):
    """Plot critical phenomena analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    for N, data in results.items():
        steps, xi_vals = zip(*data['xi_history'])
        ax.plot(steps, xi_vals, 'o-', label=f'N={N}')
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'Correlation length $\xi(t)$')
    ax.set_title('Finite-size scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    N_max = max(results.keys())
    steps, xi_vals = zip(*results[N_max]['xi_history'])
    f, Pxx = signal.welch(xi_vals, fs=0.1, nperseg=min(256, len(xi_vals)))
    ax.loglog(f, Pxx, 'r-', linewidth=2)
    ax.loglog(f, f**(-1.0), 'k--', label='1/f reference')
    ax.set_xlabel('Frequency f')
    ax.set_ylabel('Power spectral density S(f)')
    ax.set_title('1/f noise spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    t_c = 100
    nu = 0.63
    for N, data in results.items():
        steps, xi_vals = zip(*data['xi_history'])
        steps = np.array(steps)
        xi_vals = np.array(xi_vals)
        t_scaled = (steps - t_c) * N**(1 / (nu * 3))
        xi_scaled = xi_vals / np.max(xi_vals)
        ax.plot(t_scaled, xi_scaled, 's-', label=f'N={N}')
    ax.set_xlabel(r'Scaled time $(t-t_c)N^{1/\nu d}$')
    ax.set_ylabel(r'Scaled correlation length $\xi/\xi_{\max}$')
    ax.set_title(f'Data collapse ($\\nu={nu}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('critical_phenomena.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Data Field Theory Implementation")
    print("=" * 50)

    print("\n1. Generating hierarchical data...")
    X, y, y_onehot, centers = generate_hierarchical_data(
        N_samples=1000, n_categories=8, manifold='sphere'
    )
    print(f"   Generated {len(X)} samples with {len(np.unique(y))} categories")

    print("\n2. Initializing DFT model...")
    model = DataFieldTheory(N=2562, seed=42)
    model.initialize_field(k=8)
    print(f"   Model initialized with {model.N} vertices")

    rng_main = np.random.default_rng(42)
    print("\n3. Training for 100 steps...")
    for step in range(100):
        batch_size = 32
        idx = rng_main.choice(len(X), batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y_onehot[idx]

        model.step(X_batch, y_batch)

        if step % 20 == 0:
            xi = model.correlation_length()
            energy = model.free_energy()
            print(f"   Step {step}: xi={xi:.3f}, F={energy:.3f}")

    print("\n4. Computing spectral gap...")
    m_gap = model.spectral_gap()
    print(f"   Spectral gap m_gap = {m_gap:.4f}")

    print("\n5. Performing finite-size scaling...")
    results = finite_size_scaling_analysis()
    plot_critical_phenomena(results)

    print("\n6. Saving model...")
    model.save('dft_model.npz')
    print("   Model saved to 'dft_model.npz'")

    print("\nDone!")
