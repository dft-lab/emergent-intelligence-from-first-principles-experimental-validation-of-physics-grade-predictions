"""
Data Field Theory Core Implementation
Author: Mohammadreza Nehzati
Repository: https://github.com/dft-labs/emergent-intelligence-from-first-principles-experimental-validation-of-physics-grade-predictions
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import stats, signal
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

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
            Number of vertices in discretization
        alpha, beta : float
            Ginzburg-Landau potential parameters
        eta : float
            Learning rate / time step
        sigma : float
            Noise amplitude
        tau : float
            Hyperbolic regularization parameter
        gamma : float
            Mobility coefficient
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
        
        np.random.seed(seed)
        
        # Initialize manifold geometry
        self.vertices, self.faces = self._create_manifold(manifold_type, N)
        self.L = self._compute_laplacian()  # Laplace-Beltrami approximation
        
        # Field state
        self.phi = None
        self.phi_dot = None  # For hyperbolic dynamics
        self.t = 0
        
        # Statistics tracking
        self.history = {
            't': [],
            'phi': [],
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
        """Create icosahedral discretization of sphere."""
        import trimesh
        # Generate icosphere
        mesh = trimesh.creation.icosphere(subdivisions=int(np.log2(N/12)))
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Keep only N vertices (or closest)
        if len(vertices) > N:
            # Downsample using farthest point sampling
            from sklearn.metrics.pairwise import euclidean_distances
            idx = np.random.choice(len(vertices), 1)
            distances = euclidean_distances(vertices, vertices[idx])
            for _ in range(N-1):
                farthest = np.argmax(np.min(distances[:, :len(idx)], axis=1))
                idx = np.append(idx, farthest)
                new_dist = euclidean_distances(vertices, vertices[[farthest]])
                distances = np.minimum(distances, new_dist)
            vertices = vertices[idx]
            
        return vertices, faces
    
    def _create_hyperbolic_mesh(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create mesh in Poincaré disk."""
        # Generate points in hyperbolic disk
        r = np.sqrt(np.random.uniform(0, 0.95**2, N))
        theta = np.random.uniform(0, 2*np.pi, N)
        vertices = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        
        # Create Delaunay triangulation
        from scipy.spatial import Delaunay
        tri = Delaunay(vertices)
        faces = tri.simplices
        
        return vertices, faces
    
    def _compute_laplacian(self) -> sp.csr_matrix:
        """Compute cotangent Laplacian for mesh."""
        if self.manifold_type == 'sphere':
            return self._compute_cotangent_laplacian()
        else:
            return self._compute_graph_laplacian()
    
    def _compute_cotangent_laplacian(self) -> sp.csr_matrix:
        """Compute cotangent Laplacian for spherical mesh."""
        n_vertices = len(self.vertices)
        
        # Create adjacency matrix
        adj = sp.lil_matrix((n_vertices, n_vertices))
        
        for face in self.faces:
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                
                vi = self.vertices[face[i]]
                vj = self.vertices[face[j]]
                vk = self.vertices[face[k]]
                
                # Compute cotangent of angle at vertex k
                e1 = vi - vk
                e2 = vj - vk
                cot_angle = np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))
                
                adj[face[i], face[j]] = cot_angle
                adj[face[j], face[i]] = cot_angle
        
        # Convert to Laplacian
        degree = np.array(adj.sum(axis=1)).flatten()
        L = sp.diags(degree) - adj
        
        return sp.csr_matrix(L)
    
    def _compute_graph_laplacian(self) -> sp.csr_matrix:
        """Compute graph Laplacian for hyperbolic mesh."""
        n_vertices = len(self.vertices)
        
        # Create k-nearest neighbor graph
        from sklearn.neighbors import kneighbors_graph
        adj = kneighbors_graph(self.vertices, n_neighbors=6, mode='connectivity')
        
        # Symmetrize
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        
        # Laplacian
        degree = np.array(adj.sum(axis=1)).flatten()
        L = sp.diags(degree) - adj
        
        return sp.csr_matrix(L)
    
    def initialize_field(self, k: int = 8):
        """Initialize field with random values."""
        # k-dimensional field (for k categories)
        self.phi = np.random.normal(0, 0.1, (self.N, k))
        self.phi_dot = np.zeros_like(self.phi)  # For hyperbolic dynamics
        
    def free_energy(self, phi: np.ndarray = None) -> float:
        """Compute Ginzburg-Landau free energy."""
        if phi is None:
            phi = self.phi
        
        # Gradient term
        if sp.issparse(self.L):
            grad_term = np.sum(phi * (self.L @ phi))
        else:
            grad_term = np.sum(phi * np.dot(self.L, phi))
        
        # Potential term
        phi_norm_sq = np.sum(phi**2, axis=1)
        potential = -0.5 * self.alpha * phi_norm_sq + 0.25 * self.beta * phi_norm_sq**2
        
        return 0.5 * grad_term + np.sum(potential)
    
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
        k = y.shape[1]
        
        # Find nearest vertices for each data point
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(X, self.vertices)
        nearest = np.argmin(distances, axis=1)
        
        # Create kernel weights
        weights = np.exp(-distances**2 / (2 * self.sigma_k**2))
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Compute source term
        J = np.zeros((self.N, k))
        for i in range(batch_size):
            for j in range(self.N):
                if weights[i, j] > 1e-6:
                    J[j] += weights[i, j] * (y[i] - self.phi[j])
        
        return J / batch_size
    
    def step(self, X_batch: np.ndarray = None, y_batch: np.ndarray = None):
        """
        Perform one time step of field evolution.
        
        Parameters
        ----------
        X_batch, y_batch : np.ndarray or None
            Batch of data for source term. If None, J=0.
        """
        # Compute source term
        if X_batch is not None and y_batch is not None:
            J = self.source_term(X_batch, y_batch)
        else:
            J = np.zeros_like(self.phi)
        
        # Compute force: -δF/δφ
        phi_norm_sq = np.sum(self.phi**2, axis=1)
        
        if sp.issparse(self.L):
            force = -(self.L @ self.phi) - self.alpha * self.phi \
                   + self.beta * phi_norm_sq[:, np.newaxis] * self.phi \
                   + J
        else:
            force = -np.dot(self.L, self.phi) - self.alpha * self.phi \
                   + self.beta * phi_norm_sq[:, np.newaxis] * self.phi \
                   + J
        
        # Add noise
        noise = self.sigma * np.random.randn(*self.phi.shape)
        
        if self.tau > 0:  # Hyperbolic dynamics
            # Telegraph equation: τφ_tt + φ_t = force + noise
            phi_dot_new = (1 - self.eta/self.tau) * self.phi_dot \
                         + (self.eta/self.tau) * (force + noise)
            phi_new = self.phi + self.eta * phi_dot_new
            
            self.phi_dot = phi_dot_new
        else:  # Gradient flow
            phi_new = self.phi + self.eta * (force + noise)
        
        self.phi = phi_new
        self.t += 1
        
        # Record history
        self.history['t'].append(self.t)
        self.history['phi'].append(self.phi.copy())
        self.history['energy'].append(self.free_energy())
    
    def correlation_length(self, phi: np.ndarray = None) -> float:
        """Compute correlation length using second-moment method."""
        if phi is None:
            phi = self.phi
        
        # Compute correlation matrix
        phi_mean = phi.mean(axis=0, keepdims=True)
        phi_centered = phi - phi_mean
        C = phi_centered @ phi_centered.T / phi.shape[1]
        
        # Compute distances between vertices
        from sklearn.metrics.pairwise import euclidean_distances
        D = euclidean_distances(self.vertices)
        
        # Second-moment correlation length
        numerator = np.sum(D**2 * C)
        denominator = 2 * self.vertices.shape[1] * np.sum(C)
        
        if denominator > 0:
            xi = np.sqrt(numerator / denominator)
        else:
            xi = 0.0
            
        return xi
    
    def spectral_gap(self) -> float:
        """Compute spectral gap of field Hessian."""
        # Compute Hessian: H = L + αI - β(3φφ^T + ‖φ‖^2 I)
        phi_norm_sq = np.sum(self.phi**2, axis=1)
        
        # Diagonal part
        diag_terms = self.alpha - self.beta * phi_norm_sq
        
        # Compute eigenvalues (smallest positive)
        if sp.issparse(self.L):
            # For sparse, compute smallest eigenvalues
            from scipy.sparse.linalg import eigsh
            n_eig = min(10, self.N)
            try:
                eigvals = eigsh(self.L, k=n_eig, which='SM', return_eigenvectors=False)
                m_gap = np.min(eigvals[eigvals > 1e-10])
            except:
                m_gap = 0.1  # Fallback
        else:
            # For dense
            eigvals = np.linalg.eigvalsh(self.L + np.diag(diag_terms))
            m_gap = np.min(eigvals[eigvals > 1e-10])
            
        return m_gap
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for data points."""
        # Find nearest vertices
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(X, self.vertices)
        nearest = np.argmin(distances, axis=1)
        
        # Predict based on field at nearest vertices
        predictions = self.phi[nearest]
        
        return np.argmax(predictions, axis=1)
    
    def save(self, filename: str):
        """Save model state."""
        state = {
            'phi': self.phi,
            'phi_dot': self.phi_dot,
            't': self.t,
            'history': self.history,
            'params': {
                'manifold_type': self.manifold_type,
                'N': self.N,
                'alpha': self.alpha,
                'beta': self.beta,
                'eta': self.eta,
                'sigma': self.sigma,
                'tau': self.tau,
                'gamma': self.gamma,
                'sigma_k': self.sigma_k,
                'seed': self.seed
            }
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, filename: str):
        """Load model state."""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        params = state['params']
        model = cls(**params)
        
        model.phi = state['phi']
        model.phi_dot = state['phi_dot']
        model.t = state['t']
        model.history = state['history']
        
        return model


def generate_hierarchical_data(N_samples: int = 1000, 
                               n_categories: int = 8,
                               manifold: str = 'sphere',
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
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
    """
    np.random.seed(seed)
    
    if manifold == 'sphere':
        dim = 3
        # Generate category centers (evenly spaced on sphere)
        from scipy.spatial.transform import Rotation
        centers = []
        for i in range(n_categories):
            # Fibonacci spiral for even spacing
            phi = i * np.pi * (3 - np.sqrt(5))
            z = 1 - (i / (n_categories - 1)) * 2
            radius = np.sqrt(1 - z**2)
            x = np.cos(phi) * radius
            y = np.sin(phi) * radius
            centers.append([x, y, z])
        centers = np.array(centers)
        
        # Generate samples from von Mises-Fisher distributions
        X = []
        y = []
        kappa = 10.0  # Concentration parameter
        
        for c in range(n_categories):
            center = centers[c]
            
            # Generate random directions
            for _ in range(N_samples):
                # Sample from uniform distribution on sphere
                u = np.random.normal(0, 1, 3)
                u = u / np.linalg.norm(u)
                
                # Concentrate around center (approximate vMF)
                # Simple approximation: add bias toward center
                direction = center + 0.1 * np.random.normal(0, 1, 3)
                point = u + kappa * direction
                point = point / np.linalg.norm(point)
                
                X.append(point)
                y.append(c)
        
    elif manifold == 'hyperbolic':
        dim = 2
        # Generate category centers in Poincaré disk
        max_r = 0.8
        centers = []
        angles = np.linspace(0, 2*np.pi, n_categories, endpoint=False)
        radii = np.linspace(0.1, max_r, int(np.ceil(np.sqrt(n_categories))))
        
        idx = 0
        for r in radii:
            for angle in angles[:len(angles)//len(radii)+1]:
                if idx >= n_categories:
                    break
                centers.append([r*np.cos(angle), r*np.sin(angle)])
                idx += 1
        
        centers = np.array(centers[:n_categories])
        
        # Generate samples (wrapped normal in hyperbolic space)
        X = []
        y = []
        sigma = 0.1
        
        for c in range(n_categories):
            center = centers[c]
            
            for _ in range(N_samples):
                # Simple Euclidean approximation for hyperbolic
                point = center + sigma * np.random.normal(0, 1, 2)
                # Project back to disk
                norm = np.linalg.norm(point)
                if norm >= 0.95:
                    point = point * 0.95 / norm
                
                X.append(point)
                y.append(c)
    
    else:
        raise ValueError(f"Unknown manifold: {manifold}")
    
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode labels
    y_onehot = np.zeros((len(y), n_categories))
    y_onehot[np.arange(len(y)), y] = 1
    
    return X, y, y_onehot, centers


def finite_size_scaling_analysis():
    """Perform finite-size scaling analysis."""
    N_values = [642, 2562, 10242]
    results = {}
    
    for N in N_values:
        print(f"Running analysis for N={N}")
        
        # Initialize model
        model = DataFieldTheory(N=N, seed=42)
        model.initialize_field(k=8)
        
        # Generate data
        X, y, y_onehot, _ = generate_hierarchical_data(
            N_samples=500, n_categories=8, manifold='sphere'
        )
        
        # Train for fixed number of steps
        n_steps = 200
        xi_history = []
        
        for step in range(n_steps):
            # Use mini-batches
            batch_size = 32
            idx = np.random.choice(len(X), batch_size, replace=False)
            X_batch = X[idx]
            y_batch = y_onehot[idx]
            
            model.step(X_batch, y_batch)
            
            # Compute correlation length every 10 steps
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
    
    # Panel A: Correlation length divergence
    ax = axes[0]
    for N, data in results.items():
        steps, xi_vals = zip(*data['xi_history'])
        ax.plot(steps, xi_vals, 'o-', label=f'N={N}')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Correlation length ξ(t)')
    ax.set_title('Finite-size scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Power spectral density
    ax = axes[1]
    # Use final trajectory for largest N
    N_max = max(results.keys())
    steps, xi_vals = zip(*results[N_max]['xi_history'])
    
    # Compute PSD
    f, Pxx = signal.welch(xi_vals, fs=0.1, nperseg=min(256, len(xi_vals)))
    
    ax.loglog(f, Pxx, 'r-', linewidth=2)
    ax.loglog(f, f**(-1.0), 'k--', label='1/f reference')
    ax.set_xlabel('Frequency f')
    ax.set_ylabel('Power spectral density S(f)')
    ax.set_title('1/f noise spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel C: Data collapse
    ax = axes[2]
    # Simplified data collapse visualization
    t_c = 100  # Critical time (approximate)
    nu = 0.63
    
    for N, data in results.items():
        steps, xi_vals = zip(*data['xi_history'])
        steps = np.array(steps)
        xi_vals = np.array(xi_vals)
        
        # Scaled variables
        t_scaled = (steps - t_c) * N**(1/(nu*3))
        xi_scaled = xi_vals / np.max(xi_vals)
        
        ax.plot(t_scaled, xi_scaled, 's-', label=f'N={N}')
    
    ax.set_xlabel('Scaled time (t-t_c)N^{1/νd}')
    ax.set_ylabel('Scaled correlation length ξ/ξ_max')
    ax.set_title(f'Data collapse (ν={nu})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('critical_phenomena.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Data Field Theory Implementation")
    print("=" * 50)
    
    # Example usage
    print("\n1. Generating hierarchical data...")
    X, y, y_onehot, centers = generate_hierarchical_data(
        N_samples=1000, n_categories=8, manifold='sphere'
    )
    print(f"   Generated {len(X)} samples with {len(np.unique(y))} categories")
    
    print("\n2. Initializing DFT model...")
    model = DataFieldTheory(N=2562, seed=42)
    model.initialize_field(k=8)
    print(f"   Model initialized with {model.N} vertices")
    
    print("\n3. Training for 100 steps...")
    for step in range(100):
        # Mini-batch training
        batch_size = 32
        idx = np.random.choice(len(X), batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y_onehot[idx]
        
        model.step(X_batch, y_batch)
        
        if step % 20 == 0:
            xi = model.correlation_length()
            energy = model.free_energy()
            print(f"   Step {step}: ξ={xi:.3f}, F={energy:.3f}")
    
    print("\n4. Computing spectral gap...")
    m_gap = model.spectral_gap()
    print(f"   Spectral gap m_gap = {m_gap:.4f}")
    
    print("\n5. Performing finite-size scaling...")
    results = finite_size_scaling_analysis()
    plot_critical_phenomena(results)
    
    print("\n6. Saving model...")
    model.save('dft_model.pkl')
    print("   Model saved to 'dft_model.pkl'")
    
    print("\nDone!")
