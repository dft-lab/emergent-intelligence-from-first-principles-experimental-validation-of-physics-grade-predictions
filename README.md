# Data Field Theory: Emergent Intelligence from First Principles

Experimental validation of physics-grade predictions using Data Field Theory (DFT) - a framework connecting emergent intelligence to first principles from physics.

## Installation

```bash
git clone git@github.com:dft-lab/emergent-intelligence-from-first-principles-experimental-validation-of-physics-grade-predictions.git
cd emergent-intelligence-from-first-principles-experimental-validation-of-physics-grade-predictions
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from dft_core import DataFieldTheory, generate_hierarchical_data

# Generate data on the sphere
X, y, y_onehot, centers = generate_hierarchical_data(
    N_samples=500, n_categories=8, manifold='sphere'
)

# Initialize and train the model
model = DataFieldTheory(N=2562, seed=42)
model.initialize_field(k=8)

import numpy as np
rng = np.random.default_rng(42)
for step in range(100):
    idx = rng.choice(len(X), 32, replace=False)
    model.step(X[idx], y_onehot[idx])

# Predict
predictions = model.predict(X[:10])
```

### Reproducing Paper Results

Run all experiments (P1-P4):

```bash
python experiments.py
```

Generate all figures:

```bash
python reproduce_figures.py
```

### Custom Manifold Implementation

Extend the `DataFieldTheory` class to work on custom geometric manifolds:

```python
class CustomDFT(DataFieldTheory):
    def _create_manifold(self, manifold_type, N):
        # Custom manifold creation
        vertices = ...  # Your vertices
        faces = ...     # Your faces
        return vertices, faces
```

### Parameter Exploration

Study the effect of different parameters:

```python
for alpha in [0.5, 1.0, 1.5, 2.0]:
    model = DataFieldTheory(alpha=alpha)
    # Run experiments and collect results
```

### Extending to Other Riemannian Manifolds

The framework can be extended to other Riemannian manifolds. This requires proper discretization and computation of the Laplace-Beltrami operator on the target manifold.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{nehzati2024dft,
  title={Data Field Theory: Emergent Intelligence from First Principles,
         Experimental Validation of Physics-Grade Predictions},
  author={Nehzati, Mohammadreza},
  journal={Frontiers in Big Data, Machine Learning and Artificial Intelligence},
  year={2024},
  doi={10.xxxx/xxxxxx}
}
```
