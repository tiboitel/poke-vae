# 🧬 PokeVAE — Variational Autoencoder for Pokémon Stats, Types & Abilities

## Model Summary

**PokeVAE** is a **Variational Autoencoder (VAE)** trained on Pokémon base data to learn a continuous latent representation of Pokémon **base stats**, **types**, and **abilities**.

The model enables:
- Random Pokémon-like generation
- Latent interpolation and Pokémon fusion
- Alternative / regional form generation via latent perturbation

This project is designed for **creative exploration**, procedural generation, and educational experimentation with VAEs.

---

## Model Details

- **Model type:** Variational Autoencoder (VAE)
- **Framework:** PyTorch
- **Latent dimension:** 64
- **Input modalities:**
  - Base stats (standardized)
  - Pokémon types (multi-label)
  - Pokémon abilities (single-label)
- **Outputs:**
  - Base stats (continuous)
  - Types (multi-label logits)
  - Ability (categorical logits)

---

## Architecture

### Encoder
- Fully connected MLP
- Input dimension:
	- 6 (stats) + 18 (types) + 286 (abilities) = 310
	- Hidden dimension: 256
- Outputs:
	- Mean vector μ ∈ ℝ⁶⁴
	- Log-variance vector log σ² ∈ ℝ⁶⁴

### Decoder
- MLP with dropout
- Hidden layers: `[256, 128]`
- Output heads:
	- **Stats head:** linear (regression)
	- **Type head:** linear (binary logits)
	- **Talent head:** linear (categorical logits)

---

## Training

### Dataset

- Pokémon dataset loaded from `pokemons.json`
- Features:
- Base stats: `HP, Attack, Defense, Sp. Attack, Sp. Defense, Speed`
- Types: multi-hot encoding
- Abilities: one-hot encoding (first listed ability only)

### Preprocessing

- Base stats are standardized using `sklearn.StandardScaler`
- Types are treated as a multi-label classification problem
- Abilities are treated as categorical classification

---

### Loss Function

The total loss is a weighted sum of reconstruction losses and KL divergence:

```
L = MSE(stats)

BCE(types)

0.30 × CE(ability)

β × KL
```
Where:
- **MSE:** Mean Squared Error for base stats
- **BCE:** Binary Cross-Entropy with logits for types
- **CE:** Cross-Entropy for abilities
- **KL:** Kullback–Leibler divergence
- **β:** linearly annealed during the first 100 epochs
- **β_max:** 0.012

KL divergence is clamped to prevent posterior collapse.

---

### Optimization

- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: `64`
- Training epochs: `600`

---

## Inference & Generation

### Random Sampling

Samples from the latent prior and decodes into Pokémon-like entries:
- Ensures at least one type is assigned
- Ability selected via argmax
- Stats are inverse-transformed for readability

Implemented in `inference.py`.

---

### Alternative / Regional Forms

Generates a variant of an existing Pokémon by adding noise in latent space:
- Preserves original Base Stat Total (BST)
- Produces new stat distributions
- Allows type and ability changes

---

### Pokémon Fusion

Creates fusion forms by averaging latent vectors of two Pokémon:
- Stats decoded from latent space
- BST normalized to parent average
- Types and abilities sampled **only from parent Pokémon**

---

## Intended Uses

### Intended Use
- Procedural Pokémon-like content generation
- Latent space exploration
- Educational demonstrations of VAEs
- Creative tooling (fusions, variants)

### Out-of-Scope Use
- Competitive Pokémon balancing
- Canonical or official Pokémon creation
- Real-world decision making

---

## Evaluation

This model is **not quantitatively benchmarked**.

Evaluation is qualitative and exploratory:
- Plausibility of generated stats
- Smoothness of latent interpolations
- Diversity of generated forms

---

## Limitations

- Only the **first listed ability** is modeled
- No hard constraints on stat realism beyond BST normalization
- Type correlations are learned implicitly
- Dataset biases directly affect generations

---

## Ethical Considerations

This model generates **fictional content** inspired by Pokémon data.

It is intended strictly for **educational and creative use**.  
All Pokémon-related concepts, names, and data are the property of their respective rights holders.

---

## Reproducibility

### Training

You can change training hyperparameters in `config.json`. 

Start training:
```bash
python train.py
```

### Saved artifacts:

Model weights
StandardScaler statistics
Type and ability vocabularies
Training configuration

## Requirements

See requirements.txt:

```
PyTorch ≥ 2.1
scikit-learn
NumPy
```

## Model Card Authors

**Author**: tiboitel
**Model name**: poke-vae

