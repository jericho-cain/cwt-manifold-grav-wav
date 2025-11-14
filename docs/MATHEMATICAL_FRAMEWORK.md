# Mathematical Framework: Manifold Learning for Anomaly Detection

## Abstract

We formalize the autoencoder + manifold geometry approach to gravitational wave anomaly detection using the language of differential geometry. Given a probability measure $\mu$ on the data space $\mathcal{X} \subset \mathbb{R}^n$, we posit that $\text{supp}(\mu)$ lies near a smooth manifold $\mathcal{M}$ embedded in $\mathcal{X}$. The autoencoder learns a chart to a latent space, and we use local tangent space geometry to distinguish on-manifold (normal) data from off-manifold (anomalous) data.

## 1. The Manifold Hypothesis

### 1.1 Setup

Let $\mathcal{X} \subset \mathbb{R}^n$ be the **data space**. For LISA:
- $n = 64 \times 3600 = 230{,}400$ (CWT scalogram dimensions)
- Each point $x \in \mathcal{X}$ represents a preprocessed time-frequency representation

**Manifold Hypothesis**: The data-generating distribution $\mu$ on $\mathcal{X}$ has support concentrated near a **smooth manifold** $\mathcal{M}$ of dimension $d \ll n$:

$$\mathcal{M} \subset \mathcal{X}, \quad \dim(\mathcal{M}) = d \ll n$$

### 1.2 Intrinsic Dimensionality

For LISA confusion noise:
- The intrinsic dimension $d$ is determined by the **degrees of freedom** in the confusion background
- Each unresolved GB has ~6 parameters (masses, orbital parameters)
- With $N_{\text{GB}} \approx 50$ GBs, naively $d \lesssim 300$
- However, parameter degeneracies and correlations reduce effective dimension

**Empirical estimate**: $d \approx 8$-$32$ (validated via PCA, manifold dimension estimation)

### 1.3 Regularity

We assume $\mathcal{M}$ is a **smooth embedded submanifold** of $\mathcal{X}$:
- $\mathcal{M}$ is a $C^{\infty}$ manifold
- The inclusion $\iota: \mathcal{M} \hookrightarrow \mathcal{X}$ is a smooth embedding
- At each $x \in \mathcal{M}$, the tangent space $T_x\mathcal{M}$ is well-defined

## 2. Autoencoder as Chart

### 2.1 The Autoencoder Map

An autoencoder consists of:
1. **Encoder** $\phi: \mathcal{X} \to \mathcal{Z}$ where $\mathcal{Z} \subset \mathbb{R}^d$ is the **latent space**
2. **Decoder** $\psi: \mathcal{Z} \to \mathcal{X}$

The composite $\psi \circ \phi: \mathcal{X} \to \mathcal{X}$ is the **reconstruction map**.

### 2.2 Chart Interpretation

For data $x \in \mathcal{M}$ near the manifold, the encoder acts as a **chart**:

$$\phi|_{\mathcal{M}}: \mathcal{M} \to \mathcal{Z}$$

The decoder approximates the **parametrization** (inverse chart):

$$\psi: \mathcal{Z} \to \mathcal{M} \subset \mathcal{X}$$

**Reconstruction error** measures how well we stay on the manifold:

$$\epsilon(x) = \|x - \psi(\phi(x))\|^2$$

### 2.3 Training Objective

Given training data $\{x_i\}_{i=1}^N$ drawn from $\mu$ (concentrated on $\mathcal{M}$):

$$\min_{\phi, \psi} \sum_{i=1}^N \|x_i - \psi(\phi(x_i))\|^2 + \mathcal{R}(\phi, \psi)$$

where $\mathcal{R}$ is a regularization term.

**Interpretation**: We seek coordinates $(U, \phi)$ such that:
- $U \subseteq \mathcal{M}$ is an open neighborhood of training data
- $\phi: U \to \mathcal{Z}$ is a diffeomorphism onto its image
- $\psi \approx \phi^{-1}$ reconstructs points on $\mathcal{M}$

## 3. Latent Space Geometry

### 3.1 The Latent Manifold

Let $\mathcal{M}_{\mathcal{Z}} = \phi(\mathcal{M})$ be the **image of the data manifold** in latent space.

Key observation: While $\mathcal{M} \subset \mathbb{R}^n$ is a $d$-dimensional submanifold embedded in high-dimensional space, $\mathcal{M}_{\mathcal{Z}} \subset \mathbb{R}^d$ is approximately a $d$-dimensional manifold embedded in a $d$-dimensional ambient space.

**However**: $\mathcal{M}_{\mathcal{Z}}$ need not fill all of $\mathcal{Z} = \mathbb{R}^d$. It may have:
- Lower intrinsic dimension: $\dim(\mathcal{M}_{\mathcal{Z}}) < d$
- Curved geometry
- Non-trivial topology

### 3.2 Empirical Manifold Reconstruction

Given training latents $\{z_i = \phi(x_i)\}_{i=1}^N$, we construct a **discrete approximation** of $\mathcal{M}_{\mathcal{Z}}$:

$$\mathcal{M}_{\mathcal{Z}}^N = \bigcup_{i=1}^N B_{\epsilon}(z_i)$$

where $B_{\epsilon}(z_i)$ are small balls. We use a **$k$-nearest neighbors graph**:

$$G = (V, E), \quad V = \{z_1, \ldots, z_N\}, \quad (z_i, z_j) \in E \iff z_j \in \text{kNN}_k(z_i)$$

This graph approximates the **manifold as a simplicial complex**.

### 3.3 Riemannian Structure

We equip $\mathcal{M}_{\mathcal{Z}}$ with the **pullback metric** from $\mathcal{X}$:

$$g = \phi^* h$$

where $h$ is the Euclidean metric on $\mathcal{X}$. Explicitly, for $v, w \in T_z \mathcal{M}_{\mathcal{Z}}$:

$$g_z(v, w) = h_{\psi(z)}(D\psi(z) \cdot v, D\psi(z) \cdot w)$$

In practice, we work in the **ambient Euclidean space** $\mathbb{R}^d$ with standard metric.

## 4. Tangent Space Estimation

### 4.1 Local Tangent Space

At each point $z \in \mathcal{M}_{\mathcal{Z}}$, the **tangent space** $T_z \mathcal{M}_{\mathcal{Z}}$ is a $d'$-dimensional linear subspace of $\mathbb{R}^d$ (where $d' = \dim(\mathcal{M}_{\mathcal{Z}}) \leq d$).

For the discrete approximation, we estimate $T_z \mathcal{M}_{\mathcal{Z}}^N$ using the $k$ nearest neighbors $\mathcal{N}_k(z) = \{z_{i_1}, \ldots, z_{i_k}\}$:

1. **Local mean** (point on manifold):
   $$\mu_z = \frac{1}{k} \sum_{j=1}^k z_{i_j}$$

2. **Centered neighbors** (tangent vectors):
   $$v_j = z_{i_j} - \mu_z, \quad j = 1, \ldots, k$$

3. **Covariance matrix**:
   $$C = \frac{1}{k} \sum_{j=1}^k v_j v_j^T$$

4. **PCA** to extract tangent space:
   $$C = U \Lambda U^T$$
   where $U = [u_1, \ldots, u_d]$ are eigenvectors, $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$ with $\lambda_1 \geq \ldots \geq \lambda_d \geq 0$.

### 4.2 Tangent Basis

The **tangent space** at $z$ is spanned by the top $d'$ eigenvectors:

$$T_z \mathcal{M}_{\mathcal{Z}}^N = \text{span}\{u_1, \ldots, u_{d'}\}$$

where $d'$ is chosen such that:

$$\frac{\sum_{i=1}^{d'} \lambda_i}{\sum_{i=1}^d \lambda_i} \geq 0.95$$

(i.e., 95% variance explained).

Let $U_{d'} = [u_1, \cdots, u_{d'}] \in \mathbb{R}^{d \times d'}$ be the **tangent basis matrix**.

### 4.3 Normal Space

The **normal space** $N_z \mathcal{M}_{\mathcal{Z}}^N$ is the orthogonal complement:

$$N_z \mathcal{M}_{\mathcal{Z}}^N = (T_z \mathcal{M}_{\mathcal{Z}}^N)^{\perp} = \text{span}\{u_{d'+1}, \ldots, u_d\}$$

## 5. Geometric Decomposition

### 5.1 Tangent-Normal Decomposition

For any point $z \in \mathbb{R}^d$ (possibly off-manifold), consider a **reference point** $\mu_z$ on the manifold. Decompose:

$$z - \mu_z = v_{\parallel} + v_{\perp}$$

where:
- $v_{\parallel} \in T_{\mu_z} \mathcal{M}_{\mathcal{Z}}^N$ is the **tangent component**
- $v_{\perp} \in N_{\mu_z} \mathcal{M}_{\mathcal{Z}}^N$ is the **normal component**

### 5.2 Projection Operators

The **tangent projection** is:

$$\Pi_{\parallel}(v) = U_{d'} U_{d'}^T v$$

The **normal projection** is:

$$\Pi_{\perp}(v) = (I - U_{d'} U_{d'}^T) v = v - \Pi_{\parallel}(v)$$

### 5.3 Off-Manifold Distance

The **normal deviation** (off-manifold distance) is:

$$\delta_{\perp}(z) = \|v_{\perp}\| = \|(I - U_{d'} U_{d'}^T)(z - \mu_z)\|$$

This measures how far $z$ is from the manifold in the **normal direction**.

**Geometric interpretation**: $\delta_{\perp}(z)$ is the approximate distance from $z$ to $\mathcal{M}_{\mathcal{Z}}^N$, measured perpendicular to the manifold.

### 5.4 Manifold Distance

Additionally, we can measure:

$$\delta_{\parallel}(z) = \|v_{\parallel}\| = \|U_{d'} U_{d'}^T (z - \mu_z)\|$$

This is the distance within the tangent space, related to how far $z$ is along the manifold.

## 6. Density Estimation

### 6.1 Local Density

The **density score** estimates the local probability density:

$$\rho(z) \propto \frac{1}{\text{mean}(\|z - z_j\|: z_j \in \mathcal{N}_k(z))}$$

This is a **$k$-nearest neighbor density estimator**.

### 6.2 Interpretation

- **High density** ($\rho(z)$ large): $z$ is in a region with many training points
- **Low density** ($\rho(z)$ small): $z$ is in a sparse region, possibly anomalous

**Connection to manifold**: Density captures the **volume form** on $\mathcal{M}_{\mathcal{Z}}$ (how much measure $\mu$ is concentrated locally).

## 7. Anomaly Scoring

### 7.1 The Composite Score

For a test point $x \in \mathcal{X}$, we compute:

1. **Latent representation**: $z = \phi(x)$
2. **Reconstruction error**: $\epsilon(x) = \|x - \psi(z)\|^2$
3. **Off-manifold distance**: $\delta_{\perp}(z)$
4. **Density score** (optional): $\rho(z)$

The **anomaly score** is:

$$s(x) = \alpha \cdot \epsilon(x) + \beta \cdot \delta_{\perp}(\phi(x)) + \gamma \cdot \rho^{-1}(\phi(x))$$

### 7.2 Parameter Interpretation

**$\alpha$ (reconstruction weight)**:
- Measures how well $x$ can be reconstructed via the learned chart
- Large $\epsilon(x)$ means $x$ is not well-represented by $\psi \circ \phi$

**$\beta$ (geometric weight)**:
- Measures how far $\phi(x)$ is from the manifold $\mathcal{M}_{\mathcal{Z}}$ in latent space
- Large $\delta_{\perp}$ means $\phi(x)$ is in the **normal bundle** rather than on the manifold
- **This is the key coefficient we measure!**

**$\gamma$ (density weight)**:
- Penalizes low-density regions
- Large $\rho^{-1}$ means $\phi(x)$ is in a sparse region of latent space

### 7.3 The β Coefficient

**Scientific Question**: Does geometric information ($\delta_{\perp}$) help distinguish anomalies beyond reconstruction error alone?

**Baseline** ($\beta = 0$): $s(x) = \alpha \cdot \epsilon(x)$
- Anomaly detection based solely on reconstruction
- Standard autoencoder approach

**Manifold-enhanced** ($\beta > 0$): $s(x) = \alpha \cdot \epsilon(x) + \beta \cdot \delta_{\perp}(\phi(x))$
- Incorporates geometric information
- Tests the manifold hypothesis

**LIGO result**: $\beta_{\text{opt}} = 0$ (geometry didn't help)

**LISA test**: $\beta_{\text{opt}} = ?$ (to be measured)

## 8. Theoretical Justification

### 8.1 Why Off-Manifold Distance?

**Lemma**: If training data $\{x_i\}$ lie on a manifold $\mathcal{M}$ and test point $x$ is drawn from a different distribution, then in the latent space:

$$\mathbb{E}[\delta_{\perp}(\phi(x))] > \mathbb{E}[\delta_{\perp}(\phi(x_i))]$$

**Intuition**: Normal data should lie **on** the learned manifold, anomalies should lie **off** the manifold.

### 8.2 Reconstruction vs. Geometry

**Reconstruction error** $\epsilon(x)$ measures:
- Global representability via $\psi \circ \phi$
- Includes both approximation error and off-manifold distance

**Off-manifold distance** $\delta_{\perp}(z)$ measures:
- Local geometric deviation from manifold structure
- Independent of reconstruction quality

**Key insight**: $\delta_{\perp}$ provides **complementary information** when:
- The manifold has non-trivial geometry (curvature, topology)
- Anomalies move orthogonal to manifold, not just far along it

### 8.3 LISA Hypothesis

For LISA confusion noise:
- **Training**: Confusion background (noise + 50 unresolved GBs)
- **Manifold**: $\mathcal{M} \approx$ all possible confusion backgrounds
- **On-manifold**: Similar confusion backgrounds
- **Off-manifold**: Resolvable sources (MBHBs, EMRIs)

**Prediction**: Resolvable sources should have:
1. High $\epsilon(x)$ (different from confusion)
2. High $\delta_{\perp}(z)$ (geometrically distinct)

If **$\beta > 0$**, geometry provides additional discriminative power beyond reconstruction alone.

## 9. Differential Geometry Perspective

### 9.1 Embedded Submanifold

More formally, we have:
- **Ambient space**: $(\mathcal{X}, h)$ where $\mathcal{X} = \mathbb{R}^n$ with Euclidean metric $h$
- **Data manifold**: $\mathcal{M}$ is a $d$-dimensional smooth submanifold
- **Embedding**: $\iota: \mathcal{M} \hookrightarrow \mathcal{X}$
- **Induced metric**: $g = \iota^* h$ (first fundamental form)

### 9.2 Normal Bundle

The **normal bundle** $N\mathcal{M}$ consists of vectors orthogonal to $T\mathcal{M}$:

$$N\mathcal{M} = \bigcup_{x \in \mathcal{M}} N_x \mathcal{M}$$

where $N_x \mathcal{M} = (T_x \mathcal{M})^{\perp} \subset T_x \mathcal{X} = \mathbb{R}^n$.

**Tubular neighborhood theorem**: There exists $\epsilon > 0$ such that the **exponential map**:

$$\exp^{\perp}: \{v \in N\mathcal{M} : \|v\| < \epsilon\} \to \mathcal{X}$$

given by $\exp^{\perp}(x, v) = x + v$ is a diffeomorphism onto its image.

**Interpretation**: Points near $\mathcal{M}$ can be uniquely decomposed as (point on manifold) + (normal vector).

### 9.3 Distance to Manifold

For $x \in \mathcal{X}$ close to $\mathcal{M}$, the **distance** to the manifold is:

$$d(x, \mathcal{M}) = \min_{y \in \mathcal{M}} \|x - y\|$$

In the tubular neighborhood, if $x = p + v$ with $p \in \mathcal{M}$, $v \in N_p \mathcal{M}$, then:

$$d(x, \mathcal{M}) \approx \|v\|$$

**Our estimator**: $\delta_{\perp}(z)$ approximates this distance in latent space.

### 9.4 Second Fundamental Form

The **second fundamental form** $h$ (shape operator) captures curvature:

$$h: T_x \mathcal{M} \times T_x \mathcal{M} \to N_x \mathcal{M}$$

For highly curved manifolds, the normal direction changes rapidly.

**Relevance**: If $\mathcal{M}$ has significant curvature:
- Tangent space approximation is local
- $k$-NN must be small enough to capture local geometry
- Off-manifold detection is more sensitive

### 9.5 Connection to Geodesics

On $(\mathcal{M}, g)$, **geodesics** are curves $\gamma: [0,1] \to \mathcal{M}$ satisfying:

$$\nabla_{\dot{\gamma}} \dot{\gamma} = 0$$

where $\nabla$ is the Levi-Civita connection.

**Geodesic distance** between $p, q \in \mathcal{M}$:

$$d_{\mathcal{M}}(p, q) = \inf_{\gamma} L(\gamma)$$

where $L(\gamma) = \int_0^1 \|\dot{\gamma}(t)\| dt$.

**Our $k$-NN graph** approximates geodesic distance via graph distances (Dijkstra on $G$).

## 10. Computational Algorithm

### 10.1 Training Phase

**Input**: Training data $\{x_i\}_{i=1}^N$ (confusion background)

1. **Preprocess**: $x_i \leftarrow \text{CWT}(\text{signal}_i)$
2. **Train autoencoder**:
   $$\min_{\phi, \psi} \sum_{i=1}^N \|x_i - \psi(\phi(x_i))\|^2$$
3. **Extract latents**: $z_i = \phi(x_i)$ for all $i$
4. **Build $k$-NN graph**: For each $z_i$, find $k$ nearest neighbors
5. **Estimate tangent spaces**: For each $z_i$, compute PCA on $\mathcal{N}_k(z_i)$
6. **Store**: $\{z_i, U_i, \lambda_i\}_{i=1}^N$ (latents, tangent bases, eigenvalues)

### 10.2 Testing Phase

**Input**: Test point $x$, trained manifold $\mathcal{M}_{\mathcal{Z}}^N$

1. **Encode**: $z = \phi(x)$
2. **Find neighbors**: $\mathcal{N}_k(z) = k\text{-NN}(z, \{z_i\})$
3. **Compute local geometry**:
   - $\mu_z = \frac{1}{k} \sum_{z_j \in \mathcal{N}_k(z)} z_j$
   - $C_z = \frac{1}{k} \sum_{z_j \in \mathcal{N}_k(z)} (z_j - \mu_z)(z_j - \mu_z)^T$
   - $U_z \Lambda_z U_z^T = C_z$ (eigendecomposition)
4. **Compute scores**:
   - $\epsilon = \|x - \psi(z)\|^2$
   - $\delta_{\perp} = \|(I - U_z U_z^T)(z - \mu_z)\|$
   - $s = \alpha \cdot \epsilon + \beta \cdot \delta_{\perp}$

**Output**: Anomaly score $s$

### 10.3 Complexity

- **Training**: $O(N)$ forward passes + $O(Ndk^2)$ for $k$-NN + PCA
- **Testing**: $O(N)$ for $k$-NN search + $O(k d^2)$ for PCA
- **Space**: $O(Nd + Nd^2)$ for latents + tangent bases

With $N \sim 10^3$, $d \sim 32$, $k \sim 32$, this is very efficient.

## 11. LISA-Specific Considerations

### 11.1 Confusion Manifold

For LISA with $M$ unresolved GBs, each with parameters $\theta_j \in \mathbb{R}^{6}$:

$$\mathcal{M}_{\text{confusion}} \approx \{\text{CWT}(\sum_{j=1}^M h(t; \theta_j) + n(t)): \theta_j \in \Theta, n \in \mathcal{N}\}$$

where:
- $h(t; \theta)$ is the GB waveform
- $\Theta$ is the parameter space
- $\mathcal{N}$ is the instrumental noise space

**Intrinsic dimension**: Not simply $6M$ due to:
- Parameter degeneracies (e.g., similar frequencies)
- Overlap (unresolved sources blend)
- Noise averaging

**Estimate**: $d \ll 6M$, empirically $d \sim 8$-$32$.

### 11.2 Resolvable Sources

Resolvable sources (MBHBs, EMRIs) have:
- **Higher amplitude**: SNR $\sim 10$-$50$ vs. confusion SNR $< 5$
- **Different waveform structure**: Chirping, precession
- **Distinct frequency evolution**: Not constant-frequency like GBs

**Hypothesis**: Resolvable sources lie **off** $\mathcal{M}_{\text{confusion}}$ because:
1. They don't satisfy the "sum of weak GBs" structure
2. Their high SNR places them in a different amplitude regime
3. Their temporal evolution differs from confusion background

### 11.3 Why Manifold Geometry Might Help

**LIGO** ($\beta = 0$):
- Signals: BBH mergers, all similar waveform shapes
- Noise: Stationary Gaussian + glitches
- Manifold: Low-dimensional, simple structure
- Result: Reconstruction error sufficient, geometry redundant

**LISA** ($\beta > 0?$):
- Background: Overlapping signals (confusion noise)
- Manifold: High-dimensional, complex structure
- Hypothesis: Geometry captures **interaction structure** between GBs
- Prediction: Resolvable sources break this structure → large $\delta_{\perp}$

### 11.4 The Discriminative Power of β

If $\beta_{\text{opt}} > 0$, it means:

$$\mathbb{P}(\text{signal} | s_{\beta>0}) > \mathbb{P}(\text{signal} | s_{\beta=0})$$

i.e., manifold geometry **provides additional discriminative power** beyond reconstruction error alone.

**Physical interpretation**: The confusion background has geometric structure (manifold curvature, tangent space alignment) that resolvable sources violate.

## 12. Information-Theoretic Perspective

### 12.1 Intrinsic Dimensionality

The **information dimension** of $\mu$ is:

$$d_{\text{info}}(\mu) = \lim_{\epsilon \to 0} \frac{\log N_{\epsilon}(\text{supp}(\mu))}{\log(1/\epsilon)}$$

where $N_{\epsilon}$ is the $\epsilon$-covering number.

**Manifold hypothesis**: $d_{\text{info}}(\mu) \approx d \ll n$

### 12.2 Rate-Distortion

The autoencoder achieves **compression** from $\mathbb{R}^n$ to $\mathbb{R}^d$ with distortion:

$$D = \mathbb{E}[\|x - \psi(\phi(x))\|^2]$$

**Shannon's rate-distortion theorem**: There exists a fundamental tradeoff between rate $R = d \log_2 |\mathcal{Z}|$ and distortion $D$.

**Manifold structure** allows low distortion at low rate because data has low intrinsic dimension.

## 13. Connection to Other Methods

### 13.1 Principal Component Analysis (PCA)

PCA is a **global linear manifold**:
- Manifold: $\mathcal{M}_{\text{PCA}} = \{x_0 + \sum_{i=1}^d \alpha_i v_i: \alpha_i \in \mathbb{R}\}$
- Autoencoder: $\phi(x) = V^T(x - x_0)$, $\psi(z) = x_0 + Vz$

Our approach: **Local nonlinear manifold** via autoencoder + tangent PCA

### 13.2 Kernel PCA

Kernel PCA uses:
- Implicit manifold in feature space $\mathcal{H}$ (RKHS)
- Nonlinear $\phi: \mathcal{X} \to \mathcal{H}$

Our approach: **Explicit latent space** $\mathcal{Z}$ learned by neural network

### 13.3 Isomap / LLE

Isomap and Locally Linear Embedding:
- **Isomap**: Preserves geodesic distances
- **LLE**: Preserves local linear reconstructions

Our approach: **Autoencoder** for global embedding + **local tangent PCA** for geometry

### 13.4 VAE / Normalizing Flows

**VAE**: Probabilistic latent space with explicit density $p(z)$
- Anomaly score: negative ELBO $-\log p(x)$

**Our approach**: Geometric latent space with manifold structure
- Anomaly score: $\alpha \epsilon + \beta \delta_{\perp}$

**Complementary**: Could combine VAE latent + manifold geometry!

## 14. Future Directions

### 14.1 Riemannian Autoencoder

Use the **pullback metric** explicitly:

$$g_{ij}(z) = \langle \partial_i \psi(z), \partial_j \psi(z) \rangle$$

Compute geodesic distances on $(\mathcal{M}_{\mathcal{Z}}, g)$ rather than Euclidean distances.

### 14.2 Curvature-Based Scores

Estimate **Riemann curvature tensor**:

$$R: T\mathcal{M} \times T\mathcal{M} \times T\mathcal{M} \to T\mathcal{M}$$

High curvature regions may be more sensitive to anomalies.

### 14.3 Topological Data Analysis

Use **persistent homology** to:
- Detect topological features of $\mathcal{M}_{\mathcal{Z}}$ (loops, voids)
- Anomalies that create/destroy topological features

### 14.4 Geometric Deep Learning

Exploit **symmetries** (e.g., time-translation, frequency-shift) to:
- Learn equivariant representations
- Reduce manifold complexity

## References

**Differential Geometry**:
- Lee, J.M. (2012). *Introduction to Smooth Manifolds*. Springer.
- Lee, J.M. (2018). *Introduction to Riemannian Manifolds*. Springer.

**Manifold Learning**:
- Bengio, Y., et al. (2013). "Representation Learning: A Review and New Perspectives." *IEEE TPAMI*.
- Fefferman, C., et al. (2016). "Testing the Manifold Hypothesis." *JAMS*.

**Geometric Deep Learning**:
- Bronstein, M., et al. (2017). "Geometric Deep Learning: Going Beyond Euclidean Data." *IEEE Signal Processing Magazine*.

**Gravitational Waves**:
- Coughlin, M., et al. (2019). "Classifying the Unknown: Discovering Novel Gravitational-wave Detector Glitches using Similarity Learning." *Physical Review D*.

---

**Summary**: We formalize the autoencoder + manifold approach using differential geometry. The key quantity is the off-manifold distance $\delta_{\perp}(z)$, measuring deviation from the learned manifold structure. The β coefficient quantifies whether this geometric information improves anomaly detection beyond reconstruction error alone. For LISA, we hypothesize $\beta > 0$ due to the complex geometric structure of confusion noise.

