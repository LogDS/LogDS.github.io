---
layout: post
title: Expressing Tensorflow 2.x operations in plain Mathematical Notation
tags: Tensorflow, TensorAlgebra, LinearAlgebra
date: 2026-09-29
categories: logds
tabs: true
description: Documenting some of the Tensorflow 2.x operations using Mathematical notation
---

Despite the vast documentation on Tensorflow operations, those are little to no documented in mathematical notation. This makes it hard to abstract and generalize some of those ideas, as well as make it rather impossible to make those operations accessible for the common mathematician. Here is a little subset of these operations documented in such a fashion. Furthermore, I also note how you can actually use Tensor Algebra to express theta-join operations. You can find a full list of Tensorflow operations [here](https://gist.github.com/dustinvtran/cf34557fb9388da4c9442ae25c2373c9).

## 1. Element-wise Transformations

These operations apply a scalar function $f: \mathbb{R} \to \mathbb{R}$ independently to each component of a tensor without changing its dimensions.

### Mathematical Definition
Given an $N$-order tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \dots \times I_N}$, the transformation $\mathcal{Y} = f(\mathcal{X})$ is defined component-wise as:

$$\mathcal{Y}_{i_1 i_2 \dots i_N} = f\left(\mathcal{X}_{i_1 i_2 \dots i_N}\right)$$

where $i_k \in \{1, 2, \dots, I_k\}$ for each mode $k$.

### TensorFlow Implementation
```python
import tensorflow as tf

# Example using a sigmoid activation function
Y = tf.math.sigmoid(X)
```

---

## 2. Tensor Contractions & Matrix Multiplications

These operations compute inner products over specified matching dimensions, effectively reducing the rank of the combined structures.

### Mathematical Definition

#### General Tensor Contraction
If we contract an order-$P$ tensor $\mathcal{A}$ and an order-$Q$ tensor $\mathcal{B}$ over their last $k$ and first $k$ axes respectively, using Einstein summation convention (implicit summation over repeated indices):

$$\mathcal{C}_{i_1 \dots i_{P-k} j_{k+1} \dots j_Q} = \mathcal{A}_{i_1 \dots i_{P-k} m_1 \dots m_k} \mathcal{B}^{m_1 \dots m_k}_{\phantom{m_1 \dots m_k} j_{k+1} \dots j_Q}$$

#### Batch Matrix Multiplication
For two 3D tensors representing batches of matrices, $\mathcal{A} \in \mathbb{R}^{B \times I \times J}$ and $\mathcal{B} \in \mathbb{R}^{B \times J \times K}$:

$$\mathcal{C}_{b i k} = \sum_{j=1}^{J} \mathcal{A}_{b i j} \mathcal{B}_{b j k} \implies \mathcal{C}_{b i k} = \mathcal{A}_{b i j} \mathcal{B}_{b \phantom{j} k}^{\phantom{b} j}$$

### TensorFlow Implementation
```python
# General Contraction (contracting over last 2 axes of A and first 2 axes of B)
C_dot = tf.tensordot(A, B, axes=2)

# Batch Matrix Multiplication
C_matmul = tf.linalg.matmul(A, B)
```

---

## 3. Axis Permutations and Transpositions

These transformations map tensor coordinates via a permutation function $\pi$, rearranging the axes layout.

### Mathematical Definition
Given $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \dots \times I_N}$ and a bijection $\pi$ of the set of indices $\{1, 2, \dots, N\}$, the transformed tensor $\mathcal{Y} = \text{permute}(\mathcal{X}, \pi)$ is defined by:

$$\mathcal{Y}_{i_{\pi(1)} i_{\pi(2)} \dots i_{\pi(N)}} = \mathcal{X}_{i_1 i_2 \dots i_N}$$

### TensorFlow Implementation
```python
# Permuting axes mapping index positions [0, 1, 2] -> [2, 0, 1]
Y = tf.transpose(X, perm=[2, 0, 1])
```

---

## 4. Tensor Reductions

Reduction transformations collapse specific dimensions by aggregating elements via a binary operator $\bigoplus$ (such as $\sum, \max, \prod$).

### Mathematical Definition
If reducing $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times I_3}$ along the second axis ($I_2$) using summation:

$$\mathcal{Y}_{i_1 i_3} = \sum_{i_2=1}^{I_2} \mathcal{X}_{i_1 i_2 i_3} = \mathcal{X}_{i_1 i_2 i_3} \mathbf{1}^{i_2}$$

### TensorFlow Implementation
```python
# Reduction along axis 1
Y = tf.reduce_sum(X, axis=1)
```

---

## 5. Linear Convolution Transforms

Spatial transforms that slide a localized kernel filter tensor across a multi-dimensional data tensor.

### Mathematical Definition
For a 4D input $\mathcal{X}$ (Batch $b$, Height $h$, Width $w$, Channels $c$) and a filter kernel $\mathcal{K}$ (Filter Height $k_h$, Filter Width $k_w$, Input Channels $c$, Output Filters $f$), with strides $S_h, S_w$:

$$\mathcal{Y}_{b, \, h, \, w, \, f} = \sum_{\delta h} \sum_{\delta w} \sum_{c} \mathcal{X}_{b, \, h \cdot S_h + \delta h, \, w \cdot S_w + \delta w, \, c} \cdot \mathcal{K}_{\delta h, \, \delta w, \, c, \, f}$$

### TensorFlow Implementation
```python
Y = tf.nn.conv2d(X, K, strides=[1, 1, 1, 1], padding='SAME')
```

---

## 6. Tensor Slicing

Slicing extracts contiguous sub-tensors by selecting bounded index intervals along target dimensions.

### Mathematical Definition
Given $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \dots \times I_N}$, a slice with starting indices $s_k$, ending indices $e_k$, and steps $t_k$ creates an output $\mathcal{Y} \in \mathbb{R}^{J_1 \times J_2 \dots \times J_N}$:

$$\mathcal{Y}_{j_1 j_2 \dots j_N} = \mathcal{X}_{(s_1 + j_1 \cdot t_1)(s_2 + j_2 \cdot t_2)\dots(s_N + j_N \cdot t_N)}$$

where $j_k \in \{0, 1, \dots, \lfloor \frac{e_k - s_k - 1}{t_k} \rfloor\}$.

### TensorFlow Implementation
```python
# Slicing a sub-tensor using standard Python slicing notation
Y = X[s_1:e_1:t_1, s_2:e_2:t_2]
```

---

## 7. Tensor Squeezing

Squeezing eliminates singleton dimensions (axes of size 1) without mutating internal value order.

### Mathematical Definition
Let $\mathcal{X} \in \mathbb{R}^{I_1 \times \dots \times I_N}$ where a subset of axes $A$ satisfies $I_a = 1, \forall a \in A$. If $\phi: \{1, \dots, M\} \to \{1, \dots, N\} \setminus A$ is a strictly increasing monotonic index mapping:

$$\mathcal{Y}_{j_1 j_2 \dots j_M} = \mathcal{X}_{i_1 i_2 \dots i_N} \quad \text{where } i_k = \begin{cases} 1 & \text{if } k \in A \\ j_{\phi^{-1}(k)} & \text{if } k \notin A \end{cases}$$

### TensorFlow Implementation
```python
# Removes all dimensions of size 1
Y = tf.squeeze(X, axis=list(A))
```

---

## 8. Tensor Aggregation

Advanced grouping transformations over segmented regions or cluster mappings.

### Mathematical Definition
For a matrix $\mathcal{X} \in \mathbb{R}^{I \times J}$ and a segment assignment vector $S \in \mathbb{N}^I$, the segmented aggregation mapping to $\mathcal{Y} \in \mathbb{R}^{K \times J}$ evaluates via Kronecker deltas:

$$\mathcal{Y}_{k, j} = \sum_{i=1}^{I} \mathcal{X}_{i, j} \cdot \delta_{k, S_i}$$

### TensorFlow Implementation
```python
Y = tf.math.segment_sum(X, segment_ids=S)
```

---

## 9. One-Hot Encoding

Expands a discrete label tensor by adding a categorical indicator axis of depth $D$.

### Mathematical Definition
Given $\mathcal{X} \in \mathbb{N}^{I_1 \times \dots \times I_N}$, active value $\alpha$, and inactive value $\beta$, the expanded encoding $\mathcal{Y} \in \mathbb{R}^{I_1 \times \dots \times I_N \times D}$ along target index $d \in \{0, \dots, D-1\}$ is:

$$\mathcal{Y}_{i_1 i_2 \dots i_N d} = \alpha \cdot \delta_{d, \, \mathcal{X}_{i_1 i_2 \dots i_N}} + \beta \cdot \left(1 - \delta_{d, \, \mathcal{X}_{i_1 i_2 \dots i_N}}\right)$$

### TensorFlow Implementation
```python
Y = tf.one_hot(X, depth=D, on_value=alpha, off_value=beta)
```

---

## 10. Tensor Gathering and Scattering

Indexed read and write operations mapping sparse index topologies to dense configurations.

### Mathematical Definition

#### Multi-Dimensional Gather (`tf.gather_nd`)
Extracts indices from source $\mathcal{X} \in \mathbb{R}^{I_0 \dots \times I_{N-1}}$ using coordinate lookup map $\mathcal{R} \in \mathbb{N}^{J_0 \dots \times J_{M-1} \times K}$:

$$\mathcal{Y}_{j_0 \dots j_{M-1} \, i_K \dots i_{N-1}} = \mathcal{X}_{\mathcal{R}_{j_0 \dots j_{M-1} 0}, \, \dots, \, \mathcal{R}_{j_0 \dots j_{M-1} (K-1)}, \, i_K, \, \dots, \, i_{N-1}}$$

#### Multi-Dimensional Scatter Update (`tf.tensor_scatter_nd_add`)
Accumulates updates $\mathcal{U}$ into base tensor $\mathcal{X}$ along target indices $\mathcal{R}$:

$$\mathcal{Y}_{i_0 \dots i_{N-1}} = \mathcal{X}_{i_0 \dots i_{N-1}} + \sum_{j_0 \dots j_{M-1}} \mathcal{U}_{j_0 \dots j_{M-1} \, i_K \dots i_{N-1}} \cdot \prod_{k=0}^{K-1} \delta_{i_k, \, \mathcal{R}_{j_0 \dots j_{M-1} k}}$$

### TensorFlow Implementation
```python
# Gather multi-dimensional slices
Y_gather = tf.gather_nd(X, R)

# Scatter multi-dimensional updates into an existing tensor
Y_scatter = tf.tensor_scatter_nd_add(X, R, U)
```

---

## 11. Relational $\theta$-Joins Over Tensors

Mimics relational database conditional evaluation across two separate tensors while combining their coordinate spaces.

### Mathematical Definition
Given $\mathcal{A} \in \mathbb{R}^{I_1 \times \dots \times I_L}$ with join axis index $\mu$, and $\mathcal{B} \in \mathbb{R}^{J_1 \times \dots \times J_R}$ with join axis index $\nu$:

1. **Broadcast Predicate Matrix Calculation ($\mathcal{M}$):**
$$\mathcal{M}_{i_1 \dots i_L j_1 \dots j_R} = \mathbb{I}\Big(\mathcal{A}_{i_1 \dots i_mu \dots i_L} \,\, \theta \,\, \mathcal{B}_{j_1 \dots j_\nu \dots j_R}\Big)$$

2. **Coordinate Extraction via Index Lookup Matrix ($\mathcal{R}$):**
$$\mathcal{R} = \text{positions}(\mathcal{M} == 1) \in \mathbb{N}^{K \times (L+R)}$$

3. **Value Materialization:**
$$\mathcal{Y}_{k, 0} = \mathcal{A}_{\mathcal{R}_{k, 1}, \dots, \mathcal{R}_{k, L}}, \quad \mathcal{Y}_{k, 1} = \mathcal{B}_{\mathcal{R}_{k, L+1}, \dots, \mathcal{R}_{k, L+R}}$$

### TensorFlow Implementation
```python
# Step 1: Reshape to broadcast comparison over targeted axes
# Assumes A is 2D [I, M_axis] and B is 2D [J, N_axis]
A_expanded = tf.expand_dims(tf.expand_dims(A, 1), 2) # Shape: [I, 1, 1, M_axis]
B_expanded = tf.expand_dims(tf.expand_dims(B, 0), 0) # Shape: [1, 1, J, N_axis]

# Compute boolean predicate indicator mask (e.g., theta is strict equality)
M_mask = tf.reduce_all(tf.equal(A_expanded, B_expanded), axis=-1)

# Step 2 & 3: Find valid matching coordinates and materialize entries

R_coords = tf.where(M_mask)
vals_A = tf.gather_nd(A, R_coords[:, :2])
vals_B = tf.gather_nd(B, R_coords[:, 2:])
Y_join = tf.concat([vals_A, vals_B], axis=-1)
```

## 12. Complex Contextual Multi-Axis Tensor Expressions

This section demonstrates how to formulate complex, arbitrary multi-axis expressions combining generalized Einstein contractions, cell-wise logical operators ($\vee$), and existential quantifiers (∃) acting over explicit **sets of axes**.

We formalize the evaluation of a target indicator expression:
$$\mathbb{I}(\exists I. \theta(\sum_J M[J;I]))$$
where I and J are distinct index tuples (sets of axes) rather than singular dimensions.

### Mathematical Definition

Given a tensor $\mathcal{M}$ spanning multi-index layout spaces $\mathbf{j} = (j_1, \dots, j_m) \in J$ and $\mathbf{i} = (i_1, \dots, i_n) \in I$:

1. **Multi-Index Contraction over Set J:**
$$\mathcal{Z}_{\mathbf{i}} = \sum_{j_1} \dots \sum_{j_m} \mathcal{M}_{(j_1, \dots, j_m, \, i_1, \dots, i_n)}$$

2. **Existential Quantifier Disjunction over Set I via Predicate θ:**
$$y = \mathbb{I}\left(\exists \mathbf{i} \in I . \, \theta(\mathcal{Z}_{\mathbf{i}})\right) = \bigvee_{i_1} \dots \bigvee_{i_n} \mathbb{I}\big(\theta(\mathcal{Z}_{(i_1, \dots, i_n)})\big)$$

Alternatively, using an algebraic bounding framework without explicit boolean branching logic:
$$y = \text{clip}\left( \sum_{i_1} \dots \sum_{i_n} \mathbb{I}\big(\theta(\mathcal{Z}_{\mathbf{i}})\big), \,\, 0, \,\, 1 \right)$$

For cell-wise logical OR ($\vee$) statements matching conditions such as $\mathcal{A}_{ijk} > 0 \vee \mathcal{B}_{ijk} > 0$, the indicator evaluates as:
$$\text{Logical\_OR}(\mathcal{A}, \mathcal{B}) = \max\Big( \mathbb{I}(\mathcal{A} > 0), \, \mathbb{I}(\mathcal{B} > 0) \Big)$$

### TensorFlow Implementation
```python
import tensorflow as tf

# Define numerical axis positions corresponding to index sets J and I
# Example assuming M is a 5D tensor: J maps to axes, I maps to axes [2, 3, 4]
axes_J = [0, 1]
axes_I = [2, 3, 4]

# 1. Compute multi-axis contraction over the designated axis indices in set J
Z = tf.reduce_sum(M, axis=axes_J)

# 2. Evaluate cell-wise condition theta (e.g., matching a numerical boundary)
# Includes element-wise logical OR implementation logic example via '|'
condition_mask = (Z > 0.0) | (Z < -1.0)

# 3. Handle Existential Quantifier ($\exists$ I) collapsing multi-axis set I
exists_condition = tf.reduce_any(condition_mask, axis=axes_I)

# 4. Final step conversion back to a numeric indicator configuration
y_final = tf.cast(exists_condition, dtype=tf.float32)
```
