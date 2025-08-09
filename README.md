# Linear Regression from Scratch

A model that learns the relationship between input features
(house size, number of bedrooms, age) and output (price).

- **Model**: `y = w₁x₁ + w₂x₂ + ... + b` (linear combination)
- **Loss**: Mean Squared Error - how wrong your predictions are
- **Optimization**: Gradient descent - adjusting weights to reduce error

**Phase 1: Data Setup:**

```python
# Generate synthetic housing data or use Boston housing dataset
# Features: square footage, bedrooms, age, location score
# Target: price in thousands
```

**Phase 2: Manual Implementation:**

- Create weight matrix and bias term as PyTorch tensors
- Write forward pass function (matrix multiplication)
- Implement loss calculation manually
- Calculate gradients by hand using calculus

**Phase 3: Gradient Descent:**

- Update weights using gradients
- Implement training loop
- Track loss over iterations
- Visualize how the model learns

**Phase 4: PyTorch's Autograd:**

- Replace manual gradients with `loss.backward()`
- See how much easier automatic differentiation makes things
- Compare results with your manual implementation

**Phase 5: Enhancements:**

- Add data normalization
- Implement different learning rates
- Add L2 regularization to prevent overfitting
- Create train/validation splits
