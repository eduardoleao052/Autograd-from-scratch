<p align="left">
    <a href="https://github.com/eduardoleao052/autograd-from-scratch/actions/workflows/test.yml/badge.svg" alt="Unit Tests">
        <img src="https://github.com/eduardoleao052/autograd-from-scratch/actions/workflows/test.yml/badge.svg" /></a>
    <a href="https://github.com/eduardoleao052/autograd-from-scratch/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/eduardoleao052/autograd-from-scratch" /></a>
    <a href="https://github.com/eduardoleao052/autograd-from-scratch/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/eduardoleao052/autograd-from-scratch" /></a>
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/language-Python-blue">
    </a>
    <a href="mailto:eduardoleao052@usp.br">
        <img src="https://img.shields.io/badge/-Email-red?style=flat-square&logo=gmail&logoColor=white">
    </a>
    <a href="https://www.linkedin.com/in/eduardoleao052/">
        <img src="https://img.shields.io/badge/-Linkedin-blue?style=flat-square&logo=linkedin">
    </a>
</p>


# Autograd Framework From Scratch
- NeuralForge is a unit-tested and documented educational framework. Similar to PyTorch, but with more __clear code__.
- The autograd from scratch engine is in [tensor_operations.py](neuralforge/tensor_operations.py). I got a lot of inspiration from Andrej Karpathy's micrograd videos.
- The deep learning model layers are in [nn/layers.py](neuralforge/nn/layers.py).
<br/>
<details>
<summary> Check out the <b>implemented basic operations</b>: </summary>


<br/>


- [Addition](https://github.com/eduardoleao052/Autograd-from-scratch/blob/97b5d4e9d9c118375e53699043556e4d68d7fce7/neuralforge/tensor_operations.py#L205-L257)
- [Subtraction](https://github.com/eduardoleao052/Autograd-from-scratch/blob/97b5d4e9d9c118375e53699043556e4d68d7fce7/neuralforge/tensor_operations.py#L259-L286)
- [Multiplication](https://github.com/eduardoleao052/Autograd-from-scratch/blob/97b5d4e9d9c118375e53699043556e4d68d7fce7/neuralforge/tensor_operations.py#L288-L342)
- [Division](https://github.com/eduardoleao052/Autograd-from-scratch/blob/c8c9b697815bc2c9efb1e9ce4d9ee490b43f19a2/neuralforge/tensor_operations.py#L344-L398)
- [Matrix multiplication](https://github.com/eduardoleao052/Autograd-from-scratch/blob/c8c9b697815bc2c9efb1e9ce4d9ee490b43f19a2/neuralforge/tensor_operations.py#L400-L451)
- [Exponentiation](https://github.com/eduardoleao052/Autograd-from-scratch/blob/c8c9b697815bc2c9efb1e9ce4d9ee490b43f19a2/neuralforge/tensor_operations.py#L582-L609)
- [Log](https://github.com/eduardoleao052/Autograd-from-scratch/blob/c8c9b697815bc2c9efb1e9ce4d9ee490b43f19a2/neuralforge/tensor_operations.py#L611-L638)
- [Square Root](https://github.com/eduardoleao052/Autograd-from-scratch/blob/c8c9b697815bc2c9efb1e9ce4d9ee490b43f19a2/neuralforge/tensor_operations.py#L640-L667)

<br/>
  
</details>


<details>
<summary> The <b>implemented statistics</b>: </summary>


<br/>


- [Sum](https://github.com/eduardoleao052/Autograd-from-scratch/blob/c8c9b697815bc2c9efb1e9ce4d9ee490b43f19a2/neuralforge/tensor_operations.py#L492-L519)
- [Mean](https://github.com/eduardoleao052/Autograd-from-scratch/blob/c8c9b697815bc2c9efb1e9ce4d9ee490b43f19a2/neuralforge/tensor_operations.py#L521-L549)
- [Max](https://github.com/eduardoleao052/Autograd-from-scratch/blob/c8c9b697815bc2c9efb1e9ce4d9ee490b43f19a2/neuralforge/tensor_operations.py#L454-L490)
- [Variance](https://github.com/eduardoleao052/Autograd-from-scratch/blob/c8c9b697815bc2c9efb1e9ce4d9ee490b43f19a2/neuralforge/tensor_operations.py#L551-L579)

<br/>

</details>


<details>
<summary> And the <b>implemented tensor operations</b>: </summary>


<br/>


- [Reshape](https://github.com/eduardoleao052/Autograd-from-scratch/blob/4b7083149a8dd8e9bdb2b0c93fe130d9be516bf0/neuralforge/tensor_operations.py#L682-L710)
- [Transpose](https://github.com/eduardoleao052/Autograd-from-scratch/blob/4b7083149a8dd8e9bdb2b0c93fe130d9be516bf0/neuralforge/tensor_operations.py#L713-L741)
- [Concatenate](https://github.com/eduardoleao052/Autograd-from-scratch/blob/4b7083149a8dd8e9bdb2b0c93fe130d9be516bf0/neuralforge/tensor_operations.py#L744-L780)
- [Stack](https://github.com/eduardoleao052/Autograd-from-scratch/blob/4b7083149a8dd8e9bdb2b0c93fe130d9be516bf0/neuralforge/tensor_operations.py#L783-L820)
- [MaskedFill](https://github.com/eduardoleao052/Autograd-from-scratch/blob/4b7083149a8dd8e9bdb2b0c93fe130d9be516bf0/neuralforge/tensor_operations.py#L823-L851)
- [Slice](https://github.com/eduardoleao052/Autograd-from-scratch/blob/4b7083149a8dd8e9bdb2b0c93fe130d9be516bf0/neuralforge/tensor_operations.py#L854-L882)

<br/>


</details>
<br/>


## 1. Project Structure
- `neuralforge/` : Framework with python files.
  - `neuralforge/tensor_operations.py`:  File with the `Tensor` class and all of the tensor `Operations`.
  - `neuralforge/utils.py`: File with operations and helper functions.
  - `neuralforge/nn/`: Submodule of the framework. Contains full layers and optimizers.
      - `neuralforge/nn/nn.py`: Most deep learning layers, and `nn.Module` class.
      - `neuralforge/nn/optim.py` : File with optimizers.
- `data/` : Folder to store training data. Currently holds `shakespeare.txt`.
- `test/`: Folder with unit tests. Contains `test_framework.py`.
- `setup.py` : Setup file for the framework.
    
## 2. Running it Yourself
### Simple Autograd Example: 
```python
import neuralforge as forge

# Instantiate Tensors:
x = forge.randn((8,4,5))
w = forge.randn((8,5,4), requires_grad = True)
b = forge.randint((4), requires_grad = True)

# Make calculations:
out = x @ w
out += b

# Compute gradients on whole graph:
out.backward()

# Get gradients from specific Tensors:
print(w.grad)
print(b.grad)

```

### Complex Autograd Example (Transformer): 
```python
import neuralforge as forge
import neuralforge.nn as nn

# Implement Transformer class inheriting from forge.nn.Module:
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, n_timesteps: int, n_heads: int, p: float):
        super().__init__()
        # Instantiate Transformer's Layers:
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.PositionalEmbedding(n_timesteps, hidden_size)
        self.b1 = nn.Block(hidden_size, hidden_size, n_heads, n_timesteps, dropout_prob=p) 
        self.b2 = nn.Block(hidden_size, hidden_size, n_heads, n_timesteps, dropout_prob=p)
        self.ln = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        z = self.embed(x) + self.pos_embed(x)
        z = self.b1(z)
        z = self.b2(z)
        z = self.ln(z)
        z = self.linear(z)

        return z

# Get tiny Shakespeare test data:
text = load_text_data(f'{PATH}/data/shakespeare.txt')

# Create Transformer instance:
model = Transformer(vocab_size, hidden_size, n_timesteps, n_heads, dropout_p)

# Define loss function and optimizer:
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, reg=0)
        
# Training Loop:
for _ in range(n_iters):
    x, y = get_batch(test_data, n_timesteps, batch_size)

    z = model.forward(x)

    # Get loss:
    loss = loss_func(z, y)

    # Backpropagate the loss using forge.tensor's backward() method:
    loss.backward()

    # Update the weights:
    optimizer.step()

    # Reset the gradients to zero after each training step:
    optimizer.zero_grad()
```
> **Note:** You can install the framework locally with: `pip install neuralforge`
<details>
<summary> <b> Requirements </b> </summary>

<br/>
  
- The required packages are listed in `requirements.txt`.
- The requirements can be installed on a virtual environment with the command:
```
pip install -r requirements.txt
```
> **Note:** The framework is built around numpy, so there is no CUDA availability.

<br/>

</details>
<details>
<summary> <b> Build a Custom Model </b> </summary>

<br/>

- To create a custom model class, you can use the exact same syntax as you would in PyTorch, inheriting from nn.Module.
<details>
<summary> You may chose among <b>the following layers</b>: </summary>

<br/>

- [nn.Embedding](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L129-L146) (first layer, turns input indexes into vectors)
- [nn.PositionalEmbedding](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L149-L164) (second layer, adds position information to every timestep of the input)
- [nn.Linear](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L47-L64) (simple fully-connected layer)
- [nn.MultiHeadSelfAttention](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L67-L126) (core of the transformer, calculates weighted sum of inputs)
- [nn.Block](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L268-L287) (full transformer block - Contains MHSA, Linear and LayerNorm layers)
- [nn.CrossEntropyLoss](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L290-L320) (last layer, returns probabilities for next generated character)

</details>
<details>
<summary> And <b>the following functions</b>: </summary>

<br/>

- [nn.Dropout](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L167-L183) (can be added to apply dropout)
- [nn.LayerNorm](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L186-L201) (normalizes the tensors)
- [nn.Softmax](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L215-L229) (scales the values between 0 and 1)
- [nn.Tanh](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L232-L241) (scales the values between -1 and 1)
- [nn.Relu](https://github.com/eduardoleao052/Autograd-from-scratch/blob/e7569075cb3342300274839bcf4edd8ba19a1c08/neuralforge/nn/layers.py#L204-L212) (zeroes all negative values)

</details>

<br/>

</details>

## 3. Results
- The models implemented in [test_framework.py](tests/test_framework.py) all converged to __near-zero losses__.
- This framework is not as fast or as optimized as PyTorch, but I tried making it more interpretable.
- Hope you enjoy!

