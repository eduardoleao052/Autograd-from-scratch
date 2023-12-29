<p align="left">
    <a href="https://github.com/eduardoleao052/autograd-from-scratch/actions/workflows/test.yml/badge.svg" alt="Unit Tests">
        <img src="https://github.com/eduardoleao052/autograd-from-scratch/actions/workflows/test.yml/badge.svg" /></a>
    <a href="https://github.com/eduardoleao052/Transformer-from-scratch/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/eduardoleao052/Transformer-from-scratch" /></a>
    <a href="https://github.com/eduardoleao052/Transformer-from-scratch/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/eduardoleao052/Transformer-from-scratch" /></a>
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/language-Python-blue">
    </a>
    <a href="mailto:eduardoleao052@usp.br">
        <img src="https://img.shields.io/badge/-Email-red?style=flat-square&logo=gmail&logoColor=white">
    </a>
    <a href=""https://www.linkedin.com/in/eduardoleao052/">
        <img src="https://img.shields.io/badge/-Linkedin-blue?style=flat-square&logo=linkedin">
    </a>
</p>


# Deep Learning Framework From Scratch
- Unit-tested and documented educational framework. Similar to PyTorch, but with simpler sintax.
- The __autograd engine__ is in [tensor_operations.py](neuralforge/tensor_operations.py). I got a lot of inspiration from Andrej Karpathy's __micrograd__ videos.
- The deep learning model __layers__ are in [nn.py](neuralforge/nn.py).
<br/>
<details>
<summary> Check out the <b>implemented basic operations</b>: </summary>


<br/>


- Addition
- Subtraction
- Multiplication
- Division
- Matrix multiplication
- Exponentiation
- Log
- Square Root

<br/>
  
</details>


<details>
<summary> The <b>implemented statistics</b>: </summary>


<br/>


- Sum
- Mean
- Max
- Variance

<br/>

</details>


<details>
<summary> And the <b>implemented tensor operations</b>: </summary>


<br/>


- Reshape
- Transpose
- Concatenate
- Stack
- MaskedFill
- Slice

<br/>


</details>
<br/>


## 1. Project Structure
- `neuralforge/` : Framework with python files.
  - `neuralforge/tensor_operations.py`:  File with the `Tensor` class and all of the tensor `Operations`.
  - `neuralforge/nn.py`: Most deep learning layers, and `nn.Module` class.
  - `neuralforge/utils.py`: File with operations and helper functions.
  - `neuralforge/test_framework.py`: File with unit tests.
  - `neuralforge/optim.py` : File with optimizers.
- `data/` : Folder to store the text file used to test the Transformer. Currently holds `shakespeare.txt`.
- `setup.py` : Setup file for the framework.
    
## 2. Running it Yourself
### Simple Autograd Example: 
```python
import neuralforge as forge

# Instantiate Tensors:
x = forge.randn((8,4,5), requires_grad = True)
w = forge.randn((8,5,4), requires_grad = True)
b = forge.randint((5), requires_grad = True)

# Make calculations:
x = x @ w
x += b

# Compute gradients on whole graph:
x.backward()

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

```
- nn.Embedding (first layer, turns input indexes into vectors)
- nn.PositionalEmbedding (second layer, adds position information to every timestep of the input)
- nn.Linear (simple fully-connected layer)
- nn.MultiHeadSelfAttention (core of the transformer, calculates weighted sum of inputs)
- nn.RNN (Recurrent Neural Network layer)
- nn.Block (full transformer block - connects MHSA and Dense layers with residuals and LayerNorm)
- nn.CrossEntropyLoss (last layer, returns probabilities for next generated character)
```

</details>
<details>
<summary> And <b>the following functions</b>: </summary>

```
- nn.Dropout (can be added to apply dropout)
- nn.LayerNorm (normalizes the tensors)
- nn.Softmax (scales the values between 0 and 1)
- nn.Tanh (scales the values between -1 and 1)
- nn.Relu (zeroes all negative values)
```

</details>

<br/>

</details>

## 3. Results
- The models implemented in [test_framework.py](src/test_framework.py) all converged to __near-zero losses__.
- This framework is not as fast or as optimized as PyTorch, but I tried making it more interpretable.
- Hope you enjoy!

