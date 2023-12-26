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
- The __autograd engine__ is in [tensor.py](src/tensor.py).
- The deep learning model __layers__ are in [nn.py](src/nn.py).
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
- `src/` : Folder with python files.
  - `src/tensor.py`:  File with the `Tensor` class and all of the tensor `Operations`.
  - `src/nn.py`: Most deep learning layers, and `nn.Module` class.
  - `src/framework.py`: File with operations and helper functions.
  - `src/test_framework.py` : File with unit tests.
  - `src/optim.py` : File with optimizers.
- `data/` : Folder to store the text file used to test the Transformer. Currently holds `shakespeare.txt`.
    
## 2. Running it Yourself
<details>
<summary> <h3> Requirements </h3> </summary>
  
- The required packages are listed in `requirements.txt`.
- The requirements can be installed on a virtual environment with the command:
```
pip install -r requirements.txt
```
> **Note:** The framework is built around numpy, so there is no CUDA availability.


</details>
<details>
<summary> <h3> Build a Custom Model </h3> </summary>
  
- To create a custom model class, you can use the exact same syntax as you would in PyTorch, inheriting from nn.Module.
<details>
<summary> You may chose among <b>the following layers</b>: </summary>
      
    - `nn.Embedding` (first layer, turns input indexes into vectors)
    - `nn.PositionalEmbedding` (second layer, adds position information to every timestep of the input)
    - `nn.Linear` (simple fully-connected layer)
    - `nn.MultiHeadSelfAttention` (core of the transformer, calculates weighted sum of inputs)
    - `nn.RNN` (Recurrent Neural Network layer)
    - `nn.Block` (full transformer block - connects MHSA and Dense layers with residuals and LayerNorm)
    - `nn.CrossEntropyLoss` (last layer, returns probabilities for next generated character)


</details>
<details>
<summary> And <b>the following functions</b>: </summary>
      
    - `nn.Dropout` (can be added to apply dropout)
    - `nn.LayerNorm` (normalizes the tensors)
    - `nn.Softmax` (scales the values between 0 and 1)
    - `nn.Tanh` (scales the values between -1 and 1)
    - `nn.Relu` (zeroes all negative values)


</details>
</details>

## 3. Results
- The models implemented in [test_framework.py](src/test_framework.py) all converged to __near-zero losses__.
- This framework is not as fast or as optimized as PyTorch, but I tried making it more interpretable.
- Hope you enjoy!

