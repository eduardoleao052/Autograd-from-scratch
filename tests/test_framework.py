import sys
sys.path.append('..')
import neuralforge as forge
import neuralforge.nn as nn
import unittest
import numpy as np
import os

class TestNeuralForge(unittest.TestCase):
    ''' This class tests the functionalities of the framework in three levels of complexity. '''

    def test_autograd(self):
        '''
        This function tests whether the loss converges to zero in a spelled-out forward
        propagation, with weights explicitly declared.
        '''
        # Define loss function as Cross Entropy Loss:
        loss_func = nn.CrossEntropyLoss()

        # Instantiate input and output:
        x = forge.randn((8,4,5))
        y = np.random.randint(0,50,(8,4))

        # Instantiate Neural Network's Layers:
        w1 = forge.tensor(np.random.randn(5,128) / np.sqrt(5), requires_grad=True) 
        relu1 = nn.ReLU()
        w2 = forge.tensor(np.random.randn(128,128) / np.sqrt(128), requires_grad=True)
        relu2 = nn.ReLU()
        w3 = forge.tensor(np.random.randn(128,50) / np.sqrt(128), requires_grad=True)

        # Training Loop:
        for _ in range(4000):
            z = x @ w1
            z = relu1(z)
            z = z @ w2
            z = relu2(z)
            z = z @ w3
            
            # Get loss:
            loss = loss_func(z, y)

            # Backpropagate the loss using neuralforge.tensor:
            loss.backward()

            # Update the weights:
            w1 = w1 - (w1.grad * 0.005) 
            w2 = w2 - (w2.grad * 0.005) 
            w3 = w3 - (w3.grad * 0.005) 

            # Reset the gradients to zero after each training step:
            loss.zero_grad_tree()
        assert loss._data < 3e-1, "Error: Loss is not converging to zero in autograd test."

    def test_module(self):
        '''
        This function tests if the loss converges to zero in a simple Neural Network
        (Fully-Connected, three layers, with ReLU non-linearities), which uses the 
        custom nn.Module superclass.
        '''

        # Implement dummy nn.Module class:
        class NeuralNet(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                # Instantiate Neural Network's Layers:
                self.w1 = nn.Linear(5,hidden_size) 
                self.relu1 = nn.ReLU()
                self.w2 = nn.Linear(hidden_size,hidden_size)
                self.relu2 = nn.ReLU()
                self.w3 = nn.Linear(hidden_size,50)

            def forward(self, x):
                z = self.w1(x)
                z = self.relu1(z)
                z = self.w2(z)
                z = self.relu2(z)
                z = self.w3(z)
                return z

        model = NeuralNet(32)

        # Define loss function and optimizer:
        loss_func = nn.CrossEntropyLoss()
        optimizer = nn.optim.Adam(model.parameters(), lr=0.01, reg=0)

        # Instantiate input and output:
        x = forge.randn((8,4,5))
        y = np.random.randint(0,50,(8,4))

        # Training Loop:
        for _ in range(1000):
            z = model.forward(x)
            
            # Get loss:
            loss = loss_func(z, y)

            # Backpropagate the loss using neuralforge.tensor's backward() method:
            loss.backward()

            # Update the weights:
            optimizer.step()
            
            # Reset the gradients to zero after each training step:
            optimizer.zero_grad()
        
        assert loss._data < 1e-2, "Error: Loss is not converging to zero in nn.Module test."

    def test_transformer(self):
        '''
        This function tests if the loss converges to zero overfitting a full transformer on the tiny Shakespeare
        dataset. The Transformer model also uses the custom nn.Module superclass.
        '''

        # Implement helper function to load text data (tiny Shakespeare):
        def load_text_data(file):
            with open(f'{file}', 'r',encoding='utf8') as file:
                text = file.read() 
            chars = list(set(text))
            vocab_size = len(chars)

            char_to_ix = { ch:i for i,ch in enumerate(chars) }
            ix_to_char = { i:ch for i,ch in enumerate(chars) }
            
            train_text = ''
            test_text = ''
            text_phrases = text.split('\n')
            p = 0.01 * 100
            for i in range(len(text_phrases)//1000):
                text_to_add = '.'.join(text_phrases[i * 1000: (i+1) * 1000])
                if i % 100 >= p:
                    train_text += text_to_add
                else:
                    test_text += text_to_add
            
            test_data = [char_to_ix[ch] for  ch in test_text]
            return test_data, ix_to_char, vocab_size

        # Implement helper function to get a batch of text:
        def get_batch(data:list, n_timesteps:int, batch_size:int) -> tuple:
            B, T = batch_size, n_timesteps 
            pointers = np.arange(B)
            input_idxs = np.stack([data[p : p + T] for p in pointers])
            target_idxs = np.stack([data[p+1: p+1 + T] for p in pointers])

            return forge.tensor(input_idxs), target_idxs

        # Implement dummy class inheriting from nn.Module:
        class Transformer(nn.Module):
            def __init__(self, vocab_size, hidden_size, n_timesteps, n_heads, p):
                super().__init__()
                # Instantiate Transformer's Layers:
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.pos_embed = nn.PositionalEmbedding(n_timesteps, hidden_size)
                self.b1 = nn.Block(hidden_size, hidden_size, n_heads, n_timesteps, dropout_prob=p) 
                self.b2 = nn.Block(hidden_size, hidden_size, n_heads, n_timesteps, dropout_prob=p)
                self.b3 = nn.Block(hidden_size, hidden_size, n_heads, n_timesteps, dropout_prob=p)
                self.ln = nn.LayerNorm(hidden_size)
                self.linear = nn.Linear(hidden_size, vocab_size)

            def forward(self, x):
                z = self.embed(x) + self.pos_embed(x)
                z = self.b1(z)
                z = self.b2(z)
                z = self.b3(z)
                z = self.ln(z)
                z = self.linear(z)

                return z
        
        # Declare variables and hyperparameters:
        n_iters = 150
        n_timesteps = 32
        hidden_size = 128
        batch_size = 4
        n_heads = 8
        dropout_p = 0

        # Get path to root of repository:
        PATH = '/'.join(os.getcwd().split('/')[:-1])

        # Get tiny Shakespeare test data:
        test_data, ix_to_char, vocab_size = load_text_data(f'{PATH}/data/shakespeare.txt')

        # Take small subset of the data to test wether the model converges:
        test_data = test_data[:128]

        # Create Transformer instance:
        model = Transformer(vocab_size, hidden_size, n_timesteps, n_heads, dropout_p)

        # Define loss function and optimizer:
        loss_func = nn.CrossEntropyLoss()
        optimizer = nn.optim.Adam(model.parameters(), lr=0.005, reg=0)
        
        # Training Loop:
        for _ in range(n_iters):
            x, y = get_batch(test_data, n_timesteps, batch_size)

            z = model.forward(x)

            # Get loss:
            loss = loss_func(z, y)

            # Backpropagate the loss using neuralforge.tensor's backward() method:
            loss.backward()

            # Update the weights:
            optimizer.step()

            # Reset the gradients to zero after each training step:
            optimizer.zero_grad()

        #print(sample(model, 1000, n_timesteps, vocab_size, ix_to_char))
        
        assert loss._data < 1, "Error: Loss is not converging to zero in autograd test."


if __name__ == '__main__':
    unittest.main()
