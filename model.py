import numpy as np
import torch
import torch.nn as nn
import operator
from functools import reduce

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

import logging

log = logging.getLogger("root")


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        self.max_logvar = torch.tensor(
            -3 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device
        )
        self.min_logvar = torch.tensor(
            -7 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device
        )

        # Create or load networks
        self.networks = nn.ModuleList(
            [self.create_network(n) for n in range(self.num_nets)]
        ).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

    def get_output(self, output):
        """
        Argument:
          output: the raw output of a single ensemble member
        Return:
          mean and log variance
        """
        mean = output[:, 0 : self.state_dim]
        raw_v = output[:, self.state_dim :]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_loss(self, targ, mean, logvar):
        """
        Compute the negative log-likelihood loss for a Gaussian distribution.
        
        For a Gaussian N(mean, var), the negative log likelihood is:
        -log p(x|mean,var) = 0.5 * log(2π) + 0.5 * log(var) + 0.5 * (x-mean)^2 / var
        
        Arguments:
            targ: target values (actual next state deltas), shape [batch_size, state_dim]
            mean: predicted means, shape [batch_size, state_dim]
            logvar: predicted log variances, shape [batch_size, state_dim]
        
        Returns:
            scalar loss value (mean over batch and state dimensions)
        """
        # TODO: write your code here
        # Negative log likelihood loss for Gaussian distribution
        # -log p(targ | mean, var) = 0.5 * [log(var) + (targ - mean)^2 / var] + const
        
        # Convert logvar to var
        var = torch.exp(logvar)
        
        # Compute negative log likelihood
        # NLL = 0.5 * [log(var) + (targ - mean)^2 / var]
        # We can simplify: log(var) = logvar (already have this)
        nll = 0.5 * (logvar + (targ - mean) ** 2 / var)
        
        # Return mean loss over all dimensions
        return nll.mean()

    def create_network(self, n):
        layer_sizes = [
            self.state_dim + self.action_dim,
            HIDDEN1_UNITS,
            HIDDEN2_UNITS,
            HIDDEN3_UNITS,
        ]
        layers = reduce(
            operator.add,
            [
                [nn.Linear(a, b), nn.ReLU()]
                for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])
            ],
        )
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
        Argument:
          inputs: state and action inputs. Assumes that inputs are standardized.
          targets: resulting states (deltas)
        Return:
            List containing the average loss of all the networks at each train iteration

        """
        # TODO: write your code here
        # Algorithm 2: Training the Probabilistic Ensemble
        # For n in 1:N (for each network):
        #   Uniformly sample (with replacement) minibatch of size B from data
        #   Take a gradient step of the loss for sampled minibatch
        
        losses = []
        
        for itr in range(num_train_itrs):
            # Convert to tensors if needed
            if not torch.is_tensor(inputs):
                inputs_tensor = torch.tensor(inputs, device=self.device, dtype=torch.float)
            else:
                inputs_tensor = inputs
                
            if not torch.is_tensor(targets):
                targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float)
            else:
                targets_tensor = targets
            
            total_loss = 0.0
            
            # Train each network in the ensemble
            for net_idx in range(self.num_nets):
                # Uniformly sample (with replacement) a minibatch of size batch_size
                indices = np.random.choice(len(inputs), size=batch_size, replace=True)
                batch_inputs = inputs_tensor[indices]
                batch_targets = targets_tensor[indices]
                
                # Forward pass through the specific network
                output = self.networks[net_idx](batch_inputs)
                mean, logvar = self.get_output(output)
                
                # Compute loss
                loss = self.get_loss(batch_targets, mean, logvar)
                
                # Backward pass and optimization
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                total_loss += loss.item()
            
            # Average loss across all networks
            avg_loss = total_loss / self.num_nets
            losses.append(avg_loss)
            
            if itr % 10 == 0 or itr == num_train_itrs - 1:
                log.info(f"Iteration {itr}/{num_train_itrs}, Loss: {avg_loss:.4f}")
        
        return losses
