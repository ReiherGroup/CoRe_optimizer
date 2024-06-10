#!/usr/bin/python3

'''
Continual Resilient (CoRe) Optimizer
'''
__copyright__ = '''This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.'''

import unittest
import pytest
import torch
from core_optimizer import CoRe


class Model(torch.nn.Module):
    '''
    Model for tests
    '''

    def __init__(self) -> None:
        '''
        Initialization
        '''
        super().__init__()
        self.l1 = torch.nn.Linear(2, 20)
        self.l2 = torch.nn.Linear(20, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        Return: outputs
        '''
        outputs = self.l1(inputs)
        outputs = self.l2(outputs)
        return outputs


class CoReTest(unittest.TestCase):
    '''
    Continual Resilient (CoRe) optimizer tests
    '''
    torch.set_default_dtype(torch.float64)
    inputs = torch.ones((10, 2))
    inputs[:, 0] += torch.arange(10)
    inputs[:, 1] += torch.arange(10)
    labels = 2.0 * inputs
    labels[:, 1] *= 2.0
    loss_fn = torch.nn.modules.loss.MSELoss()
    model = Model()
    optimizer = CoRe(model.parameters())

    def training(self, sign: float = 1.0) -> float:
        '''
        Return: loss
        '''
        step = 0
        while step < 500:
            self.optimizer.zero_grad()
            for i in range(10):
                outputs = self.model(self.inputs[i])
                loss = sign * self.loss_fn(outputs, self.labels[i])
                loss.backward()
            self.optimizer.step()
            step += 1
        loss = 0.0
        for i in range(10):
            outputs = self.model(self.inputs[i])
            loss += self.loss_fn(outputs, self.labels[i]).detach().numpy()
        return float(loss)

    def test_default(self) -> None:
        '''
        Test of default settings
        '''
        loss_ref = 9.3937e-06
        for foreach in (None, False, True):
            torch.manual_seed(227)
            self.model = Model()
            self.optimizer = CoRe(self.model.parameters(), foreach=foreach)
            loss = self.training()
            assert pytest.approx(loss, rel=1e-5, abs=1e-8) == loss_ref, \
                f'ERROR: Loss is {loss} but it should be {loss_ref}.'
            torch.manual_seed(227)
            self.model = Model()
            self.optimizer = CoRe(self.model.parameters(), maximize=True, foreach=foreach)
            loss = self.training(sign=-1.0)
            assert pytest.approx(loss, rel=1e-5, abs=1e-8) == loss_ref, \
                f'ERROR: Loss is {loss} but it should be {loss_ref}.'

    def test_plasticity(self) -> None:
        '''
        Test with plasticity factors based on importance scores
        '''
        loss_ref = 9.2580e-06
        for foreach in (False, True):
            torch.manual_seed(227)
            self.model = Model()
            self.optimizer = CoRe(self.model.parameters(), score_history=250,
                                  frozen=[0.05, 0.0, 0.05, 0.0], foreach=foreach)
            loss = self.training()
            assert pytest.approx(loss, rel=1e-5, abs=1e-8) == loss_ref, \
                f'ERROR: Loss is {loss} but it should be {loss_ref}.'
            torch.manual_seed(227)
            self.model = Model()
            self.optimizer = CoRe(self.model.parameters(), score_history=250,
                                  frozen=[0.05, 0.0, 0.05, 0.0], maximize=True, foreach=foreach)
            loss = self.training(sign=-1.0)
            assert pytest.approx(loss, rel=1e-5, abs=1e-8) == loss_ref, \
                f'ERROR: Loss is {loss} but it should be {loss_ref}.'

    def test_no_adam(self) -> None:
        '''
        Test without Adam-like step size update
        '''
        loss_ref = 1.3110e-07
        for foreach in (False, True):
            torch.manual_seed(227)
            self.model = Model()
            self.optimizer = CoRe(self.model.parameters(), betas=(0.0, 0.0, 1.0, -1.0),
                                  foreach=foreach)
            loss = self.training()
            assert pytest.approx(loss, rel=1e-5, abs=1e-8) == loss_ref, \
                f'ERROR: Loss is {loss} but it should be {loss_ref}.'
            torch.manual_seed(227)
            self.model = Model()
            self.optimizer = CoRe(self.model.parameters(), betas=(0.0, 0.0, 1.0, -1.0),
                                  maximize=True, foreach=foreach)
            loss = self.training(sign=-1.0)
            assert pytest.approx(loss, rel=1e-5, abs=1e-8) == loss_ref, \
                f'ERROR: Loss is {loss} but it should be {loss_ref}.'

    def test_complex(self) -> None:
        '''
        Test with complex numbers
        '''
        params_real_ref = [1.00026, 3.02243]
        params_imag_ref = [2.00022, 3.97243]
        for foreach in (False, True):
            torch.manual_seed(227)
            params = torch.randn(2, dtype=torch.complex128, requires_grad=True)
            params_real = params.real.clone().detach().requires_grad_()
            params_imag = params.imag.clone().detach().requires_grad_()
            optimizer = CoRe([params], step_sizes=(1e-6, 1.0), weight_decay=0.0, foreach=foreach)
            optimizer_real = CoRe([params_real, params_imag], step_sizes=(1e-6, 1.0),
                                  weight_decay=0.0, foreach=foreach)
            step = 0
            while step < 100:
                optimizer.zero_grad()
                optimizer_real.zero_grad()
                loss = (
                    ((1.0 - params[0].real)**2 + (2.0 - params[0].real * params[0].imag)**2) + (
                        (3.0 - params[1].real)**2 + (12.0 - params[1].real * params[1].imag)**2))
                loss_real = (
                    ((1.0 - params_real[0])**2 + (2.0 - params_real[0] * params_imag[0])**2) + (
                        (3.0 - params_real[1])**2 + (12.0 - params_real[1] * params_imag[1])**2))
                loss.backward()
                loss_real.backward()
                optimizer.step()
                optimizer_real.step()
                step += 1
        params_real = params_real.detach().numpy()
        params_imag = params_imag.detach().numpy()
        for i in range(2):
            assert pytest.approx(params_real[i], rel=1e-5, abs=1e-8) == params_real_ref[i], \
                f'ERROR: Parameter is {params_real[i]} but it should be {params_real_ref[i]}.'
            assert pytest.approx(params_imag[i], rel=1e-5, abs=1e-8) == params_imag_ref[i], \
                f'ERROR: Parameter is {params_imag[i]} but it should be {params_imag_ref[i]}.'
        params_real = params.real.detach().numpy()
        params_imag = params.imag.detach().numpy()
        for i in range(2):
            assert pytest.approx(params_real[i], rel=1e-5, abs=1e-8) == params_real_ref[i], \
                f'ERROR: Parameter is {params_real[i]} but it should be {params_real_ref[i]}.'
            assert pytest.approx(params_imag[i], rel=1e-5, abs=1e-8) == params_imag_ref[i], \
                f'ERROR: Parameter is {params_imag[i]} but it should be {params_imag_ref[i]}.'

    def test_empty(self) -> None:
        '''
        Test with empty parameters
        '''
        for foreach in (False, True):
            params = torch.randn(0, requires_grad=True)
            optimizer = CoRe([params], foreach=foreach)
            optimizer.zero_grad()
            optimizer.step()
