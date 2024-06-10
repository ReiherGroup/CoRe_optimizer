#!/usr/bin/python3

'''
Continual Resilient (CoRe) Optimizer
'''
__copyright__ = '''This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.'''

from math import exp
from typing import List, Optional, Tuple, Union, Any, Dict, Iterable
from typing_extensions import TypeAlias

import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach, _foreach_doc,
    _maximize_doc)


ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


def _view_as_real(params, *state_and_grads):
    for i, p in enumerate(params):
        if torch.is_complex(p):
            params[i] = torch.view_as_real(params[i])
            for s in state_and_grads:
                s[i] = torch.view_as_real(s[i])


__all__ = ['CoRe', 'core']


class CoRe(Optimizer):
    '''
    Implements the Continual Resilient (CoRe) optimizer.
    '''
    def __init__(self,
                 params: ParamsT,
                 lr: float = 1e-3,
                 step_sizes: Tuple[float, float] = (1e-6, 1e-2),
                 etas: Tuple[float, float] = (0.7375, 1.2),
                 betas: Tuple[float, float, float, float] = (0.7375, 0.8125, 250.0, 0.99),
                 eps: float = 1e-8,
                 weight_decay: Union[float, list] = 0.1,
                 score_history: int = 0,
                 frozen: Union[float, list] = 0.0,
                 *,
                 maximize: bool = False,
                 foreach: Optional[bool] = None):
        if lr <= 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 < step_sizes[0] <= step_sizes[1]:
            raise ValueError(f'Invalid min and max step sizes: {step_sizes[0]}, {step_sizes[1]}')
        if not 0.0 < etas[0] <= 1.0 <= etas[1]:
            raise ValueError(f'Invalid eta values: {etas[0]}, {etas[1]}')
        if (
            not 0.0 <= betas[0] < 1.0
            or not 0.0 <= betas[1] < 1.0
            or betas[2] <= 0.0
            or betas[3] >= 1.0
        ):
            raise ValueError(
                f'Invalid beta values: {betas[0]}, {betas[1]}, {betas[2]}, {betas[3]}')
        if eps < 0.0:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if isinstance(weight_decay, float):
            if weight_decay < 0.0:
                raise ValueError(f'Invalid weight decay: {weight_decay}')
            weight_decay = [weight_decay]
        elif not isinstance(weight_decay, list):
            raise ValueError(f'Invalid weight decay: {weight_decay}')
        if score_history < 0:
            raise ValueError(f'Invalid score history: {score_history}')
        if isinstance(frozen, float):
            if not 0.0 <= frozen <= 1.0:
                raise ValueError(f'Invalid frozen: {frozen}')
            frozen = [frozen]
        elif not isinstance(frozen, list):
            raise ValueError(f'Invalid frozen: {frozen}')

        defaults = {'lr': lr,
                    'step_sizes': step_sizes,
                    'etas': etas,
                    'betas': betas,
                    'eps': eps,
                    'weight_decay': weight_decay,
                    'score_history': score_history,
                    'frozen': frozen,
                    'maximize': maximize,
                    'foreach': foreach,
                    'differentiable': False}
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self,
                    group,
                    params,
                    grads,
                    prevs_1,
                    prevs_2,
                    step_sizes,
                    scores,
                    frozens,
                    steps):
        has_complex = False
        i = -1
        for p in group['params']:
            i += 1
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params.append(p)
            grad = p.grad
            assert not grad.is_sparse, 'CoRe does not support sparse gradients'

            grads.append(grad)
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['prev_1'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['prev_2'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if p.dtype.is_complex:
                    # Complex number should be as if they are two independent real numbers.
                    # Hence the step_size should not be zero for imaginary part.
                    state['step_size'] = grad.new().resize_as_(grad).fill_(
                        complex(group['lr'], group['lr']))
                    state['score'] = grad.new().resize_as_(grad).fill_(
                        complex(0.0, 0.0))
                else:
                    state['step_size'] = grad.new().resize_as_(grad).fill_(group['lr'])
                    state['score'] = grad.new().resize_as_(grad).fill_(0.0)
                state['frozen'] = int(group['frozen'][i % len(group['frozen'])]
                                      * torch.prod(torch.tensor(grad.size())).item())

            prevs_1.append(state['prev_1'])
            prevs_2.append(state['prev_2'])
            step_sizes.append(state['step_size'])
            scores.append(state['score'])
            steps.append(state['step'])
            frozens.append(state['frozen'])

            state['step'] += 1
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        '''
        Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        '''
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            prevs_1 = []
            prevs_2 = []
            step_sizes = []
            scores = []
            frozens = []
            steps = []
            step_size_min, step_size_max = group['step_sizes']
            eta_minus, eta_plus = group['etas']
            beta_1_initial, beta_1_final, beta_1_step, beta_2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            score_history = group['score_history']
            maximize = group['maximize']
            foreach = group['foreach']
            differentiable = group['differentiable']

            has_complex = self._init_group(group,
                                           params,
                                           grads,
                                           prevs_1,
                                           prevs_2,
                                           step_sizes,
                                           scores,
                                           frozens,
                                           steps)

            core(params,
                 grads,
                 prevs_1,
                 prevs_2,
                 step_sizes,
                 step_size_min=step_size_min,
                 step_size_max=step_size_max,
                 eta_minus=eta_minus,
                 eta_plus=eta_plus,
                 beta_1_initial=beta_1_initial,
                 beta_1_final=beta_1_final,
                 beta_1_step=beta_1_step,
                 beta_2=beta_2,
                 eps=eps,
                 weight_decay=weight_decay,
                 scores=scores,
                 score_history=score_history,
                 frozens=frozens,
                 steps=steps,
                 maximize=maximize,
                 foreach=foreach,
                 differentiable=differentiable,
                 has_complex=has_complex)

        return loss


CoRe.__doc__ = r'''Implements the Continual Resilient (CoRe) optimizer.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt} \\
            &\textbf{input} : \theta\ \text{(params)},\ f(\theta)\ \text{(objective)},
                              \ s_\mathrm{min},s_\mathrm{max}\ \text{(step sizes)}, \\
            &\hspace{13mm} \eta_-,\eta_+\ \text{(etas)},
                           \ \beta_1^\mathrm{a},\beta_1^\mathrm{b},\beta_1^\mathrm{c},\beta_2
                           \ \text{(betas)},\ \epsilon\ \text{(eps)}, \\
            &\hspace{13mm} d\ \text{(weight decay)},\ t_\mathrm{hist}\ \text{(score history)},
                           \ p_\mathrm{frozen}\ \text{(frozen)} \\
            &\textbf{initialize} : s_0 \leftarrow \text{lr},\ g_0 \leftarrow 0,\ h_0 \leftarrow 0,\ S_0 \leftarrow 0 \\
            &\rule{110mm}{0.4pt} \\
            &\textbf{for}\ t=1\ \textbf{to}\ \ldots\ \textbf{do} \\
            &\hspace{5mm} G_t \leftarrow \nabla_{\theta}f_t(\theta_{t-1})\\
            &\hspace{5mm} \textbf{if}\ \text{maximize} \\
            &\hspace{10mm} G_t \leftarrow -G_t\\
            &\hspace{5mm} \beta_{1,t} \leftarrow \beta_1^\mathrm{b} + (\beta_1^\mathrm{a}-\beta_1^\mathrm{b})\,
                          \mathrm{exp}\{-[(t-1)/\beta_1^\mathrm{c}]^2\}\\
            &\hspace{5mm} g_t \leftarrow \beta_{1,t} g_{t-1} + (1-\beta_{1,t}) G_t \\
            &\hspace{5mm} \textbf{if}\ \beta_2 \geq 0 \\
            &\hspace{10mm} h_t \leftarrow \beta_2 h_{t-1} + (1-\beta_2) G_t^2 \\
            &\hspace{5mm} P_t \leftarrow 1 \\
            &\hspace{5mm} \textbf{if}\ t_\mathrm{hist}>0 \land t>t_\mathrm{hist}
                          \land S_{t-1}\ \mathrm{top}\text{-}p_\mathrm{frozen}\ \mathrm{in}\ \mathbf{S}_{t-1}\\
            &\hspace{10mm} P_t \leftarrow 0 \\
            &\hspace{5mm} \textbf{if}\ g_{t-1} g_t P_t > 0 \\
            &\hspace{10mm} s_t \leftarrow \mathrm{min}(\eta_+ s_{t-1}, s_\mathrm{max}) \\
            &\hspace{5mm} \textbf{else if}\ g_{t-1} g_t P_t < 0 \\
            &\hspace{10mm} s_t \leftarrow \mathrm{max}(\eta_- s_{t-1}, s_\mathrm{min}) \\
            &\hspace{5mm} \textbf{else} \\
            &\hspace{10mm} s_t \leftarrow s_{t-1} \\
            &\hspace{5mm} \textbf{if}\ \beta_2 \geq 0 \\
            &\hspace{10mm} u_t \leftarrow g_t / (1-\beta_{1,t}^t) / \{[h_t/(1-\beta_2^t)]^{0.5}+\epsilon\} \\
            &\hspace{5mm} \textbf{else} \\
            &\hspace{10mm} u_t \leftarrow \mathrm{sgn}(g_t) \\
            &\hspace{5mm} \textbf{if}\ t_\mathrm{hist} > 0\\
            &\hspace{10mm} \textbf{if}\ t \leq t_\mathrm{hist}\\
            &\hspace{15mm} S_t \leftarrow S_{t-1} + t_\mathrm{hist}^{-1} g_t u_t P_t s_t\\
            &\hspace{10mm} \textbf{else} \\
            &\hspace{15mm} S_t \leftarrow (1-t_\mathrm{hist}^{-1}) S_\xi^{\tau-1}
                           + t_\mathrm{hist}^{-1} g_t u_t P_t s_t\\
            &\hspace{5mm}\theta_t \leftarrow (1-d|u_t|P_ts_t) \theta_{t-1} - u_t P_t s_t \\
            &\rule{110mm}{0.4pt} \\[-1.ex]
            &\bf{return}\ \theta_t \\[-1.ex]
            &\rule{110mm}{0.4pt} \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to the papers
    `Lifelong Machine Learning Potentials <https://doi.org/10.1021/acs.jctc.3c00279>`_
    and `CoRe Optimizer: An All-in-One Solution for Machine Learning <https://arxiv.org/abs/2307.15663>`_.
    ''' + fr'''
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate to set initial step size (default:
            1e-3)
        step_sizes (Tuple[float, float], optional): pair of minimal and maximal
            allowed step sizes (recommendation: maximal step size of 1e-3 for
            mini-batch learning, 1.0 for batch learning, and 1e-2 for
            intermediate cases) (default: (1e-6, 1e-2))
        etas (Tuple[float, float], optional): pair of etaminus and etaplus
            that are multiplicative increase and decrease factors (default:
            (0.7375, 1.2))
        betas (Tuple[float, float, float, float], optional): coefficients
            beta1a, beta1b, beta1c, and beta2 used for computing running
            averages of gradient and its square (default: (0.7375, 0.8125,
            250.0, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float or List[float], optional): weight decay for all
            parameters or list of weight decays for parameter groups
            (default: 0.1)
        score_history (int, optional): number of optimization steps to build
            the score history before applying plasticity factors (default: 0)
        frozen (float or List[float], optional): fraction of all parameters
            frozen by the plasticity factors or list of fractions for parameter
            groups (applies if score_history > 0) (default: 0.0)
        {_maximize_doc}
        {_foreach_doc}
    '''


def core(params: List[Tensor],
         grads: List[Tensor],
         prevs_1: List[Tensor],
         prevs_2: List[Tensor],
         step_sizes: List[Tensor],
         maximize: bool = False,
         foreach: Optional[bool] = None,
         differentiable: bool = False,
         has_complex: bool = False,
         *,
         step_size_min: float,
         step_size_max: float,
         eta_minus: float,
         eta_plus: float,
         beta_1_initial: float,
         beta_1_final: float,
         beta_1_step: float,
         beta_2: float,
         eps: float,
         weight_decay: List[float],
         scores: List[Tensor],
         score_history: int,
         frozens: List[int],
         steps: List[int]):
    r'''
    Functional API that performs core algorithm computation.

    See :class:`~core_optimizer.CoRe` for details.
    '''

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

    assert not (foreach and torch.jit.is_scripting()), \
        'torch.jit.script not supported with foreach optimizers'

    if foreach:
        func = _multi_tensor_core
    else:
        func = _single_tensor_core

    func(params,
         grads,
         prevs_1,
         prevs_2,
         step_sizes,
         step_size_min=step_size_min,
         step_size_max=step_size_max,
         eta_minus=eta_minus,
         eta_plus=eta_plus,
         beta_1_initial=beta_1_initial,
         beta_1_final=beta_1_final,
         beta_1_step=beta_1_step,
         beta_2=beta_2,
         eps=eps,
         weight_decay=weight_decay,
         scores=scores,
         score_history=score_history,
         frozens=frozens,
         steps=steps,
         maximize=maximize,
         differentiable=differentiable,
         has_complex=has_complex)


def _single_tensor_core(params: List[Tensor],
                        grads: List[Tensor],
                        prevs_1: List[Tensor],
                        prevs_2: List[Tensor],
                        step_sizes: List[Tensor],
                        *,
                        step_size_min: float,
                        step_size_max: float,
                        eta_minus: float,
                        eta_plus: float,
                        beta_1_initial: float,
                        beta_1_final: float,
                        beta_1_step: float,
                        beta_2: float,
                        eps: float,
                        weight_decay: List[float],
                        scores: List[Tensor],
                        score_history: int,
                        frozens: List[int],
                        steps: List[int],
                        maximize: bool,
                        differentiable: bool,
                        has_complex: bool):

    # get properties
    n_weight_decay = len(weight_decay)
    for i, param in enumerate(params):
        grad = grads[i]
        prev_1 = prevs_1[i]
        prev_2 = prevs_2[i]
        step_size = step_sizes[i]
        score = scores[i]
        step = steps[i]

        # handle complex params
        if has_complex:
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            prev_1 = torch.view_as_real(prev_1)
            prev_2 = torch.view_as_real(prev_2)
            step_size = torch.view_as_real(step_size)
            score = torch.view_as_real(score)

        # for maximization invert gradient signs
        if maximize:
            grad = -grad

        # exponential moving average of squared gradients
        if beta_2 >= 0:
            prev_2.mul_(beta_2)
            prev_2.addcmul_(grad, grad, value=1.0 - beta_2)

        # adjust fractions of previous gradient information in current gradients
        beta_1 = (beta_1_final + (beta_1_initial - beta_1_final)
                  * exp(-(step / beta_1_step)**2))

        # exponential moving average of gradients
        grad = grad.clone(memory_format=torch.preserve_format)
        grad.lerp_(prev_1, beta_1)

        # stability-plasticity balance
        if score_history > 0:
            plasticity = score.new().resize_as_(score).fill_(1.0)
            if step >= score_history:
                if frozens[i] > 0:
                    score_max = float(torch.topk(torch.flatten(score), frozens[i])
                                      .values[frozens[i] - 1])
                    plasticity[score.ge(0.999999 * score_max)] = 0
                    score.mul_(1.0 - 1.0 / score_history)

        # determine step size updates
        step_size_update = grad.mul(prev_1)
        if score_history > 0:
            step_size_update.mul_(plasticity)
        step_size_update[step_size_update.gt(0)] = eta_plus
        step_size_update[step_size_update.lt(0)] = eta_minus
        step_size_update[step_size_update.eq(0)] = 1.0

        # update step sizes
        step_size.mul_(step_size_update)
        step_size.clamp_(step_size_min, step_size_max)

        # adjust parameter updates
        if beta_2 >= 0:
            param_update = torch.div(
                grad / (1.0 - beta_1**(step + 1)),
                torch.add(torch.sqrt(prev_2 / (1.0 - beta_2**(step + 1))), eps))
        else:
            param_update = grad.sign()
        param_update.mul_(step_size)
        if score_history > 0:
            param_update.mul_(plasticity)
            score.addcmul_(grad, param_update, value=1.0 / score_history)

        # weight decay
        param.add_(-weight_decay[i % n_weight_decay] * param_update.abs() * param)

        # update parameters
        param.add_(param_update, alpha=-1.0)

        # update previous gradients
        prev_1.copy_(grad)


def _multi_tensor_core(params: List[Tensor],
                       grads: List[Tensor],
                       prevs_1: List[Tensor],
                       prevs_2: List[Tensor],
                       step_sizes: List[Tensor],
                       *,
                       step_size_min: float,
                       step_size_max: float,
                       eta_minus: float,
                       eta_plus: float,
                       beta_1_initial: float,
                       beta_1_final: float,
                       beta_1_step: float,
                       beta_2: float,
                       eps: float,
                       weight_decay: List[float],
                       scores: List[Tensor],
                       score_history: int,
                       frozens: List[int],
                       steps: List[int],
                       maximize: bool,
                       differentiable: bool,
                       has_complex: bool):

    # check params
    if len(params) == 0:
        return

    # check differentiable
    assert not differentiable, '_foreach ops do not support autograd'

    # handle complex params
    if has_complex:
        _view_as_real(params, grads, prevs_1, prevs_2, step_sizes, scores)

    # for maximization invert gradient signs
    for i in range(len(grads)):
        grads[i] = grads[i].clone(memory_format=torch.preserve_format)
    if maximize:
        torch._foreach_neg_(grads)

    # exponential moving average of squared gradients
    if beta_2 >= 0:
        torch._foreach_mul_(prevs_2, beta_2)
        torch._foreach_addcmul_(prevs_2, grads, grads, value=1.0 - beta_2)

    # adjust fractions of previous gradient information in current gradients
    betas_1 = [grads[i].new().resize_as_(grads[i]).fill_(
        beta_1_final + (beta_1_initial - beta_1_final)
        * exp(-(steps[i] / beta_1_step)**2)) for i in range(len(steps))]

    # exponential moving average of gradients
    torch._foreach_lerp_(grads, prevs_1, betas_1)

    # stability-plasticity balance
    if score_history > 0:
        plasticities = [score.new().resize_as_(score).fill_(1.0) for score in scores]
        for i in range(len(scores)):
            if steps[i] >= score_history:
                if frozens[i] > 0:
                    score_max = float(torch.topk(torch.flatten(scores[i]), frozens[i])
                                      .values[frozens[i] - 1])
                    plasticities[i][scores[i].ge(0.999999 * score_max)] = 0
                    scores[i].mul_(1.0 - 1.0 / score_history)

    # determine step size updates
    step_size_updates = torch._foreach_mul(grads, prevs_1)
    if score_history > 0:
        torch._foreach_mul_(step_size_updates, plasticities)
    for step_size_update in step_size_updates:
        step_size_update[step_size_update.gt(0)] = eta_plus
        step_size_update[step_size_update.lt(0)] = eta_minus
        step_size_update[step_size_update.eq(0)] = 1.0

    # update step sizes
    torch._foreach_mul_(step_sizes, step_size_updates)
    for step_size in step_sizes:
        step_size.clamp_(step_size_min, step_size_max)

    # adjust parameter updates
    if beta_2 >= 0:
        param_updates = [
            torch.div(grads[i] / (1.0 - betas_1[i]**(steps[i] + 1)),
                      torch.add(torch.sqrt(prevs_2[i] / (1.0 - beta_2**(steps[i] + 1))), eps))
            for i in range(len(grads))]
    else:
        param_updates = [grad.sign() for grad in grads]
    torch._foreach_mul_(param_updates, step_sizes)
    if score_history > 0:
        torch._foreach_mul_(param_updates, plasticities)
        torch._foreach_addcmul_(scores, grads, param_updates, value=1.0 / score_history)

    # weight decay
    n_weight_decay = len(weight_decay)
    params_decay = [-weight_decay[i % n_weight_decay] * param_updates[i].abs() * params[i]
                    for i in range(len(step_sizes))]
    torch._foreach_add_(params, params_decay)

    # update parameters
    torch._foreach_add_(params, param_updates, alpha=-1.0)

    # update previous gradients
    torch._foreach_copy_(prevs_1, grads)
