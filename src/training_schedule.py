from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings

# from .optimizer import Optimizer



class ThreePhaseOneCycleLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 max_lr,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 pct_epoch=[1, 13],
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 three_phase=False,
                 last_epoch=-1,
                 verbose=False):

        # Validate optimizer
        # if not isinstance(optimizer, Optimizer):
        #     raise TypeError('{} is not an Optimizer'.format(
        #         type(optimizer).__name__))
        self.optimizer = optimizer

        # Validate total_steps
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected positive integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected positive integer epochs, but got {}".format(epochs))
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError("Expected positive integer steps_per_epoch, but got {}".format(steps_per_epoch))
            self.total_steps = epochs * steps_per_epoch

        if three_phase:
            # self._schedule_phases = [
            #     {
            #         'end_step': float(pct_start * self.total_steps) - 1,
            #         'start_lr': 'initial_lr',
            #         'end_lr': 'max_lr',
            #         'start_momentum': 'max_momentum',
            #         'end_momentum': 'base_momentum',
            #     },
            #     {
            #         'end_step': float(2 * pct_start * self.total_steps) - 2,
            #         'start_lr': 'max_lr',
            #         'end_lr': 'initial_lr',
            #         'start_momentum': 'base_momentum',
            #         'end_momentum': 'max_momentum',
            #     },
            #     {
            #         'end_step': self.total_steps - 1,
            #         'start_lr': 'initial_lr',
            #         'end_lr': 'min_lr',
            #         'start_momentum': 'max_momentum',
            #         'end_momentum': 'max_momentum',
            #     },
            # ]
            self._schedule_phases = [
                {
                    'end_step': float(pct_epoch[0] * steps_per_epoch) - 1,
                    'start_lr': 'initial_lr',
                    'end_lr': 'max_lr',
                    'start_momentum': 'max_momentum',
                    'end_momentum': 'base_momentum',
                },
                {
                    'end_step': float(pct_epoch[1] * steps_per_epoch) - 2,
                    'start_lr': 'max_lr',
                    'end_lr': 'max_lr',
                    'start_momentum': 'base_momentum',
                    'end_momentum': 'base_momentum',
                },
                {
                    'end_step': self.total_steps - 1,
                    'start_lr': 'max_lr',
                    'end_lr': 'min_lr',
                    'start_momentum': 'base_momentum',
                    'end_momentum': 'max_momentum',
                },
            ]
        else:
            self._schedule_phases = [
                {
                    'end_step': float(pct_start * self.total_steps) - 1,
                    'start_lr': 'initial_lr',
                    'end_lr': 'max_lr',
                    'start_momentum': 'max_momentum',
                    'end_momentum': 'base_momentum',
                },
                {
                    'end_step': self.total_steps - 1,
                    'start_lr': 'max_lr',
                    'end_lr': 'min_lr',
                    'start_momentum': 'base_momentum',
                    'end_momentum': 'max_momentum',
                },
            ]

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))

        # Validate anneal_strategy
        # if anneal_strategy not in ['cos', 'linear']:
        #     raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        # elif anneal_strategy == 'cos':
        #     self.anneal_func = self._annealing_cos
        # elif anneal_strategy == 'linear':
        #     self.anneal_func = self._annealing_linear
        # self.anneal_func = [self._annealing_linear, self._annealing_linear, self._annealing_cos]
        self.anneal_func = [self._annealing_cos, self._annealing_linear, self._annealing_cos]

        # Initialize learning rate variables
        max_lrs = self._format_param('max_lr', self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group['initial_lr'] = max_lrs[idx] / div_factor
                group['max_lr'] = max_lrs[idx]
                group['min_lr'] = group['initial_lr'] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super(ThreePhaseOneCycleLR, self).__init__(optimizer, last_epoch, verbose)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        for group in self.optimizer.param_groups:
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase['end_step']
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    computed_lr = self.anneal_func[i](group[phase['start_lr']], group[phase['end_lr']], pct)
                    if self.cycle_momentum:
                        computed_momentum = self.anneal_func[i](group[phase['start_momentum']], group[phase['end_momentum']], pct)
                    break
                start_step = phase['end_step']

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum

        return lrs