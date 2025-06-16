import torch
from torch.optim.optimizer import Optimizer

class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if not isinstance(base_optimizer, Optimizer):
            raise ValueError("base_optimizer는 torch.optim.Optimizer의 인스턴스여야 합니다.")

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        scale = self._get_scale(grad_norm)
        
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None: continue
                # 원래의 파라미터 저장
                self.state[p]['old_p'] = p.data.clone()
                # epsilon 방향으로 이동
                if i == len(group['params'])-1:
                    continue
                p.add_(p.grad, alpha=scale)

    @torch.no_grad()
    def second_step(self):
        # 파라미터 복원 및 기본 옵티마이저 스텝 실행
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.data = self.state[p]['old_p']  # 원래의 파라미터로 복원
        self.base_optimizer.step()  # 기본 옵티마이저 스텝 실행

    def _grad_norm(self):
        # 그래디언트의 L2 노름 계산
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2) for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def _get_scale(self, grad_norm):
        # 스케일 계산
        return self.defaults['rho'] / (grad_norm + 1e-12)

    def zero_grad(self):
        self.base_optimizer.zero_grad()
