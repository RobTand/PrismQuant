"""Actual captured X/g replay; change only the three product/reduction sites."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import shutil
import textwrap

import torch

from experiments.qdq_constant_residency import main as replay
from prismaquant import joint_aura
from prismaquant.kernels import joint_projection_reduce as kernel


class Candidate:
    schema = 'pq.joint_projection_reduction.v1'

    def __init__(self, output):
        self.output = output
        self.identity = kernel.build_identity()
        self.identity['acceptance'] = 'bit-exact FP32 individual reductions and signed probe components; no relaxed tolerance'
        self.baseline_path = Path(joint_aura.__file__)
        self.before = joint_aura.SignedJointProjectionLease._observe
        self.before_source = textwrap.dedent(inspect.getsource(self.before))
        tree = ast.parse(self.before_source)
        changed = []

        class Replace(ast.NodeTransformer):
            def visit_Call(self, node):
                operands = None
                if (isinstance(node.func, ast.Attribute) and node.func.attr == 'sum'
                        and isinstance(node.func.value, ast.BinOp) and isinstance(node.func.value.op, ast.Mult)
                        and not node.args and not node.keywords):
                    operands = node.func.value.left, node.func.value.right
                elif (isinstance(node.func, ast.Attribute) and node.func.attr == '_projection_product_sum'
                      and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'
                      and len(node.args) == 2 and not node.keywords):
                    operands = tuple(node.args)
                if operands is not None:
                    left, right = map(ast.unparse, operands)
                    component = {('gw', 'delta'): 'weight',
                                 ('d_operator', 'source_weight.float()'): 'activation',
                                 ('d_operator', 'delta'): 'mixed'}[(left, right)]
                    changed.append(component)
                    return ast.copy_location(ast.Call(func=ast.Name(id='_projection_product_sum', ctx=ast.Load()),
                        args=list(operands), keywords=[ast.keyword(arg='component', value=ast.Constant(component))]), node)
                return self.generic_visit(node)

        tree = ast.fix_missing_locations(Replace().visit(tree))
        assert sorted(changed) == ['activation', 'mixed', 'weight']
        self.after_source = ast.unparse(tree) + '\n'
        namespace = dict(vars(joint_aura))
        namespace['_projection_product_sum'] = self.product
        exec(compile(tree, str(output / 'after.py'), 'exec'), namespace)
        self.after = namespace['_observe']
        self.checking = False
        self.reduction_checks = []

    def activate(self, function):
        joint_aura.SignedJointProjectionLease._observe = function

    def persist_binary(self, output):
        binary = Path(self.identity['binary_path'])
        shutil.copy2(binary, output / binary.name)
        shutil.copy2(binary.parent / 'build.ninja', output / 'build.ninja')

    def product(self, left, right, *, component):
        if not kernel.fast_path_eligible(left, right):
            raise RuntimeError('actual projection escaped the measured fused CUDA geometry')
        actual = kernel.fused_product_sum(left, right)
        if self.checking:
            expected = (left * right).sum()
            bits = actual.view(torch.int32).item()
            reference = expected.view(torch.int32).item()
            check = {'index': len(self.reduction_checks), 'component': component,
                     'shape': list(left.shape), 'left_stride': list(left.stride()), 'right_stride': list(right.stride()),
                     'candidate_bits': bits, 'reference_bits': reference,
                     'candidate_value': actual.item(), 'reference_value': expected.item(), 'equal': bits == reference}
            self.reduction_checks.append(check)
            if bits != reference:
                (self.output / 'numerical-failure.json').write_text(json.dumps(self.reduction_checks, indent=2))
                raise AssertionError(f'fused reduction changed FP32 bits: {check}')
        return actual


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    candidate = Candidate(args.output)
    try:
        replay(variant_controller=candidate)
    finally:
        candidate.activate(candidate.before)


if __name__ == '__main__':
    main()
