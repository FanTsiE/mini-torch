from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals = list(vals)
    f1 = f(*vals)
    vals[arg] -= epsilon
    f2 = f(*vals)
    return (f1 - f2) / epsilon
   


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    res = []
    perma_mark = set()
    temp_mark = set()
    def dfs(n):
        if n.is_constant():
            return
        if n.unique_id in perma_mark:
            return
        if n.unique_id in temp_mark:
            return RuntimeError("Not a DAG")
        temp_mark.add(n.unique_id)
        for parent in n.parents:
            dfs(parent)
        temp_mark.remove(n.unique_id)
        perma_mark.add(n.unique_id)
        res.append(n)
    dfs(variable)
    res.reverse()
    return res
    


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    derivs = {variable.unique_id: deriv}
    variables = topological_sort(variable)

    for var in variables:
        if var.is_leaf():
            var.accumulate_derivative(derivs[var.unique_id])
        else:
            for v, d in var.chain_rule(derivs[var.unique_id]):
                if v.unique_id not in derivs.keys():
                    derivs[v.unique_id] = d
                else:
                    derivs[v.unique_id] += d

    
   


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
