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
    vals[arg] += epsilon
    f1 = f(*vals)
    vals[arg] -= 2 * epsilon
    f2 = f(*vals)
    return (f1 - f2) / (2 * epsilon)
    # raise NotImplementedError('Need to implement for Task 1.1')


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
        if n.is_leaf():
            pass
        else:
            for i in n.history.inputs:
                dfs(i)
        temp_mark.remove(n.unique_id)
        perma_mark.add(n.unique_id)
        res.append(n)
    dfs(variable)
    return res
    # raise NotImplementedError('Need to implement for Task 1.4')


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
    ordered = topological_sort(variable)
    d = {variable.unique_id: deriv}
    for v in ordered:
        d_output = d[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(d_output)
        else:
            for parent, d_parent in v.history.last_fn.chain_rule(d_output):
                if parent.unique_id not in d:
                    d[parent.unique_id] = 0.0
                d[parent.unique_id] += d_parent
    return
    # raise NotImplementedError('Need to implement for Task 1.4')


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
