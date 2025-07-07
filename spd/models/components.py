import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.module_utils import init_param_


class Gate(nn.Module):
    """A gate that maps a single input to a single output."""

    def __init__(self, C: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((C,)))
        self.bias = nn.Parameter(torch.zeros((C,)))
        fan_val = 1  # Since each weight gets applied independently
        init_param_(self.weight, fan_val=fan_val, nonlinearity="linear")

    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        return x * self.weight + self.bias


class GateMLP(nn.Module):
    """A gate with a hidden layer that maps a single input to a single output."""

    def __init__(self, C: int, n_ci_mlp_neurons: int):
        super().__init__()
        self.n_ci_mlp_neurons = n_ci_mlp_neurons

        self.mlp_in = nn.Parameter(torch.empty((C, n_ci_mlp_neurons)))
        self.in_bias = nn.Parameter(torch.zeros((C, n_ci_mlp_neurons)))
        self.mlp_out = nn.Parameter(torch.empty((C, n_ci_mlp_neurons)))
        self.out_bias = nn.Parameter(torch.zeros((C,)))

        init_param_(self.mlp_in, fan_val=1, nonlinearity="relu")
        init_param_(self.mlp_out, fan_val=n_ci_mlp_neurons, nonlinearity="linear")

    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        hidden = (
            einops.einsum(
                x,
                self.mlp_in,
                "... C, C n_ci_mlp_neurons -> ... C n_ci_mlp_neurons",
            )
            + self.in_bias
        )
        hidden = F.gelu(hidden)

        out = (
            einops.einsum(
                hidden,
                self.mlp_out,
                "... C n_ci_mlp_neurons, C n_ci_mlp_neurons -> ... C",
            )
            + self.out_bias
        )
        return out
    
class GateGraph(nn.Module):
    """
    A graph that consists of subcomponent nodes and component-summary nodes.
    To compute importance, we first compute representations for each subcomponent node (0-hop).
    We update the component-summary nodes based on the subcomponent nodes for the same weight (1-hop).
    Finally, each individual subcomponent is updated based on the component-summary nodes (2-hop).
    """

    def __init__(self, C: int, d_subcomponent: int, d_summary:int, components: nn.ModuleDict):
        super().__init__()

        self.C = C
        self.d_subcomponent = d_subcomponent
        self.d_summary = d_summary
        self.num_components = len(components)

        self.subcomponent_proj = nn.Linear(
            in_features=1, # Inner activation
            out_features=d_subcomponent,
        )
        self.subcomponent_to_summary = nn.Linear(d_subcomponent, d_summary)
        self.summary_to_subcomponent = nn.Linear(d_summary, d_subcomponent)
        self.summary_to_summary = nn.Linear(d_summary, d_summary)
        self.subcomponent_gate = nn.Linear(
            in_features=d_subcomponent,
            out_features=1,  # C subcomponents
        )
        

    def _get_edge_indices(self):
        total_nodes = self.num_components * (self.C + 1)  # C subcomponents + 1 summary per component
        sub_to_summary_adjacency = torch.zeros((total_nodes, total_nodes), dtype=torch.bool)
        summary_to_sub_adjacency = torch.zeros((total_nodes, total_nodes), dtype=torch.bool)
        # Let the summary nodes be the last C nodes in the total_nodes
        summary_start_idx = total_nodes - self.num_components
        # Each continuous, disjoint block of C subcomponent nodes is connected to the summary node
        for i in range(self.num_components):
            summary_node_idx = summary_start_idx + i
            subcomponent_indices = torch.arange(self.C) + i * self.C

            # Connect subcomponents to their summary node
            sub_to_summary_adjacency[summary_node_idx,
                                    subcomponent_indices] = True            
            # Connect the summary node to all its subcomponents
            summary_to_sub_adjacency[subcomponent_indices,
                                    summary_node_idx] = True


        # Register buffers
        def row_norm(A: Float[Tensor, "... nodes"]) -> Float[Tensor, "... nodes"]:
            deg = A.sum(dim=-1, keepdim=True).clamp(min=1)  # avoid /0
            return A / deg

        summary_adj = torch.ones((self.num_components, self.num_components), dtype=torch.float)
        sub_to_summary_adjacency = row_norm(sub_to_summary_adjacency.float())
        summary_to_sub_adjacency = row_norm(summary_to_sub_adjacency.float())
        summary_adj = row_norm(summary_adj)
        self.register_buffer("summary_to_summary", summary_adj.to(self.subcomponent_proj.weight.device))
        self.register_buffer("sub_to_summary_adjacency", sub_to_summary_adjacency.to(self.subcomponent_proj.weight.device))
        self.register_buffer("summary_to_sub_adjacency", summary_to_sub_adjacency.to(self.subcomponent_proj.weight.device))

    def _convert_subcomponents_to_x(self, components_inner_act: dict[str, Float[Tensor, "... C"]]) -> Float[Tensor, "... C_x_num_components d_subcomponent"]:
        # Create the x vector for the subcomponent nodes
        acts = [act for act in components_inner_act.values()]
        # Concatenate the inner activations for all components
        xs = torch.stack(acts, dim=-1)  # Shape: (... C num
        xs = self.subcomponent_proj(xs.unsqueeze(-1)) # Shape: (... C num_components d_subcomponent)
        # Reshape into (... C*num_components d_subcomponent)

        xs = xs.view(*xs.shape[:-3], self.C * self.num_components, self.d_subcomponent)
        # Shape is now (... C * num_components d_subcomponent)
        # But adjacency matrix is of shape (num_components * (C + 1), num_components * (C + 1))
        # So we need to add some 1-padding to the end of the xs tensor to represent the summary nodes
        padding = torch.ones(*xs.shape[:-2], self.num_components, self.d_subcomponent, device=xs.device)
        xs = torch.cat([xs, padding], dim=-2)  # Now shape is (... C * num_components + num_components, d_subcomponent)
        # Now we have the subcomponent nodes in the first C * num_components dimensions
        # and the summary nodes in the last
        return xs

    def forward(self, components_inner_act: dict[str, Float[Tensor, "... C"]]) -> dict[str, Float[Tensor, "... C"]]:
        """Forward pass through the gate graph."""

        # Get the edge indices
        if not hasattr(self, "sub_to_summary_adjacency"):
            self._get_edge_indices()

        # Convert subcomponents to x
        x_orig = self._convert_subcomponents_to_x(components_inner_act)

        x = einops.einsum(self.sub_to_summary_adjacency, x_orig, "a b, ... b dsubcomponent -> ... a dsubcomponent")
        x = self.subcomponent_to_summary(x)  # Shape is (... C * num_components, dsummary)
        x = F.gelu(x)
        x = einops.einsum(self.summary_to_sub_adjacency.to(x.device), x, "a b, ... b dsummary -> ... a dsummary")
        summaries_orig = x[..., -self.num_components:, :]
        summaries = einops.einsum(self.summary_to_summary, summaries_orig, "a b, ... b d -> ... a d")
        summaries = F.gelu(self.summary_to_summary(summaries))  + summaries_orig
        x[..., -self.num_components:, :] = summaries
        x = self.summary_to_subcomponent(x)  # Shape is (... C * num_components, dsubcomponent)
        x = F.gelu(x)
        x = x + x_orig
        x = self.subcomponent_gate(x)  # Shape is (... C * num_components, 1)
        x = x.squeeze(-1)  # Remove the last dimension, now shape is (... C * num_components)
        # Now we can get rid of the summary nodes, as we only care about the subcomponent importance
        x = x[..., :self.C * self.num_components]  # Keep only the first C * num_components elements
        # Now x is the importance for each subcomponent, shape (... C * num_components)
        # Reshape to (... num_components C)
        x = x.view(*x.shape[:-1], self.num_components, self.C)        # Wrap back into a dictionary with component names
        importance = {}
        for i, name in enumerate(components_inner_act.keys()):
            importance[name] = x[..., i, :]
        # Now importance is a dictionary mapping component names to their importance scores
        # Each importance score is of shape (... C)
        return importance
        

class GateGraphFast(nn.Module):
    """
    Equivalent to GateGraph but uses a more efficient implementation because it is a Star-graph.
    This is fairly limited in terms of what changes we can make, so I kept the original too.
    """
    def __init__(self, C: int, d_subcomponent: int, d_summary: int, components: nn.ModuleDict):
        super().__init__()
        self.C = C
        self.n = len(components)

        self.sub_proj = nn.Linear(1, d_subcomponent)
        self.sub_to_sum = nn.Linear(d_subcomponent, d_summary)
        self.sub_to_sub = nn.Linear(d_subcomponent, d_subcomponent)
        self.sum_to_sub = nn.Linear(d_summary, d_subcomponent)
        self.sub_gate = nn.Linear(d_subcomponent, 1)

    def forward(self, inner_act: dict[str, Tensor]):
        x = torch.stack(list(inner_act.values()), dim=-2).unsqueeze(-1)

        sub = self.sub_proj(x)

        summary = sub.sum(dim=-2) / self.C
        summary = F.gelu(self.sub_to_sum(summary))

        global_sum = summary.sum(dim=-2, keepdim=True) / self.n
        global_sum = self.sub_to_sub(global_sum)
        global_sum = F.gelu(global_sum)

        summary = summary + global_sum

        sub = sub + F.gelu(self.sum_to_sub(summary)).unsqueeze(-2)

        scores = self.sub_gate(sub).squeeze(-1)

        return {name: scores[..., i, :]for i, name in enumerate(inner_act)}


class LinearComponent(nn.Module):
    """A linear transformation made from A and B matrices for SPD.

    NOTE: In the paper, we use V and U for A and B, respectively.

    The weight matrix W is decomposed as W = B^T @ A^T, where A and B are learned parameters.
    """

    def __init__(self, d_in: int, d_out: int, C: int, bias: Tensor | None):
        super().__init__()
        self.C = C

        self.A = nn.Parameter(torch.empty(d_in, C))
        self.B = nn.Parameter(torch.empty(C, d_out))
        self.bias = bias

        init_param_(self.A, fan_val=d_out, nonlinearity="linear")
        init_param_(self.B, fan_val=C, nonlinearity="linear")

        self.mask: Float[Tensor, "... C"] | None = None  # Gets set on sparse forward passes

    @property
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """B^T @ A^T"""
        return einops.einsum(self.A, self.B, "d_in C, C d_out -> d_out d_in")

    # @torch.compile
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Forward pass through A and B matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all components
        """
        component_acts = einops.einsum(x, self.A, "... d_in, d_in C -> ... C")

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(component_acts, self.B, "... C, C d_out -> ... d_out")

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponent(nn.Module):
    """An efficient embedding component for SPD that avoids one-hot encoding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        C: int,
    ):
        super().__init__()
        self.C = C

        self.A = nn.Parameter(torch.empty(vocab_size, C))
        self.B = nn.Parameter(torch.empty(C, embedding_dim))

        init_param_(self.A, fan_val=embedding_dim, nonlinearity="linear")
        init_param_(self.B, fan_val=C, nonlinearity="linear")

        # For masked forward passes
        self.mask: Float[Tensor, "batch pos C"] | None = None

    @property
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        """A @ B"""
        return einops.einsum(
            self.A, self.B, "vocab_size C, ... C embedding_dim -> vocab_size embedding_dim"
        )

    # @torch.compile
    def forward(self, x: Float[Tensor, "batch pos"]) -> Float[Tensor, "batch pos embedding_dim"]:
        """Forward through the embedding component using nn.Embedding for efficient lookup

        NOTE: Unlike a LinearComponent, here we alter the mask with an instance attribute rather
        than passing it in the forward pass. This is just because we only use this component in the
        newer lm_decomposition.py setup which does monkey-patching of the modules rather than using
        a SPDModel object.

        Args:
            x: Input tensor of token indices
        """
        # From https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L1211
        component_acts = self.A[x]  # (batch pos C)

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(
            component_acts, self.B, "batch pos C, ... C embedding_dim -> batch pos embedding_dim"
        )
        return out
