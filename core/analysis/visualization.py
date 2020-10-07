""" Functionality for visualizing heatmaps for models trained on the PTB-XL dataset.

Example usage (e.g. in Jupyter notebook):

from core.analysis.visualization import compute_and_show_heatmap
from core.models import PTBXLClassificationModel
clf = PTBXLClassificationModel.load_from_checkpoint(
    path_to_model_checkpoint,
    hparams_file=path_to_hparams_file
)
clf.prepare_data()
label_mapping = {k: i for i, k in enumerate(clf.labels)}
compute_and_show_heatmap(
    clf.model.model, clf.val_dataset[5], label_mapping
)
"""

import cv2
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from .utils import flatten_heatmap, flatten_signal


def show_signal_heatmap(
    flattened_signals: List[np.ndarray],
    flattened_heatmap: np.ndarray,
    class_name: str,
    prob: float,
    target: str,
    layer_name: str = 'pool3',
    figsize: Tuple[int, int] = (25, 100),
    output_fp: str = None,
    show_fig: bool = True
) -> Figure:
    """Returns a visualization figure of the signal heatmap on all leads of a given signal.

    Args:
        flattened_signals (List[np.ndarray]): list of signals for each lead
        flattened_heatmap (np.ndarray): heatmap of the same length as a signal for a single
        class_name (str): name of the class for the associated heatmap output
        prob (float): probability of the class prediction being present in the signal
        target (str): the ground truth for the given signal (one of Present or Absent)
        layer_name (str): name of the layer that produced this heatmap. Defaults to 'pool3'.
        figsize (Tuple[int, int]): size of the figure for output
        output_fp (str, optional): if provided, saves figure to file, otherwise shows inline.
            Defaults to None.
        show_fig (bool, optional): if True, shows figure (e.g. inline in a notebook).
            Defaults to True.

    Returns:
        Figure: figure with all leads visualized with overlayed heatmap
    """
    x = np.arange(0, len(flattened_heatmap))
    if figsize[1] / figsize[0] <= 2:
        print(
            f'WARNING: heatmap figures might look distorted. Suggested figure size for '
            f'given width: ({figsize[0], figsize[0] * 4})'
        )
    fig, axs = plt.subplots(len(flattened_signals), sharex=True, figsize=figsize)

    for i, (ax, fs) in enumerate(zip(axs, flattened_signals)):
        points = np.array([x, fs]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(flattened_heatmap)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)
        ax.set_title(f'Lead {i + 1}')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(fs.min(), fs.max())
    plt.tight_layout()
    fig.suptitle(
        f'GradCAM ({layer_name}) for {class_name} classification '
        f'(predicted {prob:.3f}; actual: {target})',
        fontsize=20
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.97)

    if output_fp:
        plt.savefig(output_fp)
    elif show_fig:
        plt.show()
    return fig


def compute_and_show_heatmap(
    model: torch.nn.Module,
    submodule_name: str,
    datapoint: Tuple[torch.Tensor, torch.Tensor],
    label_mapping: Dict[str, int],
    class_of_concern: str = 'MI',
    figsize: Tuple[int, int] = (25, 100),
    output_fp: Optional[str] = None,
    show_fig: bool = True,
    verbose: bool = False
) -> Optional[Figure]:
    """Computes and visualizes a heatmap using `model` for a given `datapoint`.

    Args:
        model (torch.nn.Module): model to produce the heatmap
        submodule_name (str): name of submodule within `model` to get gradient-weighted activations
        datapoint (Tuple[torch.Tensor, torch.Tensor]): tuple of signal and label that is usually
            given to the model
        label_mapping (Dict[str, int]): mapping from label to class index
        class_of_concern (str, optional): class prediction to visualize heatmap for.
            Defaults to 'MI'.
        figsize (Tuple[int, int]): size of the figure for output
        output_fp (str, optional): if provided, saves figure to file, otherwise shows inline.
            Defaults to None.
        show_fig (bool, optional): if True, shows figure (e.g. inline in a notebook).
            Defaults to True.
        verbose (bool, optional): if True, prints more debug statements.


    Returns:
        Optional[Figure]: figure with all leads visualized with overlayed heatmap. If gradients
            vanished, returns None, as heatmap would be blank.
    """
    assert hasattr(model, submodule_name), f'Model does not have submodule {submodule_name}.'
    signal, label = datapoint
    if len(signal.shape) == 2:
        signal = signal[None, ...]
    label_index = label_mapping[class_of_concern]

    grad_dict = {}
    activation_dict = {}

    def get_activation_hooks(submodule):
        def activation_grad_hook(module, input, output):
            output_hook = None

            def output_grad_hook(grad):
                assert grad.shape == output.shape
                grad_dict[submodule] = grad.detach()
                output_hook.remove()

            output_hook = output.requires_grad_().register_hook(output_grad_hook)

        def activation_output_hook(module, input, output):
            activation_dict[submodule] = output.detach()

        return (
            getattr(model, submodule).register_forward_hook(activation_grad_hook),
            getattr(model, submodule).register_forward_hook(activation_output_hook)
        )

    activation_grad_handle, activation_output_handle = get_activation_hooks(submodule_name)
    try:
        model.eval()
        model.zero_grad()
        if torch.cuda.is_available():
            signal = signal.cuda()
        output = model(signal)
        probs = torch.sigmoid(output).mean(axis=0).cpu()
        label_prob = probs[label_index].item()
        if verbose:
            print(
                f"Model would have predicted "
                f"{'Present' if label_prob >= 0.5 else 'Absent'}"
                f" for {class_of_concern} presence."
            )
            print(
                f"Model should have predicted "
                f"{'Present' if label[label_index] > 0 else 'Absent'}"
                f" for {class_of_concern} presence."
            )
        probs[label_index].backward()

        gradients = grad_dict[submodule_name]
        activations = activation_dict[submodule_name]
        pooled_gradients = torch.mean(gradients, dim=[0, 1])
        activation_grid = (pooled_gradients.unsqueeze(0).unsqueeze(0) * activations)
        activation_grid = torch.mean(activation_grid, dim=1).squeeze()
        activation_grid = np.maximum(activation_grid.detach().cpu(), 0)
        activation_grid /= torch.max(activation_grid)

        if torch.isnan(activation_grid).any().item():
            if verbose:
                print(f'Gradients vanished due to very strong prediction, returning None.')
            fig = None
        else:
            signal = signal.cpu().numpy()
            resized_heatmap = activation_grid.numpy()
            resized_heatmap = cv2.resize(resized_heatmap, (signal.shape[-1], signal.shape[0]))
            fig = show_signal_heatmap(
                flatten_signal(signal), flatten_heatmap(resized_heatmap), class_of_concern,
                label_prob, 'Present' if label[label_index] > 0 else 'Absent',
                layer_name=submodule_name, figsize=figsize, output_fp=output_fp, show_fig=show_fig
            )

    finally:
        activation_grad_handle.remove()
        activation_output_handle.remove()
        return fig
