
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from . import metrics

def plot_img(x, title=None, ylabel=None):
    plt.imshow(x)
    plt.title(title)        
    plt.ylabel(ylabel)
    plt.axis("off")


def plot_label(number_of_classes, l, ylabel=None, add_color_bar=True):
    cmap=matplotlib.colors.ListedColormap([plt.cm.gist_ncar(i/number_of_classes) \
                                            for i in range(number_of_classes)])
    
    
    plt.imshow(l, vmin=0, vmax=number_of_classes, cmap=cmap, interpolation='none')
    plt.ylabel(ylabel)
    if add_color_bar:
        cbar = plt.colorbar(ticks=range(number_of_classes))
        cbar.ax.set_yticklabels([f"{i}" for i in range(number_of_classes)])  # vertically oriented colorbar
    if ylabel is None:
        plt.axis("off")
                    
def plot_segmentation_probabilities(run, l, out, out_segmentation, ylabel=None, include_f1=False, include_mae=False):     
    cmap=matplotlib.colors.ListedColormap([plt.cm.gist_ncar(i/run.number_of_classes) \
                                            for i in range(run.number_of_classes)])
    
    tout_segmentation = np.argmax(out_segmentation, axis=-1)

    cmetrics = metrics.PixelClassificationMetrics(number_of_classes=run.number_of_classes)
    cmetrics.reset_state()
    cmetrics.update_state(l.reshape(1, *l.shape), out_segmentation.reshape(-1, *out_segmentation.shape))
    f1  = cmetrics.result('f1', 'micro')
    mae = run.metrics.multiclass_proportions_mae_on_chip(l.reshape(1,*l.shape), out.reshape(1,*out.shape))
    
    if not include_f1 and not include_mae:
        title = None
    else:
        title = "onchip  "
        if include_f1:
            title += f' f1 {f1:.3f}'
        if include_mae:
            title += f' mae {mae:.3f}'

    plt.imshow(tout_segmentation, vmin=0, vmax=run.number_of_classes, cmap=cmap, interpolation='none')
    cbar = plt.colorbar(ticks=range(run.number_of_classes))
    cbar.ax.set_yticklabels([f"{i}" for i in range(run.number_of_classes)])  # vertically oriented colorbar
    plt.title(title)
    plt.ylabel(ylabel)
    if ylabel is None:
        plt.axis("off")
                
                        
def plot_proportions_prediction(run, p, l, out, show_legend=False):
    nc = run.number_of_classes
    y_pred_proportions = run.metrics.get_y_pred_as_proportions(out.reshape(1,*out.shape), argmax=True)[0]
    onchip_proportions = run.metrics.get_class_proportions_on_masks(l.reshape(1,*l.shape))[0]

    maec = run.metrics.multiclass_proportions_mae_on_chip(l.reshape(1,*l.shape), out.reshape(1,*out.shape))

    plt.bar(np.arange(nc)-.2, p, 0.2, label="on partition", alpha=.5)
    plt.bar(np.arange(nc), onchip_proportions, 0.2, label="on chip", alpha=.5)
    plt.bar(np.arange(nc)+.2, y_pred_proportions, 0.2, label="pred", alpha=.5)
    if show_legend:
        plt.legend()
    plt.grid();
    plt.xticks(np.arange(nc), np.arange(nc));
    plt.title(f"mae {maec:.3f}")            
    plt.xlabel("class number")
    plt.ylim(0,1)
    plt.ylabel("proportions")
