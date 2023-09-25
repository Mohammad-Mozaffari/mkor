import matplotlib.pyplot as plt
import os
import numpy as np
import torch


# plt.style.use('seaborn')
# plt.style.use('tex')

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


fig_dim = set_size(450, fraction=0.5)

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.titlesize": 8,
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 8,
    "font.size": 8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.figsize": fig_dim,
    'lines.linewidth': 0.4,
    'figure.facecolor': "white"
}


plt.rcParams.update(tex_fonts)





def get_model_name(model_name):
    if model_name.startswith("t2t_vit"):
        return f"T2T-ViT-{model_name.split('_')[2]}"
    elif model_name.startswith("resnet"):
        return f"ResNet-{model_name[6:]}"
    elif model_name == 'alexnet':
        return 'AlexNet'
    elif model_name == 'autoencoder':
        return 'Autoencoder'
    elif model_name.startswith('bert'):
        name = 'BERT '
        if model_name.split('_')[1] == 'base':
            name += 'Base'
        else:
            name += 'Large'
        if model_name.split('_')[2] == 'uncased':
            name += ' Uncased'
        else:
            name += ' Cased'
        if model_name.endswith("question_answering"):
            name += ' Question Answering'
        elif model_name.endswith("classification"):
            name += ' Text Classification'
        return name
    else:
        return model_name


def get_dataset_name(dataset_name):
    if dataset_name == "imagenet":
        return "ImageNet"
    elif dataset_name == "cifar10":
        return "CIFAR-10"
    elif dataset_name == "cifar100":
        return "CIFAR-100"
    elif dataset_name == "mnist":
        return "MNIST"
    elif dataset_name == "squad":
        return "SQuAD"
    elif dataset_name == "sst2":
        return "SST-2"
    elif dataset_name == "imdb":
        return "IMDB"
    elif dataset_name.startswith("wikicorpus"):
        return "Wikipedia"
    else:
        return dataset_name


def plot(x, y, hline=None, title="", xlabel="", ylabel="", legend=None, save_path=None, new_figure=True, label="",
         mark_max=False, semilogy=False, semilogx=False, **kwargs):
    if semilogx and semilogy:
        plot_func = plt.loglog
    elif semilogx:
        plot_func = plt.semilogx
    elif semilogy:
        plot_func = plt.semilogy
    else:
        plot_func = plt.plot
    if new_figure:
        plt.figure()
    if hline is not None:
        plot_func(x, y, x, len(x) * [hline], **kwargs)
    else:
        print(kwargs)
        line, = plot_func(x, y, label=label, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='major', color='grey', alpha=0.3, linewidth=0.3)
    plt.grid(True, which='minor', color='grey', linestyle='--', alpha=0.3, linewidth=0.1)
    plt.minorticks_on()
    plt.tick_params(which='major', axis="y", direction="in", width=0.5, color='grey')
    plt.tick_params(which='minor', axis="y", direction="in", width=0.3, color='grey')
    plt.tick_params(which='major', axis="x", direction="in", width=0.5, color='grey')
    plt.tick_params(which='minor', axis="x", direction="in", width=0.3, color='grey')
    ax = plt.gca()

    # Remove the upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('grey')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('grey')
    ax.spines['left'].set_linewidth(0.5)
    if mark_max:
        max_idx = np.where(y >= np.max(y) - 0.1)[0][0]
        # if np.max(y) > 99.4:
        #     max_idx = np.where(y >= 98.9)[0][0]
        print(f"{label} max at {np.max(y)} for {x[max_idx]}")
        # if label.startswith("kfac_approx_average_automated") and title.endswith("Train Accuracy"):
        #     max_idx = np.where(np.array(y) > 96.5)[0][0]
        plot_func([x[max_idx], x[max_idx]], [100, 0], "--", c=line.get_color(), **kwargs)
        plt.text(x[max_idx], 103, f"{int(x[max_idx])}", size=6, rotation=90)
        plt.ylim([0, 115])
    if legend is not None:
        plt.legend(legend, frameon=True)
    else:
        plt.legend(frameon=True)
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')


def pie_plot(data_dict, title="", save_path=None):
    val_list = []
    label_list = []

    for label, val in data_dict.items():
        if label == "train_time":
            continue
        val_list.append(val)
        label_list.append(label)

    val_list = np.array(val_list)
    val_list, label_list = zip(*sorted(zip(val_list, label_list), reverse=True))
    val_list = np.array(val_list)
    percentage = 100. * (val_list / val_list.sum())
    patches, texts = plt.pie(val_list, startangle=90)
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(label_list, percentage)]
    plt.legend(patches, labels, loc="center left", bbox_to_anchor=(-0.35, 0.5), fontsize=8, frameon=True)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')


def plot_results(method, optimizer_name, model_name, dataset_name, args, epochs, train_losses, train_accs,
                 regularizations,
                 baseline_train_loss, baseline_train_acc, test_losses, test_accs, baseline_test_loss, baseline_test_acc,
                 sparsity_ratios, timer):
    results = {"epochs": epochs, "train_losses": train_losses, "train_accs": train_accs,
               "regularizations": regularizations, "test_losses": test_losses, "test_accs": test_accs,
               "sparsity_ratios": sparsity_ratios}
    figure_path = f"./figures/{method}/{model_name}/{dataset_name}/{optimizer_name}_lr{args.lr}_wd{args.weight_decay}_freq{args.inv_freq}"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    torch.save(results, f"{figure_path}/results.pt")
    if timer.measure:
        timer.save(figure_path)
        pie_plot(timer.time, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} - {torch.cuda.get_device_name()}",
                 save_path=f"{figure_path}/timing_{torch.cuda.get_device_name()}.pdf")

    if method != "prune":
        baseline_train_acc, baseline_test_acc = None, None
    train_epochs = np.linspace(0, epochs, len(train_losses))
    test_epochs = np.linspace(0, epochs, len(test_losses))
    plot(train_epochs, train_losses, hline=baseline_train_loss, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Train Loss",
         xlabel="Epochs", ylabel="Loss", legend=["Train Loss", "Baseline Train Loss"],
         save_path=f"{figure_path}/training_loss.pdf")
    plot(train_epochs, train_accs, hline=baseline_train_acc, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Train Accuracy",
         xlabel="Epochs", ylabel="Accuracy", legend=["Train Accuracy", "Baseline Train Accuracy"],
         save_path=f"{figure_path}/training_accuracy.pdf")
    plot(test_epochs, test_losses, hline=baseline_test_loss, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Test Loss",
         xlabel="Epochs", ylabel="Loss", legend=["Test Loss", "Baseline Test Loss"],
         save_path=f"{figure_path}/test_loss.pdf")
    plot(test_epochs, test_accs, hline=baseline_test_acc, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Test Accuracy",
         xlabel="Epochs", ylabel="Accuracy", legend=["Test Accuracy", "Baseline Test Accuracy"],
         save_path=f"{figure_path}/test_accuracy.pdf")
    if method == "prune":
        plot(train_epochs, regularizations, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Regularization", xlabel="Epochs",
             ylabel="Regularization", legend=["Regularization"], save_path=f"{figure_path}/regularization.pdf")
        plot(test_epochs, sparsity_ratios, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Sparsity Ratio", xlabel="Epochs",
             ylabel="Sparsity Ratio", legend=["Sparsity Ratio"], save_path=f"{figure_path}/sparsity_ratio.pdf")
    if hasattr(timer, "train_time"):
        train_time = np.linspace(0, timer.train_time / 1000, len(train_losses))
        test_time = np.linspace(0, timer.train_time / 1000, len(test_losses))
        plot(train_time, train_losses, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Train Loss", xlabel="Time (s)",
             ylabel="Loss", legend=["Train Loss"], save_path=f"{figure_path}/training_loss_time.pdf")
        plot(train_time, train_accs, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Train Accuracy", xlabel="Time (s)",
             ylabel="Accuracy", legend=["Train Accuracy"], save_path=f"{figure_path}/training_accuracy_time.pdf")
        plot(test_time, test_losses, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Test Loss", xlabel="Time (s)",
             ylabel="Loss", legend=["Test Loss"], save_path=f"{figure_path}/test_loss_time.pdf")
        plot(test_time, test_accs, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Test Accuracy", xlabel="Time (s)",
             ylabel="Accuracy", legend=["Test Accuracy"], save_path=f"{figure_path}/test_accuracy_time.pdf")
        if method == "prune":
            plot(train_time, regularizations, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Regularization",
                 xlabel="Time (s)", ylabel="Regularization", legend=["Regularization"],
                 save_path=f"{figure_path}/regularization_time.pdf")
            plot(test_time, sparsity_ratios, title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - {optimizer_name} Sparsity Ratio", xlabel="Time (s)",
                 ylabel="Sparsity Ratio", legend=["Sparsity Ratio"], save_path=f"{figure_path}/sparsity_ratio_time.pdf")


def plot_all_optimizers(method, model_name, dataset_name, min_train_acc=0, min_test_acc=0):
    figure_path = f"./figures/{method}/{model_name}/{dataset_name}/"
    optimizer_data = {}
    has_timing = False
    for folder in os.listdir(figure_path):
        if folder.endswith(".pdf"):
            continue
        optimizer_name = folder
        data = torch.load(f"{figure_path}{folder}/results.pt")
        if torch.any(torch.tensor(data["test_accs"]) >= min_test_acc) and torch.any(
                torch.tensor(data["train_accs"]) >= min_train_acc):
            optimizer_data[optimizer_name] = data
            for file in os.listdir(f"{figure_path}{folder}"):
                if file.endswith(".time"):
                    timer = torch.load(f"{figure_path}{folder}/{file}")
                    if 'train_time' in timer:
                        optimizer_data[optimizer_name]["timer"] = timer
                        has_timing = True
    # Plot training loss
    plt.figure()
    for optimizer, data in optimizer_data.items():
        step_size = len(data["train_losses"]) // data["epochs"]
        x_samples = np.array([0] + [i * step_size - 1 for i in range(1, data["epochs"] + 1)]).astype(int)
        train_epochs = np.linspace(0, data["epochs"], len(data["train_losses"]))
        plot(train_epochs[x_samples], np.array(data["train_losses"])[x_samples], title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Train Loss",
             xlabel="Epochs", ylabel="Loss", label=optimizer, save_path=f"{figure_path}/training_loss.pdf",
             new_figure=False)

    # Plot training accuracy
    plt.figure()
    for optimizer, data in optimizer_data.items():
        step_size = len(data["train_losses"]) // data["epochs"]
        x_samples = np.array([0] + [i * step_size - 1 for i in range(1, data["epochs"] + 1)]).astype(int)
        train_epochs = np.linspace(0, data["epochs"], len(data["train_accs"]))
        plot(train_epochs[x_samples], np.array(data["train_accs"])[x_samples], title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Train Accuracy",
             xlabel="Epochs", ylabel="Accuracy", label=optimizer, save_path=f"{figure_path}/training_accuracy.pdf",
             new_figure=False, mark_max=True)

    # Plot test loss
    plt.figure()
    for optimizer, data in optimizer_data.items():
        test_epochs = np.linspace(0, data["epochs"], len(data["test_losses"]))
        plot(test_epochs, data["test_losses"], title=f"{get_model_name(model_name)}", xlabel="Epochs", ylabel="Loss",
             label=optimizer, save_path=f"{figure_path}/test_loss.pdf", new_figure=False)

    # Plot test accuracy
    plt.figure()
    for optimizer, data in optimizer_data.items():
        test_epochs = np.linspace(0, data["epochs"], len(data["test_accs"]))
        plot(test_epochs, data["test_accs"], title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Test Accuracy", xlabel="Epochs", ylabel="Accuracy",
             label=optimizer, save_path=f"{figure_path}/test_accuracy.pdf", new_figure=False, mark_max=True)

    if has_timing:
        # Plot training loss
        plt.figure()
        for optimizer, data in optimizer_data.items():
            if not 'timer' in data:
                continue
            step_size = len(data["train_losses"]) // data["epochs"]
            x_samples = np.array([0] + [i * step_size - 1 for i in range(1, data["epochs"] + 1)]).astype(int)
            train_time = np.linspace(0, data["timer"]['train_time'] / 1000, len(data["train_losses"]))
            plot(train_time[x_samples], np.array(data["train_losses"])[x_samples], title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Train Loss",
                 xlabel="Time (s)", ylabel="Loss", label=optimizer, save_path=f"{figure_path}/training_loss_time.pdf",
                 new_figure=False)
        # Plot training accuracy
        plt.figure()
        for optimizer, data in optimizer_data.items():
            if not 'timer' in data:
                continue
            step_size = len(data["train_losses"]) // data["epochs"]
            x_samples = np.array([0] + [i * step_size - 1 for i in range(1, data["epochs"] + 1)]).astype(int)
            train_time = np.linspace(0, data["timer"]['train_time'] / 1000, len(data["train_accs"]))
            print(train_time[-1], data["timer"]['train_time'])
            plot(train_time[x_samples], np.array(data["train_accs"])[x_samples], title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Train Accuracy",
                 xlabel="Time (s)", ylabel="Accuracy", label=optimizer,
                 save_path=f"{figure_path}/training_accuracy_time.pdf", new_figure=False, mark_max=True)

        # Plot test loss
        plt.figure()
        for optimizer, data in optimizer_data.items():
            if not 'timer' in data:
                continue
            test_time = np.linspace(0, data["timer"]['train_time'] / 1000, len(data["test_losses"]))
            plot(test_time, data["test_losses"], title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Test Loss", xlabel="Time (s)", ylabel="Loss",
                 label=optimizer, save_path=f"{figure_path}/test_loss_time.pdf", new_figure=False)
        # Plot test accuracy
        plt.figure()
        for optimizer, data in optimizer_data.items():
            if not 'timer' in data:
                continue
            test_time = np.linspace(0, data["timer"]['train_time'] / 1000, len(data["test_accs"]))
            plot(test_time, data["test_accs"], title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Test Accuracy", xlabel="Time (s)",
                 ylabel="Accuracy", label=optimizer, save_path=f"{figure_path}/test_accuracy_time.pdf",
                 new_figure=False, mark_max=True)

    if method == "prune":
        # Plot regularization
        plt.figure()
        for optimizer, data in optimizer_data.items():
            train_epochs = np.linspace(0, data["epochs"], len(data["regularizations"]))
            plot(train_epochs, data["regularizations"], title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Regularization", xlabel="Epochs",
                 ylabel="Regularization", label=optimizer, save_path=f"{figure_path}/regularization.pdf",
                 new_figure=False)
        # Plot sparsity ratio
        plt.figure()
        for optimizer, data in optimizer_data.items():
            test_epochs = np.linspace(0, data["epochs"], len(data["sparsity_ratios"]))
            plot(test_epochs, data["sparsity_ratios"], title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Sparsity Ratio", xlabel="Epochs",
                 ylabel="Sparsity Ratio", label=optimizer, save_path=f"{figure_path}/sparsity_ratio.pdf",
                 new_figure=False)


def get_optimizer_name(optimizer):
    optimizer = optimizer.lower()
    if optimizer.startswith("sgd"):
        return "SGD"
    elif optimizer.startswith('kfac_approx_average') or optimizer.startswith('mkor'):
        return "MKOR"
    elif optimizer.startswith('kfac'):
        return "KFAC"
    elif optimizer.startswith('hylo_kis'):
        return "HyLo-KIS"
    elif optimizer.startswith('hylo'):
        return 'HyLo'
    elif optimizer.startswith('kaisa'):
        return "KAISA"
    elif optimizer.startswith('lamb'):
        return "LAMB"
    elif optimizer.startswith('adamw'):
        return "ADAMW"
    elif optimizer.startswith('adam'):
        return "ADAM"



def compare_optimizers(method, model_name, dataset_name):

    figure_path = f"./figures/{method}/{model_name}/{dataset_name}/"
    optimizer_data = {}
    labels = ["Factor Computation", "Precondition", "Update Weights"]#, "Forward", "Backward"]
    for folder in os.listdir(figure_path):
        if folder.endswith(".pdf"):
            continue
        for file in os.listdir(f"{figure_path}{folder}"):
            if file.endswith(".time"):
                epoch = torch.load(f"{figure_path}{folder}/results.pt")["epochs"]
                timer = torch.load(f"{figure_path}{folder}/{file}")
                time = []
                folder = folder[1:]
                if folder.startswith('hylo') or 'kfac' in folder or folder.startswith('kaisa') or folder.startswith('mkor'):
                    if folder.startswith('hylo'):
                        time.append(timer["update_inv"] + timer["broadcast_inv"] + timer["compute_factors"] + timer[
                            "allgather_factors"])
                        time[-1] *= 2
                    elif folder.startswith('kaisa'):
                        time.append(timer['reduce_invert_and_broadcast_factors'])
                    elif folder.startswith('kfac_approx') or folder.startswith('mkor'):
                        time.append(timer["reduce_and_update_factors"])
                    else:
                        time.append(timer['update_inv'] + timer['reduce_factors'])
                    time.append(timer["precondition"])
                    try:
                        time.append(timer["update_weights"])
                    except:
                        time.append(timer["apply_updates"])
                elif folder.startswith('sgd') or folder.startswith('adam') or folder.startswith('lamb'):
                    time.append(0)
                    time.append(0)
                    time.append(timer["optimizer"])
                # time.append(timer["forward"])
                # time.append(timer["backward"])
                optimizer_data[folder] = [t / epoch for t in time]
    num_optimizers = len(optimizer_data)
    optimizer_cnt = 0
    width = 0.8 / num_optimizers
    fig, ax = plt.subplots()
    optimizer_list = ["mkor", "hylo", "kaisa", "lamb", "sgd", "adam"]
    for opt in optimizer_list:
        for optimizer in optimizer_data:
            if opt in optimizer:
                ax.bar(np.arange(len(labels)) + optimizer_cnt * width, np.array(optimizer_data[optimizer]) / 1000, width,
                        label=get_optimizer_name(optimizer))
                optimizer_cnt += 1
    ax.set_yscale('log')
    plt.ylabel("Time (s)")
    plt.grid(True, which='major', color='grey', alpha=0.3, linewidth=0.3)
    plt.grid(True, which='minor', color='grey', linestyle='--', alpha=0.3, linewidth=0.1)
    # plt.minorticks_on()
    plt.tick_params(which='major', axis="y", direction="in", width=0.5, color='grey')
    plt.tick_params(which='minor', axis="y", direction="in", width=0.3, color='grey')
    plt.tick_params(which='major', axis="x", direction="out", width=0.5, color='grey')
    # plt.tick_params(which='minor', axis="x", direction="in", width=0.3, color='grey')
    ax = plt.gca()

    # Remove the upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('grey')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('grey')
    ax.spines['left'].set_linewidth(0.5)
    plt.xticks(np.arange(len(labels)) + width * num_optimizers / 2, labels, rotation=15, fontsize=6)
    plt.legend(frameon=True)
    plt.title(f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)}")
    plt.savefig(f"{figure_path}/optimizer_comparison.pdf", format='pdf', bbox_inches='tight')


def plot_eigenvalues():
    import pandas as pd
    import os
    eigenvalues = pd.read_csv("./results/eigen.csv")
    min_eigenvalues = eigenvalues['min_r_abs_eigen'].values
    max_eigenvalues = eigenvalues['max_r_abs_eigen'].values
    average_length = 51
    for i in range(len(min_eigenvalues) - average_length):
        min_eigenvalues[i] = np.mean(min_eigenvalues[i:i + average_length])
        max_eigenvalues[i] = np.mean(max_eigenvalues[i:i + average_length])
    min_eigenvalues = min_eigenvalues[:len(min_eigenvalues) - average_length]
    max_eigenvalues = max_eigenvalues[:len(max_eigenvalues) - average_length]
    condition_numbers = max_eigenvalues / min_eigenvalues
    if not os.path.exists("./figures/eigenvalues"):
        os.makedirs("./figures/eigenvalues")
    plot(range(len(min_eigenvalues)), min_eigenvalues, title="ResNet-50 - CIFAR-10 - Eigenvalues of Right Factor", xlabel="Iteration", ylabel="Eigenvalue", save_path="./figures/eigenvalues/right_eigenvalues.pdf", new_figure=True, label="Minimum Eigenvalues", semilogy=True)
    plot(range(len(max_eigenvalues)), max_eigenvalues, title="ResNet-50 - CIFAR-10 - Eigenvalues of Right Factor", xlabel="Iteration", ylabel="Eigenvalue", save_path="./figures/eigenvalues/right_eigenvalues.pdf", new_figure=False, label="Maximum Eigenvalues", semilogy=True)
    plot(range(len(condition_numbers)), condition_numbers, title="ResNet-50 - CIFAR-10 - Condition Number of Right Factor", xlabel="Iteration", ylabel="Condition Number", save_path="./figures/eigenvalues/right_condition_number.pdf", new_figure=True, label="Condition Number", semilogy=True)


def plot_scalability(args):
    import pandas as pd
    import os
    global_batch_size = args.global_batch_size
    data = pd.read_csv("./results/training_times.csv")
    data = data[data['Model'] == args.model_name]
    data = data[data['Dataset'] == args.dataset]
    optimizers = data['Optimizer'].unique()
    min_gpu, max_gpu = torch.inf, 0
    if not os.path.exists("./figures/scalability"):
        os.makedirs("./figures/scalability")
    plt.figure()
    for optimizer in optimizers:
        optimizer_data = data[data['Optimizer'] == optimizer]
        x = optimizer_data['World Size'].values
        indices = np.argsort(x)
        x = x[indices]
        world_sizes = x
        local_batch_sizes = optimizer_data['Batch Size per GPU'].values[indices]
        min_gpu = min(min_gpu, x[0])
        max_gpu = max(max_gpu, x[-1])
        training_time = optimizer_data['Training Time per Epoch'].values[indices]
        optimizer_time = optimizer_data['Optimizer Time'].values[indices]
        forward_backward_time = training_time - optimizer_time
        current_batch_sizes = world_sizes * local_batch_sizes
        grad_accum_iters = (global_batch_size / current_batch_sizes)
        # print(optimizer)
        # print(grad_accum_iters)
        # print(forward_backward_time)
        scaled_forward_backward_time = forward_backward_time * grad_accum_iters
        y = scaled_forward_backward_time + optimizer_time
        y = y[0] / y * x[0]
        # print(y)
        # print(x[0])
        plot(x, y, title=f"{get_model_name(args.model_name)} - {get_dataset_name(args.dataset)} - Scalability", xlabel="Number of GPUs", ylabel="Speedup", save_path=f"./figures/scalability/{args.model_name}_{args.dataset}_scalability.pdf", new_figure=False, label=optimizer, marker=".")
    plot([min_gpu, max_gpu], [min_gpu, max_gpu],
         title=f"{get_model_name(args.model_name)} - {get_dataset_name(args.dataset)} - Scalability",
         xlabel="Number of GPUs", ylabel="Speedup",
         save_path=f"./figures/scalability/{args.model_name}_{args.dataset}_scalability.pdf", new_figure=False,
         label="Linear Scalaing")


def plot_iteration_costs():
    if not os.path.exists("./figures/iteration_costs"):
        os.makedirs("./figures/iteration_costs")
    frequencies = [1, 10, 100, 1000]
    mkor_time = [12.101695, 7.986541345, 7.707057307, 7.616709121]
    kaisa_time = [259.6036081, 33.73774658, 12.2326193, 10.09648684]
    plot(frequencies, mkor_time, title="Autoencoder",
         xlabel="Factor Reuse Time", ylabel="Time/Iteration (ms)",
         save_path="./figures/iteration_costs/inversion_freq_sensitivity_autoencoder_cifar100.pdf", new_figure=True, label="MKOR", semilogx=True)

    plot(frequencies, kaisa_time, title="Autoencoder",
         xlabel="Factor Reuse Time", ylabel="Time/Iteration (ms)",
         save_path="./figures/iteration_costs/inversion_freq_sensitivity_autoencoder_cifar100.pdf", new_figure=False, label="KAISA", semilogx=True)

    mkor_time = [342.0261563, 261.883375, 252.6682344, 258.9976406]
    kaisa_time = [7329.896509, 935.8902067, 295.2770516, 230.8643684]

    plot(frequencies, mkor_time, title="BERT-Large-Uncased",
         xlabel="Factor Reuse Time", ylabel="Time/Iteration (ms)",
         save_path="./figures/iteration_costs/inversion_freq_sensitivity_bert_large_uncased_wikipedia.pdf", new_figure=True, label="MKOR", semilogx=True)

    plot(frequencies, kaisa_time, title="BERT-Large-Uncased",
         xlabel="Factor Reuse Time", ylabel="Time/Iteration (ms)",
         save_path="./figures/iteration_costs/inversion_freq_sensitivity_bert_large_uncased_wikipedia.pdf", new_figure=False, label="KAISA", semilogx=True)


def plot_loss_from_csv(file_dir, model_name, dataset_name):
    import pandas as pd
    data = pd.read_csv(file_dir)
    figure_path = f"./dense/{model_name}/{dataset_name}/"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.figure()
    for optimizer in data.columns:
        losses = data[optimizer].values
        end_time = losses[-1]
        losses = losses[:-1]
        while np.isnan(end_time):
            end_time = losses[-1]
            losses = losses[:-1]
        plot(range(1, losses.shape[0] + 1), losses, new_figure=False,
             title=f"{get_model_name(model_name)}",
             xlabel="Epochs", ylabel="Loss", legend=data.columns,
             save_path=f"figures/{figure_path}/{model_name}_loss.pdf")
    plt.figure()
    for optimizer in data.columns:
        losses = data[optimizer].values
        end_time = losses[-1]
        losses = losses[:-1]
        while np.isnan(end_time):
            end_time = losses[-1]
            losses = losses[:-1]
        time = np.linspace(0, end_time, losses.shape[0])
        print("End time:", end_time)
        plot(time, losses, new_figure=False,
             # title=f"{get_model_name(model_name)} - {get_dataset_name(dataset_name)} - Training Loss",
             xlabel="Time (h)", ylabel="Loss", legend=data.columns,
             save_path=f"figures/{figure_path}/{model_name}_loss_time.pdf")


def plot_scalability_from_csv(file_dir, model_name):
    import pandas as pd
    data = pd.read_csv(file_dir)
    figure_path = f"./scalability/"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.figure()
    num_gpus = data[data.columns[0]].values
    plot(num_gpus, num_gpus, new_figure=True, legend="Linear Scaling")
    for optimizer in data.columns[1:]:
        time = data[optimizer].values
        scalability = time[0] / time
        scalability *= num_gpus[0]
        print(scalability)
        # while np.isnan(end_time):
        #     end_time = losses[-1]
        #     losses = losses[:-1]
        plot(num_gpus, scalability, new_figure=False,
             title=f"{get_model_name(model_name)} Scalability",
             xlabel="\#GPUs", ylabel="Speedup", legend=data.columns,
             save_path=f"figures/{figure_path}/{model_name}_scalability.pdf")