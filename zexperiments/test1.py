import os
import sys
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel, IndexKernel
from ztest import HammingDistanceKernel
from skgpytorch.models import ExactGPRegressor
import matplotlib.pyplot as plt
from bayesian_benchmarks.data import Boston, Wilson_autompg
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from skgpytorch.metrics import negative_log_predictive_density
import json
import pandas as pd


class Data:
    pass


if sys.argv[1] == "boston":
    data = Boston()
    cat_index = [3]

elif sys.argv[1] == "autompg":
    data = Wilson_autompg()
    cat_index = [0, 6]

elif sys.argv[1] == "laptops":
    laptops = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "laptops_clean.csv"), encoding="latin-1"
    )
    data = Data()
    data.name = "laptops"
    for i in range(laptops.shape[1] - 1):
        try:
            scaler = StandardScaler()
            laptops.iloc[:, i] = scaler.fit_transform(
                laptops.iloc[:, i].values.reshape(-1, 1)
            )
        except:
            scaler = LabelEncoder()
            laptops.iloc[:, i] = scaler.fit_transform(laptops.iloc[:, i].values)
            # print(laptops[laptops.columns[i]].unique())
            print("Labeling column {}".format(laptops.columns[i]))
    data.X = laptops.iloc[:, :-1].values
    data.y = laptops.iloc[:, -1].values
    data.y = StandardScaler().fit_transform(data.y.reshape(-1, 1)).ravel()
    data.X_train, data.X_test, data.Y_train, data.Y_test = train_test_split(
        data.X,
        data.y,
        test_size=0.2,
        random_state=42,
    )
    cat_index = [0, 1, 4, 8]
    encoder = LabelEncoder()
    for c in cat_index:
        encoder.fit(data.X[:, c])
        data.X_train[:, c] = encoder.transform(data.X_train[:, c])
        data.X_test[:, c] = encoder.transform(data.X_test[:, c])

if sys.argv[1] != "laptops":
    encoder = LabelEncoder()
    for c in cat_index:
        data.X_train[:, c] = encoder.fit_transform(data.X_train[:, c])
        data.X_test[:, c] = encoder.transform(data.X_test[:, c])

cont_index = sorted(set(range(data.X_train.shape[1])) - set(cat_index))
n_iters = 100
seed = 0


# print(set(data.X_train[:, 3]))
# sys.exit()

cont_kernel = RBFKernel(ard_num_dims=len(cont_index), active_dims=cont_index)
X = torch.tensor(data.X_train, dtype=torch.float)
Y = torch.tensor(data.Y_train, dtype=torch.float).ravel()

hamming_kernel = None
for c in cat_index:
    if hamming_kernel is None:
        hamming_kernel = HammingDistanceKernel(
            num_categories=len(set(data.X_train[:, c])),
            ard_num_dims=1,
            active_dims=[c],
        )
    else:
        hamming_kernel = hamming_kernel * HammingDistanceKernel(
            num_categories=len(set(data.X_train[:, c])),
            ard_num_dims=1,
            active_dims=[c],
        )

index_kernel = None
for c in cat_index:
    if index_kernel is None:
        index_kernel = IndexKernel(
            num_tasks=len(set(data.X_train[:, c])),
            ard_num_dims=1,
            active_dims=[c],
        )
    else:
        index_kernel = index_kernel * IndexKernel(
            num_tasks=len(set(data.X_train[:, c])),
            ard_num_dims=1,
            active_dims=[c],
        )

ii = 0
cat_kernels = [
    hamming_kernel,
    RBFKernel(ard_num_dims=len(cat_index), active_dims=cat_index),
    index_kernel,
][ii:]
cat_names = ["Hamming", "RBF", "Index"][ii:]
cat_colors = ["red", "green", "blue"][ii:]
titles = ["NLPD"]
nlpd = {i: [] for i in cat_names}
for i in range(len(cat_names)):
    for seed in range(20):
        print(i, seed)
        full_kernel = ScaleKernel(cont_kernel * cat_kernels[i])
        gp = ExactGPRegressor(
            X,
            Y,
            kernel=full_kernel,
            random_state=seed,
            verbose=0,
            device="cuda",
        )
        gp.fit(n_iters=n_iters)
        nlpd[cat_names[i]].append(
            negative_log_predictive_density(
                gp.model,
                gp.likelihood,
                torch.tensor(data.X_test, dtype=torch.float).cuda(),
                torch.tensor(data.Y_test, dtype=torch.float).cuda().ravel(),
            )
        )

        plt.plot(gp.history["train_loss"], color=cat_colors[i], linewidth=1)
    plt.plot(
        gp.history["train_loss"], label=cat_names[i], color=cat_colors[i], linewidth=1
    )
    tmp_title = (
        cat_names[i]
        + ": Mean = "
        + str(round(sum(nlpd[cat_names[i]]) / len(nlpd[cat_names[i]]), 4))
        + " Std: "
        + str(round(torch.std(torch.tensor(nlpd[cat_names[i]])).item(), 4))
    )
    print(tmp_title)
    titles.append(tmp_title)

plt.title("\n".join(titles))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), f"loss_{data.name}.png"))
json.dump(
    nlpd, open(os.path.join(os.path.dirname(__file__), f"loss_{data.name}.json"), "w")
)
