1. Fetch the data and other utilities
Link : https://rebrand.ly/presto
Download resnet18.pt and place it in PyTorch_CIFAR10/cifar10_models/state_dicts
Download resnet18.pt and place it in PyTorch_CIFAR100/cifar100_models/state_dicts
Download resnet18.pt and place it in PyTorch_STL10/stl10_models/state_dicts

2. Creation of features - Run the following 2 commands to generate the features.

cd PyTorch_[CIFAR10/CIFAR100/STL10]-ROTATE
python3 [cifar-rotate/cifar100-rotate/stl-rotate]_features.py

The following commands involve a number of command line arguments, each of which are described as follows - 

(a) cuda : This argument specifies the ID of the GPU core to be used
(b) ckpt : This argument specifies the directory where checkpoints would be saved
(c) cluster_init : This argument specifies the clustering method to be used for the experiment. It must be one of {EqKMeans, KMeans++, Agglomerative, GMM, BGM}
(d) dataset : This argument specifies the dataset on which the experiment will be performed. It must be one of {cifar10, pathmnist, dermamnist, svhn}
(e) lee_way : This argument specifies the lee_way parameter which is used for the data-partitioning
(f) n_bins : This argument specifies the number of bins to be used
(g) clustering_baseline : If this argument is included, the code does not run PRESTO, it runs the chosen clustering baseline

2. Running PRESTO

python3 presto.py --cuda [] --ckpt [] --cluster_init [] --dataset [] --lee_way [] --n_bins []

3. Running clustering baselines

python3 presto.py --cuda [] --ckpt [] --cluster_init [] --dataset [] --lee_way [] --n_bins [] --clustering_baseline

4. Running Learn-MLR baseline

python3 learn_mlr_baseline.py --cuda [] --ckpt [] --dataset [] --n_bins []

5. Running KD/DGKD baseline

python3 KD.py --cuda [] --ckpt [] --dataset []
python3 DGKD.py --cuda [] --ckpt [] --dataset []

6. Running pruning baselines (recommended 150 epochs for baselines, 20 epochs for GraSP/SNIP)

We have adapted the code from : https://github.com/JingtongSu/sanity-checking-pruning

(a) presto_baseline : This argument is necessary to run the GraSP/SNIP baseline adapted for PRESTO
(b) epochs : This argument specifies the epochs for which the experiment will be run
(c) save_dir : This argument specifies the directory where the checkpoints will be saved
(d) dataset : This argument specifies the dataset. It should be one of {cifar10-rotate, cifar100-rotate, stl-rotate}
(e) model : This argument specifies the path of the baseline model
(f) init_prune_ratio : This argument specifies the pruning ratio for GraSP/SNIP - this specifies how many network parameters you actually want whose gradient will flow through. All others will be set to zero. In the form of a ratio.

cd pruning_baselines
python3 baseline.py --presto_baseline --epochs [] --save_dir [] --dataset []
python3 train_ticket.py --presto_baseline --lr 0.1 --model []  --save_dir [] --writerdir tensorboard/ --init_prune_ratio [] --epochs [] --GraSP 1
python3 train_ticket.py --presto_baseline --lr 0.1 --model []  --save_dir [] --writerdir tensorboard/ --init_prune_ratio [] --epochs [] --SNIP 1

7. Running MoE

cd MoE
python3 moe.py