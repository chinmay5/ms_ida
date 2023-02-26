import random
from collections import defaultdict

import numpy as np
import os
import pickle
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR, get_configurations_dtype_boolean, get_configurations_dtype_float, \
    device, get_configurations_dtype_string, get_configurations_dtype_int
from utils.sup_contras_loss import SupConLoss
from utils.viz_utils import plot_bar_plot

smallness_threshold = get_configurations_dtype_int(section='SETUP', key='DIFF_GRAPH_THRESHOLD')


class LogWriterWrapper(object):
    def __init__(self, summary_writer=None):
        self.summary_writer = summary_writer

    def add_scalar(self, *args, **kwargs):
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(*args, **kwargs)

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag="default", metadata_header=None):
        if self.summary_writer is not None:
            self.summary_writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=global_step,
                                              tag=tag, metadata_header=metadata_header)


class CustomDictKey(object):
    def __init__(self, key_name, key_iden):
        super(CustomDictKey, self).__init__()
        self.key_name = key_name
        self.key_iden = key_iden

    def __eq__(self, other):
        return isinstance(other, CustomDictKey) and \
               other.key_name == self.key_name and \
               other.key_iden == self.key_iden

    def __hash__(self):
        return hash((self.key_name, self.key_iden))

    def __repr__(self):
        return f'{self.key_name} - {self.key_iden}'


class LabelEncoder(object):

    def __init__(self):
        enc = OneHotEncoder()
        possible_labels = np.array([0, 1]).reshape(-1, 1)
        enc.fit(possible_labels)
        self.encoder = enc

    def __call__(self):
        return self.encoder


class RunTimeConfigs(object):
    def __init__(self):
        self.configs = []

    def write_to_disk(self):
        base_log_dir = os.path.join(PROJECT_ROOT_DIR, self.logdir)
        os.makedirs(base_log_dir, exist_ok=True)
        filename = os.path.join(base_log_dir, "configs_for_run.cfg")
        with open(filename, 'w') as configfile:
            for config, value in vars(self):
                configfile.write(f"{config}: {value} \n")


# Ensuring reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


def drop_nodes(graph):
    # We can drop 10% of the nodes at random.
    node_mask = torch.rand(graph.num_nodes) >= 0.1
    # We do not prune when there are a very few nodes left.
    if node_mask.sum() <= 4:
        return graph
    sub_graph = graph.subgraph(node_mask)
    return sub_graph


def shuffle_dataset(loader, dataset_refresh_metadata, mixup_train_loader=None):
    graph_dataset, train_idx = dataset_refresh_metadata
    train_with_more_augmentations = [(graph_dataset[idx.item()][1], graph_dataset[idx.item()][2]) for idx in train_idx]
    vars(loader)['dataset'] = train_with_more_augmentations
    if mixup_train_loader is not None:
        # We have to update the training dataset.
        # Forgetting it leads to mixup getting applied only on the original samples and not augmented ones.
        vars(mixup_train_loader)['dataset'] = train_with_more_augmentations


class MixupTrainer(object):
    def __init__(self, gnn_model, criterion, optimizer, regr_criterion, train_writer):
        super(MixupTrainer, self).__init__()
        self.gnn_model = gnn_model
        # Idea: Since we would do a mixup on two dataloaders, with more weight to the instance based version,
        # we would expect an intrinsic un-balanced nature that we need to account for.
        self.criterion = criterion
        # A hacky solution to get the same lr and weight decay terms
        # self.optim = torch.optim.Adam(
        #     chain(self.gnn_model.lin2.parameters(), self.gnn_model.lin_new_lesion_regr.parameters()))
        self.optim = optimizer
        # self.optim.defaults = optimizer.defaults
        self.regr_criterion = regr_criterion
        self.train_writer = train_writer

    def run_through_gnn(self, graph, graph2):
        # with torch.no_grad():
        out1 = self.gnn_model(graph)
        h1 = self.gnn_model.graph_level_feat
        out2 = self.gnn_model(graph2)
        h2 = self.gnn_model.graph_level_feat
        mixed_features, lam = self.mixup_data(graph_features1=h1, graph_features2=h2)
        # Now we compute loss only on the classification layer
        prediction = self.gnn_model.lin2(mixed_features)
        vol_pred = self.gnn_model.lin_new_lesion_regr(mixed_features).squeeze(-1)
        return prediction, vol_pred, lam, out1['node_vol'], out2['node_vol'], h1, h2

    def mixup_data(self, graph_features1, graph_features2, alpha=4):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        mixed_features = lam * graph_features1 + (1 - lam) * graph_features2
        return mixed_features, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def mixup_regr_criterion(self, pred, y_a, y_b, lam):
        return lam * self.regr_criterion(pred, y_a) + (1 - lam) * self.regr_criterion(pred, y_b)

    def run_mixup_training(self, graph, idx, label, graph2, label2, criterion_vol_regr, scheduler):
        class_pred, vol_pred, lam, node_vol_1, node_vol_2, feat1, feat2 = self.run_through_gnn(graph=graph,
                                                                                               graph2=graph2)
        cls_loss = self.mixup_criterion(pred=class_pred, y_a=label.squeeze(), y_b=label2.squeeze(), lam=lam)
        regr_loss = self.mixup_regr_criterion(pred=vol_pred, y_a=graph.graph_vol, y_b=graph2.graph_vol, lam=lam)
        # Let us also add in the lesion volume loss
        node_regr_loss = criterion_vol_regr(node_vol_1, graph.node_labels) + criterion_vol_regr(node_vol_2,
                                                                                                graph2.node_labels)
        loss = cls_loss + regr_loss + node_regr_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        if scheduler is not None:
            scheduler.step()
        self.train_writer.add_scalar('mixup_cls_loss', cls_loss.item(), global_step=idx)
        self.train_writer.add_scalar('mixup_regr_loss', regr_loss.item(), global_step=idx)
        self.train_writer.add_scalar('loss', loss.item(), global_step=idx)
        # Since we are computing the regression loss for two graphs, dividing by 2 while plotting.
        self.train_writer.add_scalar('vol_regr_loss', node_regr_loss.item() / 2, global_step=idx)


class ContrastiveTrainer(object):
    def __init__(self):
        super(ContrastiveTrainer, self).__init__()
        self.contras_loss = SupConLoss()
        self.balanced_criterion = nn.CrossEntropyLoss()

    def execute_contras_training(self, criterion_vol_regr, data, epoch, idx, label, loader, model, optimizer,
                                 total_loss, train_writer, scheduler):
        # Pass through the model
        out = model(data)
        graph_level_feat = model.graph_level_feat.unsqueeze(1)
        # Normalizing the features
        graph_level_contrast_feat = F.normalize(graph_level_feat, dim=-1)
        # Pass through the contrastive learning pipeline
        contras_loss = self.contras_loss(graph_level_contrast_feat, label)
        # Let us detach the representation and compute FC output
        # squeezing back so that the downstream tasks work as expected
        graph_level_feat = graph_level_feat.squeeze().detach()
        optimizer.zero_grad()
        # passing through the expected pipeline
        graph_label = model.lin2(graph_level_feat)
        new_lesion_vol = model.lin_new_lesion_regr(graph_level_feat).squeeze()
        graph_cls_loss = self.balanced_criterion(graph_label, label.view(-1))
        new_lesion_vol_regr_loss = criterion_vol_regr(new_lesion_vol, data.graph_vol)
        node_regr_loss = criterion_vol_regr(out['node_vol'], data.node_labels)
        loss = graph_cls_loss + node_regr_loss + new_lesion_vol_regr_loss + contras_loss
        loss_sans_contras = graph_cls_loss + node_regr_loss + new_lesion_vol_regr_loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Let us also plot per iteration loss to get some fine-grained information
        train_writer.add_scalar('vol_regr_loss', node_regr_loss.item(), global_step=epoch * len(loader) + idx)
        train_writer.add_scalar('contras_loss', contras_loss.item(), global_step=epoch * len(loader) + idx)
        train_writer.add_scalar('graph_cls_loss', graph_cls_loss.item(), global_step=epoch * len(loader) + idx)
        train_writer.add_scalar('graph_regr_loss', new_lesion_vol_regr_loss.item(),
                                global_step=epoch * len(loader) + idx)
        return loss_sans_contras.item()


def graph_size_based_sampler(train_dataset):
    per_sample_wt = [0] * len(train_dataset)
    for idx, (graph, label) in enumerate(train_dataset):
        per_sample_wt[idx] = graph.x.size(0)  # Assigning the length weight to our sampler
    weighted_sampler = WeightedRandomSampler(per_sample_wt, num_samples=len(per_sample_wt), replacement=False)
    return weighted_sampler


def balanced_batch_sampler(train_dataset):
    per_sample_wt = [0] * len(train_dataset)
    class_weights = get_class_weights(train_dataset)
    for idx, (graph, label) in enumerate(train_dataset):
        class_weight = class_weights[label]
        per_sample_wt[idx] = class_weight  # Assigning the length weight to our sampler
    weighted_sampler = WeightedRandomSampler(per_sample_wt, num_samples=len(per_sample_wt), replacement=True)
    return weighted_sampler


def get_class_weights(train_dataset):
    # We also need to obtain class weights to ensure we do not have data imbalance issues.
    pos_samples = sum([sample[1] for sample in train_dataset])
    neg_samples = len(train_dataset) - pos_samples
    # Label 0 is negative and label 1 is positive
    if pos_samples > neg_samples:
        class_balance_weights = torch.as_tensor([pos_samples / neg_samples, 1], device=device)
    else:
        class_balance_weights = torch.as_tensor([1, neg_samples / pos_samples], device=device)
    return class_balance_weights


def get_training_enhancements(criterion, criterion_vol_regr, model, optimizer, train_writer, class_balance_weights,
                              train_dataset, batch_size, lr, weight_decay, train_loader):
    # Assigning each value to None.
    # Python specific way of assinging. `None` is immutable so we are fine. If it were a list/collection, all
    # inputs would map to the same object and it might cause issues.
    mixup_trainer = contras_trainer = mixup_train_loader = feature_alignment_loss = optimizer_centre_loss = None
    use_mixup = get_configurations_dtype_boolean(section='TRAINING', key='USE_MIXUP')
    print(f"using Mixup {use_mixup}.")
    if use_mixup:
        mixup_trainer = MixupTrainer(gnn_model=model, criterion=criterion, train_writer=train_writer,
                                     optimizer=optimizer, regr_criterion=criterion_vol_regr)
        # weighted_sampler = create_weighted_sampler(class_weights=class_balance_weights, dataset=train_dataset)
        mixup_train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True,
                                        worker_init_fn=seed_worker, generator=g)
        # We need to make drop_last=True for mixup training to work.
        # This would ensure that both dataloaders always have the same shape
        train_loader = DataLoader(train_dataset, batch_size, drop_last=True, sampler=train_loader.sampler,
                                  worker_init_fn=seed_worker, generator=g)

    use_contras_training = get_configurations_dtype_boolean(section='TRAINING', key='USE_CONTRAS_TRAINING')
    print(f"using Contrastive training {use_contras_training}.")
    if use_contras_training:
        if use_mixup:
            raise AttributeError("Contrastive training not supported with mixup. Aborting!")
        contras_trainer = ContrastiveTrainer()

    use_centre_loss = get_configurations_dtype_boolean(section='TRAINING', key='USE_CENTRE_LOSS')
    print(f"using Centre loss: {use_centre_loss}.")
    if use_centre_loss:
        feature_alignment_loss = CenterLoss(feat_dim=model.lin1.in_features, num_classes=4,
                                            # Since we have 4 brain clusters
                                            ).to(device)
        optimizer_centre_loss = Adam(feature_alignment_loss.parameters(), lr=lr, weight_decay=weight_decay)
    return contras_trainer, mixup_trainer, mixup_train_loader, feature_alignment_loss, optimizer_centre_loss, train_loader


def get_dataset_and_auxiliary_loss(dataset, test_idx, train_idx, val_idx, no_aug):
    is_node_level_dataset = get_configurations_dtype_boolean(section='SETUP', key='PERFORM_NODE_LEVEL_PREDICTION')
    print(f"Is node level dataset: {is_node_level_dataset}")
    training_set_reduction_factor = get_configurations_dtype_float(section='TRAINING',
                                                                   key='TRAINING_SIZE_REDUCTION_FACTOR',
                                                                   default_value=1)
    if not training_set_reduction_factor == 1:
        print("Using reduced training size.")
        print(f"original size was {len(train_idx)}")
        train_idx = np.random.choice(train_idx, size=int(len(train_idx) * training_set_reduction_factor))
        print(f"new size is {len(train_idx)}")

    if is_node_level_dataset:
        # Index 0 represents non-augmented graph while 1 represents augmented graph
        if no_aug:
            train_dataset = [(dataset[idx.item()][0], dataset[idx.item()][2]) for idx in train_idx]
        else:
            train_dataset = [(dataset[idx.item()][1], dataset[idx.item()][2]) for idx in train_idx]
        test_dataset = [(dataset[idx.item()][0], dataset[idx.item()][2]) for idx in test_idx]
        val_dataset = [(dataset[idx.item()][0], dataset[idx.item()][2]) for idx in val_idx]
        criterion_vol_regr = nn.L1Loss()
        # criterion_vol_regr = lambda a, b: (a - b).abs().sum()
    else:
        # Should we apply it to linear models?
        train_dataset = [dataset[idx.item()] for idx in train_idx]
        test_dataset = [dataset[idx.item()] for idx in test_idx]
        val_dataset = [dataset[idx.item()] for idx in val_idx]
        criterion_vol_regr = None
    return criterion_vol_regr, test_dataset, train_dataset, val_dataset


def create_weighted_sampler(class_weights, dataset):
    # Trying the weighted sampler idea rather than the loss re-weighting
    # We need to assign a weight to each of the samples in our dataset
    per_sample_wt = [0] * len(dataset)
    for idx, (graph, label) in enumerate(dataset):
        cls_wt = class_weights[label]  # Finding the weight associated with our given label
        per_sample_wt[idx] = cls_wt  # Assigning this class-based weight to our sample
    weighted_sampler = WeightedRandomSampler(per_sample_wt, num_samples=len(per_sample_wt), replacement=False)
    return weighted_sampler


def sanity_check(train_indices, val_indices, test_indices):
    per_split_result = []
    for idx in range(len(train_indices)):
        train_set = set(train_indices[idx].numpy().tolist())
        val_set = set(val_indices[idx].numpy().tolist())
        test_set = set(test_indices[idx].numpy().tolist())
        per_split_result.append(
            all([len(train_set.intersection(val_set)) == 0, len(val_set.intersection(test_set)) == 0,
                 len(train_set.intersection(test_set)) == 0]))
    return all(per_split_result)


def k_fold(dataset, folds):
    # We define the splits once and re-use them.
    # This is one way of reducing possible stochasticity.
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    create_fresh_split = True
    if os.path.exists(k_fold_split_path):
        num_folds = pickle.load(open(os.path.join(k_fold_split_path, "num_splits.pkl"), 'rb'))
        if num_folds == folds:
            print("Using a pre-defined k fold split. Done for easy reproducibility.")
            train_indices = pickle.load(open(os.path.join(k_fold_split_path, "train_indices.pkl"), 'rb'))
            val_indices = pickle.load(open(os.path.join(k_fold_split_path, "val_indices.pkl"), 'rb'))
            test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'rb'))
            create_fresh_split = False
        else:
            print("Number of folds differ. Creating a fresh train-val-test split")
    # Create a new data split
    if create_fresh_split:
        print("Generating a new k-fold split")
        skf = StratifiedKFold(folds, shuffle=True, random_state=42)

        test_indices, train_indices = [], []
        # https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
        # for _, idx in skf.split(torch.zeros(len(dataset)), dataset.graph_catogory_label.cpu().numpy().tolist()):
        label_and_graph_size = [str(y) + "_" + str(size) for y, size in
                                zip(dataset.y, dataset.graph_catogory_label.cpu().numpy().tolist())]
        for _, idx in skf.split(torch.zeros(len(dataset)), label_and_graph_size):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

        val_indices = [test_indices[i - 1] for i in range(folds)]
        # 70-20-10 attempt
        # test_indices = [torch.cat((test_indices[i], test_indices[i - 1])) for i in range(folds)]
        # val_indices = [torch.cat((test_indices[i - 2], test_indices[i - 3])) for i in range(folds)]

        for i in range(folds):
            train_mask = torch.ones(len(dataset), dtype=torch.bool)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
        # Now, let us go ahead and save these values
        assert sanity_check(train_indices, val_indices, test_indices), "Something wrong with the splits"
        os.makedirs(k_fold_split_path, exist_ok=False)  # Exists ok is not fine here.
        pickle.dump(train_indices, open(os.path.join(k_fold_split_path, "train_indices.pkl"), 'wb'))
        pickle.dump(val_indices, open(os.path.join(k_fold_split_path, "val_indices.pkl"), 'wb'))
        pickle.dump(test_indices, open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'wb'))
        pickle.dump(folds, open(os.path.join(k_fold_split_path, "num_splits.pkl"), 'wb'))

    return train_indices, val_indices, test_indices


def train_val_loop(criterion, enc, epochs, model, optimizer, roc_auc,
                   train_loader, train_writer, val_loader, val_losses,
                   val_writer, log_dir, fold, criterion_vol_regr=None, dataset_refresh_metadata=None,
                   mixup_trainer=None, mixup_train_loader=None, contras_trainer=None,
                   feature_alignment_loss_and_optim=None, scheduler=None, tune_obj=None):
    best_val_roc = 0
    min_loss = 1e10
    best_model_save_epoch = -1
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train(model=model, optimizer=optimizer, loader=train_loader, criterion=criterion, epoch=epoch,
                           train_writer=train_writer, criterion_vol_regr=criterion_vol_regr,
                           mixup_trainer=mixup_trainer, mixup_train_loader=mixup_train_loader,
                           contras_trainer=contras_trainer,
                           feature_alignment_loss_and_optim=feature_alignment_loss_and_optim)
        val_loss = eval_loss(model=model, loader=val_loader, criterion=criterion, epoch=epoch, writer=val_writer,
                             criterion_vol_regr=criterion_vol_regr,
                             plotting_offset=len(val_loader.dataset))
        val_roc = eval_roc_auc(model=model, loader=val_loader, enc=enc, epoch=epoch, writer=val_writer,
                               criterion_vol_regr=criterion_vol_regr)
        train_roc = eval_roc_auc(model=model, loader=train_loader, enc=enc, epoch=epoch, writer=train_writer,
                                 criterion_vol_regr=criterion_vol_regr)
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_roc': val_roc,
            'train_roc': train_roc
        }
        # Appending the results for selecting best model
        val_losses.append(val_loss)
        roc_auc.append(val_roc)

        if val_roc > best_val_roc:
            best_model_save_epoch = epoch
            best_val_roc = val_roc
            torch.save(model.state_dict(), os.path.join(log_dir, f"{model}_{fold}.pth"))

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, f"{model}_loss_{fold}.pth"))

        if scheduler is not None:
            scheduler.step(val_loss)
        # Our training dataset is a list with data objects.
        # So, in order to get more augmentations, we have to "reload" the list.
        # This ensures the __get_item__ is called repeatedly and thus, we get more augmentations.
        if dataset_refresh_metadata is not None:
            shuffle_dataset(loader=train_loader, dataset_refresh_metadata=dataset_refresh_metadata,
                            mixup_train_loader=mixup_train_loader)
    print(f"Best model saved at {best_model_save_epoch}")
    if tune_obj is not None:
        tune_obj.report(roc_auc=val_roc)


def entropy_loss(weights):
    entropy_loss = torch.tensor([0], device=device)
    if weights is not None:
        EPS = 1e-15
        ent = -weights * torch.log(weights + EPS) - (1 - weights) * torch.log(1 - weights + EPS)
        entropy_loss = ent.mean()
    return entropy_loss


def sparsity_loss(weights):
    return 0.1 * weights.sum() if weights is not None else torch.tensor([0.]).to(device)


def execute_graph_classification_epoch(criterion, data, label, model, optimizer, total_loss, scheduler):
    out = model(data)
    loss = criterion(out['graph_pred'], label.view(-1))
    loss.backward()
    total_loss += loss.item()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return total_loss


def execute_node_and_graph_classification_epoch(criterion, criterion_vol_regr, data, epoch,
                                                idx, label, dataset, model, optimizer,
                                                total_loss, train_writer, scheduler=None):
    out = model(data)
    graph_cls_loss = criterion(out['graph_pred'], label.view(-1))
    new_lesion_vol_regr_loss = criterion_vol_regr(out['graph_vol'], data.graph_vol)
    node_regr_loss = criterion_vol_regr(out['node_vol'], data.node_labels)
    # Including node alignment loss
    weights = out.get('weight_coeff', None)
    # entropy_loss_conn = entropy_loss(weights=weights)
    sparsity_loss_conn = sparsity_loss(weights=weights)
    # Include the same for features
    loss = graph_cls_loss + node_regr_loss + new_lesion_vol_regr_loss + sparsity_loss_conn
    loss.backward()
    total_loss += loss.item()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    # Let us also plot per iteration loss to get some fine-grained information
    train_writer.add_scalar('vol_regr_loss', node_regr_loss.item(), global_step=epoch * len(dataset) + idx)
    train_writer.add_scalar('graph_cls_loss', graph_cls_loss.item(), global_step=epoch * len(dataset) + idx)
    train_writer.add_scalar('graph_regr_loss', new_lesion_vol_regr_loss.item(),
                            global_step=epoch * len(dataset) + idx)
    train_writer.add_scalar('sparsity_loss_conn', sparsity_loss_conn.item(),
                            global_step=epoch * len(dataset) + idx)
    return total_loss


def execute_node_and_graph_classification_with_centre_loss_epoch(criterion, criterion_vol_regr, data, epoch,
                                                                 feature_alignment_loss, centre_loss_optim, idx, label,
                                                                 loader, model, optimizer,
                                                                 total_loss, train_writer, scheduler):
    out = model(data)
    graph_cls_loss = criterion(out['graph_pred'], label.view(-1))
    new_lesion_vol_regr_loss = criterion_vol_regr(out['graph_vol'], data.graph_vol)
    node_regr_loss = criterion_vol_regr(out['node_vol'], data.node_labels)
    centre_loss_optim.zero_grad()
    contras_centre_loss = feature_alignment_loss(model.x, data.cluster.squeeze() - 1,
                                                 batch_size=out['graph_pred'].shape[0])  # label)
    loss = graph_cls_loss + node_regr_loss + new_lesion_vol_regr_loss + contras_centre_loss
    loss.backward()
    total_loss += loss.item()
    optimizer.step()
    centre_loss_optim.step()
    if scheduler is not None:
        scheduler.step()
    # Let us also plot per iteration loss to get some fine-grained information
    train_writer.add_scalar('vol_regr_loss', node_regr_loss.item(), global_step=epoch * len(loader) + idx)
    train_writer.add_scalar('graph_cls_loss', graph_cls_loss.item(), global_step=epoch * len(loader) + idx)
    train_writer.add_scalar('centre_loss', contras_centre_loss.item(), global_step=epoch * len(loader) + idx)
    train_writer.add_scalar('graph_regr_loss', new_lesion_vol_regr_loss.item(),
                            global_step=epoch * len(loader) + idx)
    return total_loss


def execute_contrastive_training_epoch(contras_trainer, criterion_vol_regr, epoch, idx, mixup_train_loader, model,
                                       optimizer, total_loss, train_writer, scheduler):
    balanced_graph, balanced_labels = next(iter(mixup_train_loader))
    balanced_graph, balanced_labels = balanced_graph.to(device), balanced_labels.to(device)
    total_loss = contras_trainer.execute_contras_training(criterion_vol_regr=criterion_vol_regr,
                                                          data=balanced_graph, epoch=epoch, idx=idx,
                                                          label=balanced_labels, loader=mixup_train_loader,
                                                          model=model,
                                                          optimizer=optimizer, total_loss=total_loss,
                                                          train_writer=train_writer,
                                                          scheduler=scheduler)
    return total_loss


def execute_mixup_epoch(criterion_vol_regr, data, epoch, feature_alignment_loss_and_optim, idx, label, loader,
                        mixup_train_loader, mixup_trainer, scheduler):
    balanced_graph, balanced_labels = next(iter(mixup_train_loader))
    balanced_graph, balanced_labels = balanced_graph.to(device), balanced_labels.to(device)
    mixup_trainer.run_mixup_training(graph=data, idx=epoch * len(loader) + idx, label=label,
                                     graph2=balanced_graph, label2=balanced_labels,
                                     criterion_vol_regr=criterion_vol_regr,
                                     scheduler=scheduler)


def train(model, optimizer, loader, criterion, epoch, train_writer, criterion_vol_regr=None, mixup_trainer=None,
          mixup_train_loader=None, contras_trainer=None, feature_alignment_loss_and_optim=None, scheduler=None):
    model.train()
    is_node_and_graph_cls = criterion_vol_regr is not None
    feature_alignment_loss, centre_loss_optim = feature_alignment_loss_and_optim
    # Some information needed for logging on tensorboard
    total_loss = 0
    for idx, (data, label) in enumerate(loader):
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        # Apply mixup
        if mixup_trainer is not None:
            execute_mixup_epoch(criterion_vol_regr=criterion_vol_regr, data=data, epoch=epoch,
                                feature_alignment_loss_and_optim=feature_alignment_loss_and_optim, idx=idx, label=label,
                                loader=loader,
                                mixup_train_loader=mixup_train_loader, mixup_trainer=mixup_trainer, scheduler=scheduler)
        elif contras_trainer is not None:
            total_loss = execute_contrastive_training_epoch(contras_trainer, criterion_vol_regr, epoch, idx,
                                                            mixup_train_loader, model, optimizer, total_loss,
                                                            train_writer, scheduler=scheduler)
        # This is the centre loss section
        elif feature_alignment_loss is not None:
            total_loss = execute_node_and_graph_classification_with_centre_loss_epoch(criterion, criterion_vol_regr,
                                                                                      data, epoch,
                                                                                      feature_alignment_loss,
                                                                                      centre_loss_optim, idx, label,
                                                                                      loader, model, optimizer,
                                                                                      total_loss, train_writer,
                                                                                      scheduler=scheduler)
        elif is_node_and_graph_cls:
            total_loss = execute_node_and_graph_classification_epoch(criterion, criterion_vol_regr, data, epoch,
                                                                     idx, label,
                                                                     loader, model, optimizer, total_loss,
                                                                     train_writer,
                                                                     scheduler=scheduler)
        else:
            total_loss = execute_graph_classification_epoch(criterion, data, label, model, optimizer, total_loss,
                                                            scheduler=scheduler)

    avg_train_loss = total_loss / len(loader)
    train_writer.add_scalar('loss', avg_train_loss, global_step=epoch)
    return avg_train_loss


# Eval utils

# We decide for three possible graph sizes
graph_size_small = CustomDictKey(key_name=f"less than {smallness_threshold}", key_iden=0)
graph_size_large = CustomDictKey(key_name=f"more than {smallness_threshold}", key_iden=1)


def min_max_normalize(vector, factor):
    vector = factor * (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return vector


def normalize_features(features):
    for ii in range(np.shape(features)[1]):
        features[:, ii] = min_max_normalize(features[:, ii], 1)


def eval_acc(model, loader, criterion_vol_regr, writer=None):
    model.eval()
    correct = 0
    out_feat = torch.FloatTensor().to(device)
    outGT = torch.FloatTensor().to(device)
    is_node_and_graph_cls = criterion_vol_regr is not None
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            out = model(data)
            if is_node_and_graph_cls:
                pred = out['graph_pred'].max(1)[1]
                out_feat = torch.cat((out_feat, model.graph_level_feat), 0)
                outGT = torch.cat((outGT, labels), 0)
            else:
                pred = out['graph_pred'].max(1)[1]
        correct += pred.eq(labels.view(-1)).sum().item()
    if writer is not None:
        return correct / len(loader.dataset), out_feat, outGT
    return correct / len(loader.dataset)


def eval_acc_with_confusion_matrix(model, dataset, criterion_vol_regr=None):
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    correct = 0
    for data, labels in dataset:
        batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
        ptr = torch.tensor([0, data.x.shape[0]], dtype=int, device=data.x.device)
        data.batch = batch
        data.ptr = ptr
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            out = model(data)  # Ignoring the node & regr component for the time being
            pred = out['graph_pred'].max(1)[1]
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, labels), 0)
        # correct += balanced_accuracy_score(labels.view(-1).cpu().numpy(), pred.cpu().numpy())
        correct += pred.eq(labels.view(-1)).sum().item()
    confusion_mat = compute_confusion_matrix(gt=outGT, predictions=outPRED, is_prediction=True)
    return correct / len(dataset), confusion_mat


@torch.no_grad()
def eval_roc_auc(model, loader, enc, epoch=0, writer=None, criterion_vol_regr=None):
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    is_node_and_graph_cls = criterion_vol_regr is not None
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            out = model(data)  # Ignoring the node & regr component for the time being
        outPRED = torch.cat((outPRED, out['graph_pred']), 0)
        outGT = torch.cat((outGT, labels), 0)
    predictions = torch.softmax(outPRED, dim=1)
    predictions, target = predictions.cpu().numpy(), outGT.cpu().numpy()
    # Encoder is callable.
    # Hence, we execute callable which returns the self.encoder instance
    target_one_hot = enc().transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    roc_auc_value = roc_auc_score(target_one_hot, predictions, average='weighted')
    if writer is not None:
        writer.add_scalar('roc', roc_auc_value, global_step=epoch)
    return roc_auc_value


def eval_loss(model, loader, criterion, epoch, writer, criterion_vol_regr=None, plotting_offset=-1):
    model.eval()
    is_node_and_graph_cls = criterion_vol_regr is not None
    # Some information needed for logging on tensorboard
    total_loss = 0
    nodes_on = []
    plotting_offset = len(loader) if plotting_offset == -1 else plotting_offset
    for idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            if is_node_and_graph_cls:
                out = model(data)  # Ignoring the regression part for the time being
                node_regr_loss = criterion_vol_regr(out['node_vol'], data.node_labels)
                graph_cls_loss = criterion(out['graph_pred'], labels.view(-1))
                new_lesion_vol_regr_loss = criterion_vol_regr(out['graph_vol'], data.graph_vol)
                writer.add_scalar('vol_regr_loss', node_regr_loss.item(), global_step=epoch * plotting_offset + idx)
                writer.add_scalar('graph_cls_loss', graph_cls_loss.item(), global_step=epoch * plotting_offset + idx)
                writer.add_scalar('graph_regr_loss', new_lesion_vol_regr_loss.item(),
                                  global_step=epoch * plotting_offset + idx)
                # loss = graph_cls_loss + node_regr_loss + new_lesion_vol_regr_loss
                loss = graph_cls_loss
                total_loss += loss
                nodes_on.append(out.get('on_ratio', torch.tensor(1)).item())
            else:
                out = model(data)
                loss = criterion(out['graph_pred'], labels.view(-1)).item()
                total_loss += loss
    avg_val_loss = total_loss / len(loader)
    writer.add_scalar('loss', avg_val_loss, global_step=epoch)
    writer.add_scalar('on_ratio', sum(nodes_on) / len(nodes_on), global_step=epoch)
    return avg_val_loss


def decide_graph_category_based_on_size(graph_size):
    if graph_size <= smallness_threshold:
        return graph_size_small
    else:
        return graph_size_large


def eval_graph_len_acc(model, dataset, criterion_vol_regr=None):
    model.eval()
    # A dictionary of the format
    # {
    #   size_a:
    #           [
    #           (tensor([prob_0, prob_1]), gt1),
    #           (tensor([prob_0, prob_1]), gt2),
    #           (tensor([prob_0, prob_1]), gt3),
    #           ]
    #  size_b:
    #           ...
    # }
    size_cm_dict = defaultdict(list)
    is_node_and_graph_cls = criterion_vol_regr is not None
    correct = 0
    for idx in range(len(dataset)):
        graph, graph_label = dataset[idx]
        # We need to add a dummy batch attribute to our graph.
        batch = torch.zeros(graph.x.shape[0], dtype=int, device=graph.x.device)
        ptr = torch.tensor([0, graph.x.shape[0]], dtype=int, device=graph.x.device)
        graph.batch = batch
        graph.ptr = ptr
        graph, graph_label = graph.to(device), graph_label.to(device)
        graph_size_categ = decide_graph_category_based_on_size(graph.x.size(0))
        with torch.no_grad():
            out = model(graph)  # Ignoring the node & regr component for the time being
            pred = out['graph_pred'].max(1)[1]
        size_cm_dict[graph_size_categ].append([out['graph_pred'], graph_label.item()])
        correct += pred.item() == graph_label.item()
    return correct / len(dataset), size_cm_dict


def _compute_roc_for_graph_size(predictions, gt, enc):
    predictions, gt = predictions.cpu().numpy(), gt.cpu().numpy()
    target_one_hot = enc().transform(gt.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    roc_auc_value = roc_auc_score(target_one_hot, predictions, average='weighted')
    return roc_auc_value


def compute_confusion_matrix(gt, predictions, is_prediction=False):
    if not is_prediction:
        predicted_label = predictions.max(1)[1]
    else:
        predicted_label = predictions
    gt, predicted_label = gt.cpu().numpy(), predicted_label.cpu().numpy()
    return confusion_matrix(gt, predicted_label)


def plot_results_based_on_graph_size(size_cm_dict, filename_acc, filename_roc, model_type=None, output_dir=None, fold=0,
                                     is_plotting_enabled=True, split_acc_based_on_labels=False):
    accuracy_dictionary, roc_dictionary, cm_dict = {}, {}, {}
    enc = LabelEncoder()
    skip_this_round = False
    for graph_size, model_predictions_list in size_cm_dict.items():
        predictions, gt = torch.concat([x[0] for x in model_predictions_list]), torch.stack(
            [torch.as_tensor(x[1]) for x in model_predictions_list])
        gt = gt.to(predictions.device)
        if split_acc_based_on_labels:
            zero_acc, ones_acc = compute_label_wise_acc(gt, predictions)
            accuracy_dictionary[graph_size] = (zero_acc, ones_acc)
        else:
            acc = compute_acc(gt, predictions)
            accuracy_dictionary[graph_size] = acc
        cm = compute_confusion_matrix(gt, predictions)
        cm_dict[graph_size] = cm
        # ROC is not defined in case the gt is all 0s or all 1s.
        # So, we would use the accuracy in these cases to give us an indication
        try:
            roc = _compute_roc_for_graph_size(predictions, gt, enc)
            roc_dictionary[graph_size] = roc
        except ValueError:
            print(f"roc not defined since gt is {gt}")
            skip_this_round = True
            return skip_this_round, None, None
    if is_plotting_enabled:
        plot_bar_plot(dictionary_to_plot=accuracy_dictionary, y_label='accuracy',
                      title=f'{model_type} accuracy vs. size',
                      filename=filename_acc, output_dir=output_dir)
        plot_bar_plot(dictionary_to_plot=roc_dictionary, y_label='roc', title=f'{model_type} roc vs. size',
                      filename=filename_roc, output_dir=output_dir, color='b')
    if output_dir is not None:
        cm_save_path = os.path.join(output_dir, f'cm{fold}.pkl')
        pickle.dump(cm_dict, open(cm_save_path, 'wb'))
    return skip_this_round, accuracy_dictionary, roc_dictionary


def plot_avg_of_dictionary(input_dict, y_label, filename, output_dir, color):
    """

    :param input_dict: A dictionary with string key and a list of values to reduce
    :param y_label: plot label
    :param filename: filename to save the plot
    :param output_dir: directory location for saving plots
    :param color: color of bar plot
    :return: None
    """
    avg_dict = {}
    for key, item_list in input_dict.items():
        avg_dict[key] = sum(item_list) / len(item_list)
    plot_bar_plot(dictionary_to_plot=avg_dict, y_label=y_label, title=f'{filename} {y_label} vs. size',
                  filename=filename, output_dir=output_dir, color=color)


def compute_acc(gt, predictions):
    predicted_label = predictions.max(1)[1]
    acc = predicted_label.eq(gt.view(-1)).sum().item() / predictions.shape[0]
    return acc


def eval_regr_loss(model, loader, criterion_vol_regr):
    # The computation  is defined only when we are working with regression target
    if criterion_vol_regr is None:
        return -1
    # Let us now compute the regression-loss
    new_lesion_vol_regr_loss = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            out = model(data)
            new_lesion_vol_regr_loss += criterion_vol_regr(out['graph_vol'], data.graph_vol)
    avg_new_lesion_vol_regr_loss = new_lesion_vol_regr_loss / len(loader)
    return avg_new_lesion_vol_regr_loss


def compute_label_wise_acc(gt, predictions):
    zero_indices = torch.where(gt == 0)[0]
    ones_indices = torch.where(gt == 1)[0]
    predicted_label = predictions.max(1)[1]
    zero_acc = predicted_label[zero_indices].eq(gt[zero_indices].view(-1)).sum().item() / zero_indices.shape[0]
    ones_acc = predicted_label[ones_indices].eq(gt[ones_indices].view(-1)).sum().item() / ones_indices.shape[0]
    return zero_acc, ones_acc


def pretty_print_avg_dictionary(input_dict):
    for key, values in input_dict.items():
        print(f"{key}---------{sum(values) / len(values)}")


def print_custom_avg_of_dictionary(input_dict):
    """

    :param input_dict: A dictionary with string key and a list of values to reduce
    :param y_label: plot label
    :param filename: filename to save the plot
    :param output_dir: directory location for saving plots
    :param color: color of bar plot
    :return: None
    """
    # The input dictionary has
    # {"large": [acc_lab0, acc_lab1]}
    avg_dict = {}
    for key, nested_list in input_dict.items():
        list_zeros, list_ones = [], []
        for x in nested_list:
            list_zeros.append(x[0])
            list_ones.append(x[1])
        avg_dict[f"{key}_0"] = sum(list_zeros) / len(list_zeros)
        avg_dict[f"{key}_1"] = sum(list_ones) / len(list_ones)
    for key, value in avg_dict.items():
        print(f"{key} has the accuracy {value}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import torch.nn.functional as F


class FocalLoss(torch.nn.Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        print(f"Using Focal loss with alpha {self.weight} and gamma {self.gamma}")

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


# Used from -> https://github.com/FLHonker/Losses-in-image-classification-task
class CenterLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, lambda_c=2.0):
        super(CenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label, batch_size=None):
        if batch_size is None:
            batch_size = feat.shape[0]
        expanded_centers = self.centers.index_select(dim=0, index=label)
        intra_distances = feat.dist(expanded_centers)
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
        return loss


class ContrastiveCenterLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, lambda_c=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]

        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_feat = feat.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_feat - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, label.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / \
               (inter_distances + epsilon) / 0.1

        return loss


if __name__ == '__main__':
    x = torch.randn((2, 5))
    y = torch.as_tensor([0, 1])
    criterion = CenterLoss(feat_dim=5, num_classes=2, weights=torch.as_tensor([0.5, 1]))
    print(criterion(x, y))
    criterion = ContrastiveCenterLoss(feat_dim=5, num_classes=2)
    print(criterion(x, y))
