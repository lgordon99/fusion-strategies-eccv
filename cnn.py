# imports
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sys import argv
from torchsummary import summary
from torchvision.models import resnet50
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import utils

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params, 'trainable_params': trainable_params}

class EarlyFusion(nn.Module):
    def __init__(self, device, modalities, num_classes):
        super().__init__() # initializes nn.Module class

        self.device = device
        self.modalities = modalities
        self.model = resnet50(weights='DEFAULT') # imports a ResNet-50 CNN pretrained on ImageNet-1k v2
        original_weights = self.model.conv1.weight.clone()
        self.model.conv1 = torch.nn.Conv2d(in_channels=5,
                                           out_channels=self.model.conv1.out_channels,
                                           kernel_size=self.model.conv1.kernel_size,
                                           stride=self.model.conv1.stride,
                                           padding=self.model.conv1.padding,
                                           bias=self.model.conv1.bias)

        with torch.no_grad():
            self.model.conv1.weight[:, 1:4] = original_weights # keeps the weights of the middle three channels the same
            mean_original_weights = torch.mean(original_weights, dim=1)
            self.model.conv1.weight[:, 0], self.model.conv1.weight[:, 4] = mean_original_weights, mean_original_weights
            self.model.conv1.weight *= 3/5

        num_features = self.model.fc.in_features # input to final layer (2048)
        print('num_features', num_features)
        self.model.fc = nn.Linear(num_features, num_classes) # sets the output size to the number of classes
        print(f'Model: {count_parameters(self.model)}')

    def forward(self, images):
        concatenated_images = torch.tensor(np.array([np.concatenate([image[modality] for modality in self.modalities], axis=0) for image in images])).float().to(self.device)

        return self.model(concatenated_images)

class FeatureExtractor1Channel(nn.Module):
    def __init__(self, num_features_extracted=256):
        super().__init__() # initializes nn.Module class
        self.feature_extractor_1_channel = resnet50(weights='DEFAULT') # imports a ResNet-50 CNN pretrained on ImageNet-1k v2
        original_weights = self.feature_extractor_1_channel.conv1.weight.clone()
        self.feature_extractor_1_channel.conv1 = torch.nn.Conv2d(in_channels=1,
                                                                 out_channels=self.feature_extractor_1_channel.conv1.out_channels,
                                                                 kernel_size=self.feature_extractor_1_channel.conv1.kernel_size,
                                                                 stride=self.feature_extractor_1_channel.conv1.stride,
                                                                 padding=self.feature_extractor_1_channel.conv1.padding,
                                                                 bias=self.feature_extractor_1_channel.conv1.bias)

        with torch.no_grad():
            mean_original_weights = torch.mean(original_weights, dim=1)
            self.feature_extractor_1_channel.conv1.weight[:, 0] = mean_original_weights
            self.feature_extractor_1_channel.conv1.weight *= 3

        num_features = self.feature_extractor_1_channel.fc.in_features # input to final layer
        self.feature_extractor_1_channel.fc = nn.Linear(num_features, num_features_extracted) # sets the output size to the number of features to be extracted
        print(f'1 channel extractor: {count_parameters(self.feature_extractor_1_channel)}')

    def forward(self, images):
        return self.feature_extractor_1_channel(images)

class FeatureExtractor3Channel(nn.Module):
    def __init__(self, num_features_extracted=256):
        super().__init__() # initializes nn.Module class
        self.feature_extractor_3_channel = resnet50(weights='DEFAULT')
        num_features = self.feature_extractor_3_channel.fc.in_features # input to final layer
        self.feature_extractor_3_channel.fc = nn.Linear(num_features, num_features_extracted) # sets the output size to the number of features to be extracted
        print(f'3 channel extractor: {count_parameters(self.feature_extractor_3_channel)}')

    def forward(self, images):
        return self.feature_extractor_3_channel(images)

class LateFusion(nn.Module): # weights for each class for the features of the modalities
    def __init__(self, device, modalities, num_classes, num_features_extracted=256):
        super().__init__() # initializes nn.Module class
        self.device = device
        self.modalities = modalities
        self.feature_extractors = nn.ModuleDict({'thermal': FeatureExtractor1Channel(), 'rgb': FeatureExtractor3Channel(), 'lidar': FeatureExtractor1Channel()})
        self.fusion_layer = nn.Linear(len(self.modalities) * num_features_extracted, num_classes)
        print(f'Fusion layer: {count_parameters(self.fusion_layer)}')

    def forward(self, images):
        features_extracted = torch.cat([self.feature_extractors[modality](torch.tensor(np.array([image[modality] for image in images])).float().to(self.device)) for modality in self.modalities], dim=1)
        output = self.fusion_layer(features_extracted)

        return output

class Classifier(nn.Module):
    def __init__(self, num_classes, num_features_extracted=256):
        super().__init__() # initializes nn.Module class
        self.classifier = nn.Linear(num_features_extracted, num_classes)
        print(f'Classifier: {count_parameters(self.classifier)}')

    def forward(self, features_extracted):
        return self.classifier(features_extracted)

class GatingNetwork(nn.Module):
    def __init__(self, num_modalities, num_features_extracted=256):
        super().__init__() # initializes nn.Module class

        self.gating_network = nn.Sequential(nn.Linear(num_modalities * num_features_extracted, num_features_extracted),
                                            nn.ReLU(),
                                            nn.Linear(num_features_extracted, num_modalities),
                                            nn.Softmax(dim=1))
        print(f'Gating network: {count_parameters(self.gating_network)}')

    def forward(self, concatenated_features):
        output = self.gating_network(concatenated_features)

        return output

class MixtureOfExperts(nn.Module):
    def __init__(self, device, modalities, num_classes):
        super().__init__() # initializes nn.Module class
        self.device = device
        self.modalities = modalities
        self.feature_extractors = nn.ModuleDict({'thermal': FeatureExtractor1Channel(), 'rgb': FeatureExtractor3Channel(), 'lidar': FeatureExtractor1Channel()})
        self.classifiers = nn.ModuleDict({'thermal': Classifier(num_classes), 'rgb': Classifier(num_classes), 'lidar': Classifier(num_classes)})
        self.gating_network = GatingNetwork(num_modalities=len(self.modalities))

    def forward(self, images, test=False):
        features_extracted = {modality: self.feature_extractors[modality](torch.tensor(np.array([image[modality] for image in images])).float().to(self.device)) for modality in self.modalities}
        classifier_outputs = torch.stack([self.classifiers[modality](features_extracted[modality]) for modality in self.modalities], dim=1)
        concatenated_features = torch.cat([features_extracted[modality] for modality in self.modalities], dim=1)
        gating_weights = self.gating_network(concatenated_features).unsqueeze(2)
        classifier_probabilities = torch.nn.functional.softmax(classifier_outputs, dim=2)
        output = torch.log(torch.sum(gating_weights * classifier_probabilities, dim=1))

        if test:
            return output, gating_weights

        return output

class CNN:
    def __init__(self, setting, site, trial=None, lr=0.001, hp_tuning=False):
        self.setting = setting
        self.site_dir = utils.get_site_dir(site)
        self.trial = trial
        self.constants = utils.process_yaml('constants.yaml')
        self.classes = ['empty', 'midden', 'mound', 'water']
        self.identifiers = np.load(f'{self.site_dir}/identifiers.npy')
        self.identifier_matrix = np.load(f'{self.site_dir}/identifier_matrix.npy')
        self.labels = np.load(f'{self.site_dir}/labels.npy')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 64
        self.lr = lr
        self.hp_tuning = hp_tuning
        self.criterion = nn.NLLLoss() if setting == 'mixture_of_experts' else nn.CrossEntropyLoss()
        self.modalities = ['thermal', 'rgb', 'lidar']
        model_dir = utils.get_model_dir()
        seed = 0 if hp_tuning or trial is None else trial
        random.seed(seed)
        os.makedirs(f'{model_dir}/models/{setting}', exist_ok=True)

        print(f'Setting = {setting}')
        print(f'Random seed = {seed}')
        print(f'Learning rate = {lr}')
        print(f'Hyperparameter tuning = {hp_tuning}')
        print(f'Using {torch.cuda.device_count()} GPU(s)')

        if hp_tuning:
            os.makedirs(f'hp_tuning/{setting}', exist_ok=True)
        else:
            self.model_save_path = f'{model_dir}/models/{setting}/{setting}' if trial is None else f'{model_dir}/models/{setting}/{setting}_{trial}'

        train_identifiers = [identifier for identifier in self.identifier_matrix.T[:50].T.ravel() if identifier in self.identifiers] # identifiers in last 61 columns
        train_indices = [np.where(self.identifiers == train_identifier)[0][0] for train_identifier in train_identifiers] # indices corresponding to those identifiers
        original_train_indices = np.array(train_indices.copy())
        max_nonempty_train_class_count = max([len(np.where(self.labels[train_indices] == self.classes.index(image_class))[0]) for image_class in self.classes[1:]])

        # print([len(np.where(self.labels[train_indices] == self.classes.index(image_class))[0]) for image_class in self.classes])
        print(f'Identifiers length = {len(self.identifiers)}')
        # print(f'Identifier matrix shape = {self.identifier_matrix.shape}')
        # print(f'Labels length = {len(self.labels)}')
        print(f'Class counts = {[len(self.labels[self.labels == self.classes.index(image_class)]) for image_class in self.classes]}')
        # print(f'Max non-empty train class count = {max_nonempty_train_class_count}')

        # undersample empty class
        while len(np.where(self.labels[train_indices] == 0)[0]) > max_nonempty_train_class_count: # more empty images than desired
            del train_indices[random.choice(np.where(self.labels[train_indices] == 0)[0])] # randomly remove one empty image
        # print('Undersampled empty class')

        # upsample non-empty classes
        for class_index in range(1, len(self.classes)):
            while len(np.where(self.labels[train_indices] == class_index)[0]) < max_nonempty_train_class_count:
                train_indices.append(random.choice(original_train_indices[np.where(self.labels[original_train_indices] == class_index)[0]]))
        # print('Upsampled non-empty classes')

        train_identifiers = self.identifiers[train_indices]
        val_identifiers = [identifier for identifier in self.identifier_matrix.T[50:59].T.ravel() if identifier in self.identifiers]
        test_identifiers = [identifier for identifier in self.identifier_matrix.T[59:].T.ravel() if identifier in self.identifiers]

        val_indices = [np.where(self.identifiers == val_identifier)[0][0] for val_identifier in val_identifiers]
        test_indices = [np.where(self.identifiers == test_identifier)[0][0] for test_identifier in test_identifiers]

        train_labels = self.labels[train_indices].ravel()
        val_labels = self.labels[val_indices].ravel()
        test_labels = self.labels[test_indices].ravel()

        print(f'Train class counts = {[len(train_labels[train_labels == self.classes.index(image_class)]) for image_class in self.classes]}')
        print(f'Validation class counts = {[len(val_labels[val_labels == self.classes.index(image_class)]) for image_class in self.classes]}')
        print(f'Test class counts = {[len(test_labels[test_labels == self.classes.index(image_class)]) for image_class in self.classes]}')
        # print(f'Train identifiers length = {len(train_identifiers)}')
        # print(f'Train indices length = {len(train_indices)}')
        # print(f'Validation identifiers length = {len(val_identifiers)}')
        # print(f'Validation indices length = {len(val_indices)}')
        # print(f'Test identifiers length = {len(test_identifiers)}')
        # print(f'Test indices length = {len(test_indices)}')

        if setting == 'early_fusion':
            tiles = {modality: np.load(f'{self.site_dir}/upsampled_tiles/{modality}_upsampled_tiles.npy').transpose(0, 3, 1, 2) if modality == 'rgb' else np.load(f'{self.site_dir}/upsampled_tiles/{modality}_upsampled_tiles.npy')[..., np.newaxis].transpose(0, 3, 1, 2) for modality in self.modalities}
            model = EarlyFusion(device=self.device, modalities=self.modalities, num_classes=len(self.classes))
        elif setting == 'late_fusion':
            tiles = {modality: np.load(f'{self.site_dir}/tiles/{modality}_tiles.npy').transpose(0, 3, 1, 2) if modality == 'rgb' else np.load(f'{self.site_dir}/tiles/{modality}_tiles.npy')[..., np.newaxis].transpose(0, 3, 1, 2) for modality in self.modalities}
            model = LateFusion(device=self.device, modalities=self.modalities, num_classes=len(self.classes))
        elif setting == 'mixture_of_experts':
            tiles = {modality: np.load(f'{self.site_dir}/tiles/{modality}_tiles.npy').transpose(0, 3, 1, 2) if modality == 'rgb' else np.load(f'{self.site_dir}/tiles/{modality}_tiles.npy')[..., np.newaxis].transpose(0, 3, 1, 2) for modality in self.modalities}
            model = MixtureOfExperts(device=self.device, modalities=self.modalities, num_classes=len(self.classes))

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        print(count_parameters(model))
        model.to(self.device)
        # print(summary(model, (3,224,224)))

        train_images = {modality: tiles[modality][train_indices] for modality in self.modalities}
        print(train_images['thermal'].shape, train_images['rgb'].shape, train_images['lidar'].shape)
        val_images = {modality: tiles[modality][val_indices] for modality in self.modalities}
        test_images = {modality: tiles[modality][test_indices] for modality in self.modalities}
        train_means = {modality: np.mean(train_images[modality], axis=(0, 2, 3)) for modality in self.modalities}
        train_stds = {modality: np.std(train_images[modality], axis=(0, 2, 3)) for modality in self.modalities}
        train_images = {modality: (train_images[modality] - train_means[modality][:, np.newaxis, np.newaxis]) / train_stds[modality][:, np.newaxis, np.newaxis] for modality in self.modalities}
        val_images = {modality: (val_images[modality] - train_means[modality][:, np.newaxis, np.newaxis]) / train_stds[modality][:, np.newaxis, np.newaxis] for modality in self.modalities}
        test_images = {modality: (test_images[modality] - train_means[modality][:, np.newaxis, np.newaxis]) / train_stds[modality][:, np.newaxis, np.newaxis] for modality in self.modalities}
        print(train_images['thermal'].shape, train_images['rgb'].shape, train_images['lidar'].shape)
        print(train_means)
        print(train_stds)

        train_loader = self.make_loader(train_images, train_labels, train_identifiers, self.batch_size)
        val_loader = self.make_loader(val_images, val_labels, val_identifiers, self.batch_size)
        test_loader = self.make_loader(test_images, test_labels, test_identifiers, self.batch_size)

        self.passive_train(model, train_loader, val_loader)

        if not hp_tuning:
            self.test(test_loader)

    def make_loader(self, images, labels, identifiers, batch_size):
        images = [{modality: images[modality][i] for modality in self.modalities} for i in range(len(labels))]
        print(len(images))
        data = list(map(list, zip(images, labels, identifiers))) # each image gets grouped with its label and identifier
        data = random.sample(data, len(data)) # shuffle the training data
        loader = {}
        image_batch = []
        label_batch = []
        identifier_batch = []
        batch_number = 0

        # batch the data
        for i in range(len(data) + 1):
            if (i % batch_size == 0 and i != 0) or (i == len(data)):
                loader[f'batch {batch_number}'] = {}
                loader[f'batch {batch_number}']['images'] = image_batch
                loader[f'batch {batch_number}']['labels'] = torch.tensor(np.array(label_batch))
                loader[f'batch {batch_number}']['identifiers'] = identifier_batch
                image_batch = []
                label_batch = []
                identifier_batch = []
                batch_number += 1

            if i != len(data):
                image_batch.append(data[i][0])
                label_batch.append(data[i][1])
                identifier_batch.append(data[i][2])

        return loader

    def passive_train(self, model, train_loader, val_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, capturable=True)
        counter = 0
        patience = 10
        best_val_auc = 0
        epoch = 0

        while counter <= patience:
            model.train()
            epoch += 1
            train_epoch_loss = 0
            val_y_true = torch.tensor([], dtype=torch.long, device=self.device)
            val_probabilities = torch.tensor([], dtype=torch.long, device=self.device)

            for batch in train_loader:
                images = train_loader[batch]['images']
                labels = train_loader[batch]['labels'].to(self.device)
                optimizer.zero_grad() # zeros the parameter gradients
                outputs = model(images) # forward pass
                loss = self.criterion(outputs, labels) # mean loss per item in batch
                loss.backward() # backward pass
                optimizer.step() # optimization
                train_epoch_loss += loss.item()

            train_epoch_loss /= len(train_loader)
            model.eval()

            with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
                for batch in val_loader:
                    images = val_loader[batch]['images']
                    labels = val_loader[batch]['labels'].to(self.device)
                    outputs = model(images) # forward pass
                    val_batch_probabilities = torch.exp(outputs) if self.setting == 'mixture_of_experts' else torch.nn.functional.softmax(outputs, dim=1) # applies softmax to the logits
                    val_y_true = torch.cat((val_y_true, labels), dim=0)
                    val_probabilities = torch.cat((val_probabilities, val_batch_probabilities), dim=0)

            val_y_true = val_y_true.cpu().numpy()
            val_probabilities = val_probabilities.cpu().numpy()
            val_roc_auc = roc_auc_score(val_y_true, val_probabilities, multi_class='ovr', average='macro')
            print(f'Epoch {epoch} Validation AUC = {round(val_roc_auc, 3)}, Counter = {counter}')

            if val_roc_auc > best_val_auc:
                best_val_auc = val_roc_auc
                counter = 0

                if not self.hp_tuning:
                    torch.save(model, self.model_save_path)
                    print('Saved!')
            else:
                counter += 1

        if self.hp_tuning:
            np.save(f'hp_tuning/{self.setting}/lr={self.lr}.npy', best_val_auc)

    def test(self, test_loader):
        model = torch.load(self.model_save_path)
        model.eval()
        y_true = torch.tensor([], dtype=torch.long, device=self.device)
        probabilities = torch.tensor([], dtype=torch.long, device=self.device)

        if self.setting == 'mixture_of_experts':
            all_labels = []
            all_gating_weights = []

        with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for batch in test_loader:
                images = test_loader[batch]['images']
                labels = test_loader[batch]['labels'].to(self.device)

                if self.setting == 'mixture_of_experts':
                    outputs, gating_weights = model(images, test=True)
                    all_labels += labels.detach().cpu().numpy().tolist()
                    all_gating_weights += np.squeeze(gating_weights.detach().cpu().numpy()).tolist()
                else:
                    outputs = model(images) # forward pass

                batch_probabilities = torch.exp(outputs) if self.setting == 'mixture_of_experts' else torch.nn.functional.softmax(outputs, dim=1) # applies softmax to the logits
                y_true = torch.cat((y_true, labels), dim=0)
                probabilities = torch.cat((probabilities, batch_probabilities), dim=0)

        y_pred = torch.argmax(probabilities, dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
        average_precision, average_recall, average_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        roc_auc = roc_auc_score(y_true, probabilities, multi_class='ovr', average='macro')

        results = {'precision': precision.tolist(),
                   'recall': recall.tolist(),
                   'f1': f1.tolist(),
                   'average_precision': average_precision,
                   'average_recall': average_recall,
                   'average_f1': average_f1,
                   'auc': roc_auc}

        if self.trial is not None:
            os.makedirs(f'results/{self.setting}', exist_ok=True)

            with open(f'results/{self.setting}/{self.setting}_{self.trial}.json', 'w') as json_file:
                json.dump(results, json_file, indent=4)

        if self.setting == 'mixture_of_experts':
            print(np.array(all_labels).shape)
            print(np.array(all_gating_weights).shape)

            gating_weights_dict = {'labels': all_labels, 'gating_weights': all_gating_weights}
            os.makedirs(f'moe_gating_weights', exist_ok=True)

            with open(f'moe_gating_weights/trial_{self.trial}.json', 'w') as json_file:
                json.dump(gating_weights_dict, json_file, indent=4)

        print(results)

if __name__ == '__main__':
    # CNN(setting='early_fusion', site='firestorm-3', lr=0.001, hp_tuning=False)
    # CNN(setting='late_fusion', site='firestorm-3', lr=0.001, hp_tuning=False)
    # CNN(setting='mixture_of_experts', site='firestorm-3', lr=0.0001, hp_tuning=False, trial=-1)

    hp_tuning = True if argv[4] == 'True' else False

    CNN(setting=argv[1], trial=int(argv[2]), site='firestorm-3', lr=float(argv[3]), hp_tuning=hp_tuning)
