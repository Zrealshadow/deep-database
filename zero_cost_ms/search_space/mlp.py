import itertools
import random
from copy import deepcopy
from typing import Generator
import torch
import torch.nn as nn

DEFAULT_LAYER_CHOICES_20 = [8, 16, 24, 32,  # 8
                            48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,  # 16
                            384, 512]


class MlpMacroCfg:
    def __init__(self, nfield: int, nfeat: int, nemb: int,
                 num_layers: int,
                 num_labels: int,
                 layer_choices: list):
        self.nfield = nfield
        self.nfeat = nfeat
        self.nemb = nemb
        self.layer_choices = layer_choices
        self.num_layers = num_layers
        self.num_labels = num_labels


class MlpMicroCfg:

    @classmethod
    def builder(cls, encoding: str):
        return MlpMicroCfg([int(ele) for ele in encoding.split("-")])

    def __init__(self, hidden_layer_list: list):
        self.hidden_layer_list = hidden_layer_list

    def __str__(self):
        return "-".join(str(x) for x in self.hidden_layer_list)


class Embedding(nn.Module):

    def __init__(self, nfeat, nemb):
        super().__init__()
        self.embedding = nn.Embedding(nfeat, nemb)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: dict):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    embeddings B*F*E
        """
        emb = self.embedding(x['id'])  # B*F*E
        return emb * x['value'].unsqueeze(2)  # B*F*E


class MLP(nn.Module):

    def __init__(self, ninput: int, hidden_layer_list: list, dropout_rate: float, noutput: int, use_bn: bool):
        super().__init__()
        """
        Args:
            ninput: number of input feature dim
            hidden_layer_list: [a,b,c..] each value is number of Neurons in corresponding hidden layer
            dropout_rate: if use drop out
            noutput: number of labels. 
        """

        layers = list()
        # 1. all hidden layers.
        for index, layer_size in enumerate(hidden_layer_list):
            layers.append(nn.Linear(ninput, layer_size))
            if use_bn:
                layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            ninput = layer_size
        # 2. last hidden layer
        if len(hidden_layer_list) == 0:
            last_hidden_layer_num = ninput
        else:
            last_hidden_layer_num = hidden_layer_list[-1]
        layers.append(nn.Linear(last_hidden_layer_num, noutput))

        # 3. generate the MLP
        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        """
        each element represents the probability of the positive class.
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)

    def _initialize_weights(self, method='he'):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if method == 'lecun':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                elif method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'he':
                    nn.init.kaiming_uniform_(m.weight)
                # m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()

    def reset_zero_grads(self):
        self.zero_grad()


class DNNModel(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """

    def __init__(self, nfield: int, nfeat: int, nemb: int,
                 hidden_layer_list: list, dropout_rate: float,
                 noutput: int, use_bn: bool = True):
        """
        Args:
            nfield: the number of fields
            nfeat: the number of features
            nemb: embedding size
        """
        super().__init__()
        self.nfeat = nfeat
        self.nemb = nemb
        self.embedding = None
        self.mlp_ninput = nfield * nemb
        self.mlp = MLP(self.mlp_ninput, hidden_layer_list, dropout_rate, noutput, use_bn)
        # self.sigmoid = nn.Sigmoid()

        # for weight-sharing
        self.is_masked_subnet = False
        self.hidden_layer_list = hidden_layer_list
        # Initialize subnet mask with ones
        self.subnet_mask = [torch.ones(size) for size in hidden_layer_list]

    def init_embedding(self, cached_embedding=None, requires_grad=False):
        if self.embedding is None:
            if cached_embedding is None:
                self.embedding = Embedding(self.nfeat, self.nemb)
            else:
                self.embedding = cached_embedding

        # in scoring process
        # Disable gradients for all parameters in the embedding layer
        if not requires_grad:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def generate_all_ones_embedding(self, batch_size=1):
        """
        Only for the MLP
        Returns:
        """
        batch_data = torch.ones(batch_size, self.mlp_ninput).double()
        return batch_data

    def forward_wo_embedding(self, x):
        """
        Only used when embedding is generated outside, eg, all 1 embedding.
        """
        y = self.mlp(x)  # B*label
        return y.squeeze(1)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        if self.is_masked_subnet:
            return self.forward_w_mask(x)
        else:
            x_emb = self.embedding(x)  # B*F*E
            y = self.mlp(x_emb.view(-1, self.mlp_ninput))  # B*label
            # this is for binary classification
            return y.squeeze(1)

    def sample_subnet(self, arch_id: str, device: str):
        # arch_id e.g., '128-128-128-128'
        sizes = list(map(int, arch_id.split('-')))
        self.is_masked_subnet = True
        # randomly mask neurons in the layers.

        for idx, size in enumerate(sizes):
            # Create a mask of ones and zeros with the required length
            mask = torch.cat([
                torch.ones(size),
                torch.zeros(self.hidden_layer_list[idx] - size)],
                dim=0).to(device)
            # Shuffle the mask to randomize which neurons are active
            mask = mask[torch.randperm(mask.size(0))]
            self.subnet_mask[idx] = mask

    def forward_w_mask(self, x):
        x_emb = self.embedding(x)  # B*F*E
        x_emb = x_emb.view(-1, self.mlp_ninput)

        # Loop till the second last layer of the MLP
        for idx, layer in enumerate(self.mlp.mlp[:-1]):  # Exclude the last Linear layer
            # 1. subnet_mask: idx // 4 is to map computation later => mlp later
            # 2. unsqueeze(1): convert to 2 dimension,
            #    and then the mask is broadcasted across the row, correspond to one neuron,
            # 3. matrix multiplication between input and the transposed weight
            if isinstance(layer, nn.Linear):
                weight = layer.weight * self.subnet_mask[idx // 4].unsqueeze(1)
                x_emb = torch.nn.functional.linear(x_emb, weight, layer.bias)
            else:
                x_emb = layer(x_emb)  # apply activation, dropout, batchnorm, etc.

        # Handle the output layer
        output_layer = self.mlp.mlp[-1]
        y = output_layer(x_emb)
        return y.squeeze(1)


class MlpSpace:
    def __init__(self, modelCfg: MlpMacroCfg):
        self.model_cfg = modelCfg
        self.name = "mlp"

    def load(self):
        pass

    @classmethod
    def serialize_model_encoding(cls, arch_micro: MlpMicroCfg) -> str:
        assert isinstance(arch_micro, MlpMicroCfg)
        return str(arch_micro)

    @classmethod
    def deserialize_model_encoding(cls, model_encoding: str) -> MlpMicroCfg:
        return MlpMicroCfg.builder(model_encoding)

    @classmethod
    def new_arch_scratch(cls, arch_macro: MlpMacroCfg, arch_micro: MlpMicroCfg, bn: bool = True):
        assert isinstance(arch_micro, MlpMicroCfg)
        assert isinstance(arch_macro, MlpMacroCfg)
        mlp = DNNModel(
            nfield=arch_macro.nfield,
            nfeat=arch_macro.nfeat,
            nemb=arch_macro.nemb,
            hidden_layer_list=arch_micro.hidden_layer_list,
            dropout_rate=0,
            noutput=arch_macro.num_labels,
            use_bn=bn,
        )
        return mlp

    def new_arch_scratch_with_default_setting(self, model_encoding: str, bn: bool):
        model_micro = MlpSpace.deserialize_model_encoding(model_encoding)
        return MlpSpace.new_arch_scratch(self.model_cfg, model_micro, bn)

    def new_architecture(self, arch_id: str):
        assert isinstance(self.model_cfg, MlpMacroCfg)
        """
        Args:
            arch_id: arch id is the same as encoding.
        Returns:
        """
        arch_micro = MlpSpace.deserialize_model_encoding(arch_id)
        assert isinstance(arch_micro, MlpMicroCfg)
        mlp = DNNModel(
            nfield=self.model_cfg.nfield,
            nfeat=self.model_cfg.nfeat,
            nemb=self.model_cfg.nemb,
            hidden_layer_list=arch_micro.hidden_layer_list,
            dropout_rate=0,
            noutput=self.model_cfg.num_labels)
        return mlp

    def new_architecture_with_micro_cfg(self, arch_micro: MlpMicroCfg):
        assert isinstance(arch_micro, MlpMicroCfg)
        assert isinstance(self.model_cfg, MlpMacroCfg)
        mlp = DNNModel(
            nfield=self.model_cfg.nfield,
            nfeat=self.model_cfg.nfeat,
            nemb=self.model_cfg.nemb,
            hidden_layer_list=arch_micro.hidden_layer_list,
            dropout_rate=0,
            noutput=self.model_cfg.num_labels)
        return mlp

    def micro_to_id(self, arch_struct: MlpMicroCfg) -> str:
        assert isinstance(arch_struct, MlpMicroCfg)
        return str(arch_struct.hidden_layer_list)

    def __len__(self):
        assert isinstance(self.model_cfg, MlpMacroCfg)
        return len(self.model_cfg.layer_choices) ** self.model_cfg.num_layers

    def get_arch_size(self, arch_micro: MlpMicroCfg) -> int:
        assert isinstance(arch_micro, MlpMicroCfg)
        result = 1
        for ele in arch_micro.hidden_layer_list:
            result = result * ele
        return result

    def sample_all_models(self) -> Generator[str, MlpMicroCfg, None]:
        assert isinstance(self.model_cfg, MlpMacroCfg)

        # 2-dimensional matrix for the search space
        space = []
        for _ in range(self.model_cfg.num_layers):
            space.append(self.model_cfg.layer_choices)
        print("explore all models")
        # generate all possible combinations
        combinations = list(itertools.product(*space))

        # Shuffle the combinations for random order
        random.shuffle(combinations)

        # encoding each of them and yield
        for ele in combinations:
            # debug only
            # yield "8-16-32-64", MlpMicroCfg([8, 16, 32, 64])
            model_micro = MlpMicroCfg(list(ele))
            model_encoding = str(model_micro)
            yield model_encoding, model_micro

    def random_architecture_id(self) -> (str, MlpMicroCfg):
        assert isinstance(self.model_cfg, MlpMacroCfg)
        arch_encod = []
        for _ in range(self.model_cfg.num_layers):
            layer_size = random.choice(self.model_cfg.layer_choices)
            arch_encod.append(layer_size)

        model_micro = MlpMicroCfg(arch_encod)
        # this is the model id == str(model micro)
        model_encoding = str(model_micro)
        return model_encoding, model_micro

    '''Below is for EA'''

    def mutate_architecture(self, parent_arch: MlpMicroCfg) -> (str, MlpMicroCfg):
        assert isinstance(parent_arch, MlpMicroCfg)
        assert isinstance(self.model_cfg, MlpMacroCfg)
        child_layer_list = deepcopy(parent_arch.hidden_layer_list)

        # 1. choose layer index
        chosen_hidden_layer_index = random.choice(list(range(len(child_layer_list))))

        # 2. choose size of the layer index, increase the randomness
        while True:
            cur_layer_size = child_layer_list[chosen_hidden_layer_index]
            mutated_layer_size = random.choice(self.model_cfg.layer_choices)
            if mutated_layer_size != cur_layer_size:
                child_layer_list[chosen_hidden_layer_index] = mutated_layer_size
                new_model = MlpMicroCfg(child_layer_list)
                return str(new_model), new_model

    def mutate_architecture_move_proposal(self, parent_arch: MlpMicroCfg):
        assert isinstance(parent_arch, MlpMicroCfg)
        assert isinstance(self.model_cfg, MlpMacroCfg)
        child_layer_list = deepcopy(parent_arch.hidden_layer_list)

        all_combs = set()
        # 1. choose layer index
        for chosen_hidden_layer_index in list(range(len(child_layer_list))):

            # 2. choose size of the layer index, increase the randomness
            while True:
                cur_layer_size = child_layer_list[chosen_hidden_layer_index]
                mutated_layer_size = random.choice(self.model_cfg.layer_choices)
                if mutated_layer_size != cur_layer_size:
                    child_layer_list[chosen_hidden_layer_index] = mutated_layer_size
                    new_model = MlpMicroCfg(child_layer_list)
                    all_combs.add((str(new_model), new_model))
                    break

        return list(all_combs)
