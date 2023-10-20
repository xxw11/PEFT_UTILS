import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SuperScalableLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank):
        super(SuperScalableLinear, self).__init__(in_features=in_features, out_features=out_features)
        config_A_B = [f'LoRA_{rank}', 'vector', 'constant',"none"]
        config_C = [f'LoRA_{rank}', 'vector',"none"]
        config_D_E = ['constant', 'vector',"none"]
        self.configs = []
        for A in config_A_B:
            for B in config_A_B:
                for C in config_C:
                    for D in config_D_E:
                        for E in config_D_E:
                            config = {'A':A,'B':B,'C':C,'D':D,'E':E}
                            self.configs.append(config)
        self.path_config = random.choice(self.configs)
        self.Ad, self.Au = self.make_param((out_features, in_features), f'LoRA_{rank}')
        self.Bd, self.Bu = self.make_param((out_features, in_features), f'LoRA_{rank}')
        self.Cd, self.Cu = self.make_param((in_features, 1), f'LoRA_{rank}')
        self.D = nn.Parameter(torch.zeros(out_features))
        self.E = nn.Parameter(torch.zeros(out_features))
        self.eval_config = None
        nn.init.xavier_uniform_(self.Au)
        nn.init.xavier_uniform_(self.Bu)
        nn.init.xavier_uniform_(self.Cu)
    
    def prepare_path(self, config, Xd, Xu=None):
        if Xu is not None:
            # 定义门控值
            gate_LoRA = 1 if 'LoRA' in config else 0
            gate_vector = 1 if 'vector' in config else 0
            gate_constant = 1 if 'constant' in config else 0
            # 对于LoRA方式的数据处理
            if 'LoRA' in config:
                rank = int(config.split('_')[1])
            else:
                rank = Xd.shape[1]  # 或其它合适的默认值
            X_LoRA = torch.matmul(Xd[:, :rank], Xu[:rank, :]) * gate_LoRA
            # 对于vector方式的数据处理
            X_vector = Xd[:, 0].unsqueeze(1) * gate_vector
            # 对于constant方式的数据处理
            X_constant = Xd[0, 0] * gate_constant
            # 将所有的结果加在一起
            X = X_LoRA + X_vector + X_constant
        else:
            # 定义门控值
            gate_vector = 1 if 'vector' in config else 0
            gate_constant = 1 if 'constant' in config else 0
            # 对于vector方式的数据处理
            X_vector = Xd * gate_vector
            # 对于constant方式的数据处理
            X_constant = Xd[0] * gate_constant
            # 将所有的结果加在一起
            X = X_vector + X_constant
        return X
    
    def make_param(self, shape, config=None):
        if 'LoRA' in config:
            out_feature = shape[0]
            in_feature = shape[1]
            try:
                rank = int(config.split('_')[1])
            except:
                rank = 4
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape))
        
    def forward(self, input):
        if self.eval_config is not None:
            self.path_config = self.eval_config
        else:
            self.path_config = random.choice(self.configs)
        A = self.prepare_path(self.path_config['A'], self.Ad, self.Au)
        B = self.prepare_path(self.path_config['B'], self.Bd, self.Bu)
        C = self.prepare_path(self.path_config['C'], self.Cd, self.Cu)
        D = self.prepare_path(self.path_config['D'], self.D)
        E = self.prepare_path(self.path_config['E'], self.E)
        optimal_weight = self.weight + self.weight*A + B
        if torch.is_tensor(self.bias):
            optimal_bias = self.bias + self.bias*D + E
        else:
            optimal_bias =0*D + E
        optimal_prompt = torch.matmul(self.weight, C).squeeze()
        return F.linear(input, optimal_weight, optimal_bias+optimal_prompt)

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = SuperScalableLinear(linear_module.in_features, linear_module.out_features, rank)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear

class GloraModuleInjection:
    @staticmethod
    def make_scalable(linear_module, rank=4):
        """Make a (linear) layer super scalable.
        :param linear_module: A Linear module
        :return: a suepr linear that can be trained to
        """
        new_linear = SuperScalableLinear.from_linear(linear_module, rank)
        return new_linear
    
def set_glora(model, lora_rank):
    layers = []
    for name, l in model.named_modules():
        if isinstance(l, nn.Linear):
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            layers.append([layer, tokens[-1]])
    for parent_layer, last_token in layers:
        if not 'head' in last_token:
            setattr(parent_layer, last_token, GloraModuleInjection.make_scalable(getattr(parent_layer, last_token), lora_rank))

@torch.no_grad()
def save_glora(save_path, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E']]):
            trainable[n] = p.data
    torch.save(trainable, save_path )
    
def load_glora(load_path, model):
    weights = torch.load(load_path)
    loaded = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E']]):
            p.data = weights[n]
            loaded +=1
    print(f'successfully loaded {loaded} trained parameter tensors')
    return model

def set_glora_eval_config(eval_config, model):
    i=0
    for name, l in model.named_modules():
        if isinstance(l, torch.nn.Linear):
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            glora_layer = getattr(layer, tokens[-1])
            if hasattr(glora_layer,"eval_config"):
                glora_layer.eval_config = eval_config[i]
                print(f'layer_name:{name},eval_config:{glora_layer.eval_config}')
                i = i+1