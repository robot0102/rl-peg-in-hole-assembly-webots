import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        q1 = F.relu(self.l1(xu))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(xu))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    
class Option(nn.Module):
    def __init__(self, state_dim, action_dim, option_num=3):
        super(Option, self).__init__()
        self.encoder_1 = nn.Linear(state_dim + action_dim, 400)
        self.encoder_2 = nn.Linear(400, 300)
        self.encoder_3 = nn.Linear(300, option_num)

        self.decoder_1 = nn.Linear(option_num, 300)
        self.decoder_2 = nn.Linear(300, 400)
        self.decoder_3 = nn.Linear(400, state_dim + action_dim)
        self.option_num = option_num

    def encode(self, xu):
        encoded_out = F.relu(self.encoder_1(xu))
        encoded_out = F.relu(self.encoder_2(encoded_out))
        encoded_out = self.encoder_3(encoded_out)
        return encoded_out

    def decode(self, encoded_out):
        decoded_out = F.relu(self.decoder_1(encoded_out))
        decoded_out = F.relu(self.decoder_2(decoded_out))
        decoded_out = self.decoder_3(decoded_out)
        return decoded_out

    def forward(self, x, u):
        '''
        :param x: (batch_num, state_dim)
        :param u: (batch_num, action_dim)
        :return: output_option: (batch_num, option_num)
        '''
        xu = torch.cat([x, u], 1)
        encoded_option = self.encode(xu)
        output_option = torch.softmax(encoded_option, dim=-1)

        xu_noise = add_randn(xu, vat_noise=0.005)
        encoded_option_noise = self.encode(xu_noise)
        output_option_noise = torch.softmax(encoded_option_noise, dim=-1)
        decoded_xu = self.decode(encoded_option)

        return xu, decoded_xu, output_option, output_option_noise


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class Option_Network(nn.Module):
    def __init__(self, state_dim, action_dim, option_dim, hidden_dim):
        super(Option_Network, self).__init__()

        # Option architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, option_dim)
        self.out_option = nn.softmax(option_dim)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        # action rescaling
        if max_action is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.tensor(max_action)
            self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.noise = torch.tensor(action_dim)

        self.apply(weights_init_)

        # action rescaling
        if max_action is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.tensor(max_action)
            self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class Actor1D(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, option_num = 3):
        super(Actor1D, self).__init__()
        '''
        Input size: (batch_num, channel = state_dim * option_num, length = 1)
        '''
        self.conv1 = nn.Conv1d(state_dim * option_num, 400 * option_num, kernel_size=1, groups=option_num)
        self.conv2 = nn.Conv1d(400 * option_num, 300 * option_num, kernel_size=1, groups=option_num)
        self.conv3 = nn.Conv1d(300 * option_num, action_dim * option_num, kernel_size=1, groups=option_num)
        self.max_action = max_action
        self.option_num = option_num

    def forward(self, x):
        # (batch_num, state_dim) -> (batch_num, channel = state_dim * option_num, length = 1)
        x = x.view(x.shape[0], -1, 1).repeat(1, self.option_num, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_action * torch.tanh(self.conv3(x))

        # (batch_num, action_dim * self.option_num) -> (batch_num, action_dim, option_num)
        x = x.view(x.shape[0], self.option_num, -1)
        x = x.transpose(dim0=1, dim1=2)
        return x
    
    
class ActorList(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, option_num=3):
        super(ActorList, self).__init__()
        self.l1 = nn.ModuleList([nn.Linear(state_dim, 400) for i in range(option_num)])
        self.l2 = nn.ModuleList([nn.Linear(400, 300) for i in range(option_num)])
        self.l3 = nn.ModuleList([nn.Linear(300, action_dim) for i in range(option_num)])
        self.max_action = max_action
        self.option_num = option_num

    def forward(self, x):
        x_out = []
        for o in range(self.option_num):
            x_o = F.relu(self.l1[o](x))
            x_o = F.relu(self.l2[o](x_o))
            x_o = self.max_action * torch.tanh(self.l3[o](x_o))
            x_out.append(x_o)

        return torch.stack(x_out, dim=2)
    
    
class OptionValue(nn.Module):
    def __init__(self, state_dim, option_num=3):
        super(OptionValue, self).__init__()
        self.option_num = option_num
        self.encoder_1 = nn.Linear(state_dim, 400)
        self.encoder_2 = nn.Linear(400, 300)
        self.encoder_3 = nn.Linear(300, self.option_num)

        self.decoder_1 = nn.Linear(option_num, 300)
        self.decoder_2 = nn.Linear(300, 400)
        self.decoder_3 = nn.Linear(400, self.option_num)

    def encode(self, xu):
        encoded_out = F.relu(self.encoder_1(xu))
        encoded_out = F.relu(self.encoder_2(encoded_out))
        encoded_out = self.encoder_3(encoded_out)
        return encoded_out

    def decode(self, encoded_out):
        decoded_out = F.relu(self.decoder_1(encoded_out))
        decoded_out = F.relu(self.decoder_2(decoded_out))
        decoded_out = self.decoder_3(decoded_out)
        return decoded_out

    def forward(self, x):
        '''
        :param x: (batch_num, state_dim)
        :param u: (batch_num, action_dim)
        :return: output_option: (batch_num, option_num)
        '''
        option_value = self.encode(x)
        output_option = torch.softmax(option_value, dim=-1)
        return option_value, output_option
    
    
def add_randn(x_input, vat_noise):
    """
    add normal noise to the input
    """
    epsilon = torch.FloatTensor(torch.randn(size=x_input.size())).to(device)
    return x_input + vat_noise * epsilon * torch.abs(x_input)