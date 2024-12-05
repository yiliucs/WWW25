import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

class encoder1(nn.Module):
    def __init__(self):
        super(encoder1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=(2, 2))
        self.fc1 = nn.Linear(64 * 1 * 6, 128)
        self.dense1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(256, 56)


    def forward(self, x):
        x=x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 1 * 6)
        x = F.relu(self.dense1_bn(self.fc1(x)))

        # pdb.set_trace()
        return x

class encoder2(nn.Module):
    def __init__(self):
        super(encoder2, self).__init__()

        self.Vconv1 = nn.Conv3d(1, 16, (3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Vbn1 = nn.BatchNorm3d(16)
        self.Vconv2 = nn.Conv3d(16, 32, (3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1))
        self.Vbn2 = nn.BatchNorm3d(32)
        self.Vconv3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.Vbn3 = nn.BatchNorm3d(64)
        self.Vconv4 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.Vbn4 = nn.BatchNorm3d(64)
        self.Vpool1 = nn.MaxPool3d(2, stride=2)
        self.Vpool2 = nn.MaxPool3d((2, 2, 2), stride=(1, 2, 2))
        self.Vfc1 = nn.Linear(64 * 2 * 7 * 5, 128)
        self.Vdense1_bn1 = nn.BatchNorm1d(128)

    def forward(self,  y):

        y=y.unsqueeze(1)

        y = self.Vpool1(F.relu(self.Vbn1(self.Vconv1(y))))
        y = self.Vpool1(F.relu(self.Vbn2(self.Vconv2(y))))
        y = self.Vpool1(F.relu(self.Vbn3(self.Vconv3(y))))
        y = self.Vpool1(F.relu(self.Vbn4(self.Vconv4(y))))
        y = y.view(-1, 64 * 2 * 7 * 5)
        y = F.relu(self.Vdense1_bn1(self.Vfc1(y)))
        # pdb.set_trace()
        return y





class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = encoder1()
        self.encoder_2 = encoder2()

    def forward(self, x1, x2):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)

        return feature_1, feature_2



class MyMMModel(nn.Module):
    """Model for human-activity-recognition."""

    def __init__(self, num_classes, miss_modal, miss_rate):  # [4352539]
        super().__init__()
        self.modality = 'all'
        self.miss_modal = miss_modal
        self.miss_rate = miss_rate

        self.encoder = Encoder()

        # Classify output, fully connected layers
        # self.classifier = nn.Linear(1920, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(256, 56),
            nn.BatchNorm1d(56),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(56, 7)

            )

    def forward(self, x1, x2,latent=False):
        if latent:
            feature=x1
        else:

            feature_1, feature_2 = self.encoder(x1, x2)

            feature = torch.cat((feature_1, feature_2), dim=1) # weighted sum
        # fused_feature = torch.cat((acc_output,gyro_output), dim=1) #concate
        # print(fused_feature.shape)

        output = self.classifier(feature)

        return output,feature

class DivLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward2(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer=layer.view((layer.size(0), -1))
        chunk_size=layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

class Generator(nn.Module):
    def __init__(self,  modality,opt,input_dim=7, embedding=False, latent_layer_idx=-1):
        super(Generator, self).__init__()
        self.embedding = embedding
        #self.model=
        self.opt=opt
        self.latent_layer_idx = latent_layer_idx
        self.modality =modality
        #0.4 use 512 0.6 1024
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = 1024, 256, 1, 7, 7
        self.fc_configs = [input_dim*2, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss=nn.NLLLoss(reduce=False) # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        # print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        labels.cuda(self.opt.cuda_device)
        # print(labels)
        result = {}
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim)).cuda(self.opt.cuda_device)# sampling from Gaussian
        if verbose:
            result['eps'] = eps
        if self.embedding: # embedded dense vector
            y_input = self.embedding_layer(labels).cuda(self.opt.cuda_device)
        else: # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class).cuda(self.opt.cuda_device)
            y_input.zero_()
            #labels = labels.view
            y_input.scatter_(1, labels.view(-1,1), 1)
            # print(y_input)

        z = torch.cat((eps, y_input), dim=1)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result


class NoiseGenerator(nn.Module):
    def __init__(self, input_shapex,input_shapey,hid1,hid2):
        super(NoiseGenerator, self).__init__()
        self.fcx = nn.Linear(input_shapex, hid1)
        self.bnx = nn.BatchNorm1d(hid1)
        self.actx = nn.ReLU()
        self.repx=nn.Linear(hid1, input_shapex)
        self.fcy = nn.Linear(input_shapey, hid2)
        self.bny = nn.BatchNorm1d(hid2)
        self.acty = nn.ReLU()
        self.repy=nn.Linear(hid2, input_shapey)


    def forward(self, x,y):
        noisex=x.reshape(x.shape[0], -1)
        noisey=y.reshape(y.shape[0], -1)
        noisex=self.fcx(noisex)
        noisex=self.bnx(noisex)
        noisex=self.actx(noisex)
        noisex = self.repx(noisex)
        noisex=noisex.view(x.shape)
        noisey = self.fcy(noisey)
        noisey = self.bny(noisey)
        noisey = self.acty(noisey)
        noisey = self.repy(noisey)
        noisey=noisey.view(y.shape)


        return noisex,noisey
