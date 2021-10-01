import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max', model_path=None, freeze=True):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        if model_path:
            self.load_state_dict(torch.load(model_path))

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        Fl = input.view(b, c, h*w)
        G = torch.bmm(Fl, Fl.transpose(1,2))
        G.div_(h*w)
        return G


def masked_features(features, mask):
    features_cropped = features[:, :, mask.squeeze() > 0]
    features_cropped = features_cropped.unsqueeze(3)

    if features_cropped.shape[2] == 0:
        return torch.zeros_like(features).reshape(features.shape[0], features.shape[1], -1).unsqueeze(3)
    else:
        return features_cropped


def mask_features_all(features, mask):
    if mask is None:
        return features
    while len(mask.shape) < 4:
        mask = mask.unsqueeze(0)
    return [masked_features(f, F.interpolate(mask.float(), f.shape[2:], mode='bilinear')) for f in features]


def total_variation_loss(x, mask=None):
    def masked(img):
        return torch.where(mask.bool(), img, torch.zeros_like(img))
    if mask is not None:
        x = masked(x)
    b, c, h_x, w_x = x.shape
    h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
    w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
    return h_tv + w_tv


def pre():
    return transforms.Compose([
        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                             std=[1, 1, 1]),
        transforms.Lambda(lambda x: x.mul_(255)), # from 0..1 to 0..255
    ])


def transform(img):
    t = pre()
    for i in range(img.shape[0]):
        img[i] = t(img[i])
    return img


class GatysVggLoss(torch.nn.Module):
    # define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']

    # these are good weights settings:
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1 for i in range(len(content_layers))]

    def __init__(self, model_path,
                 style_layers=style_layers, content_layers=content_layers,
                 style_weights=style_weights, content_weights=content_weights):
        super(GatysVggLoss, self).__init__()

        # create vgg model
        if not model_path:
            raise ValueError("No model_path provided")
        self.vgg = VGG(model_path=model_path)

        # define layers
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.layers = style_layers + content_layers

        # define weights
        self.style_weights = style_weights
        self.content_weights = content_weights

        # define loss function
        self.mse = nn.MSELoss()

        self.style_targets = None

    def forward(self, pred, target_content, target_style, mask=None):
        # compute prediction (just one joint forward pass for content and style layers needed)
        out = self.vgg(transform(pred), self.layers)
        out = mask_features_all(out, mask)
        out_style = [GramMatrix()(o) for o in out[:len(self.style_layers)]]
        out_content = out[len(self.style_layers):]

        # compute style targets.
        if self.style_targets is None:
            self.style_targets = [GramMatrix()(s).detach() for s in self.vgg(transform(target_style), self.style_layers)]

        # compute content targets.
        content_targets = [c.detach() for c in self.vgg(transform(target_content), self.content_layers)]
        content_targets = mask_features_all(content_targets, mask)

        # compute tv loss
        tv_loss = total_variation_loss(pred, mask)

        # compute losses
        style_losses = [self.style_weights[i] * self.mse(y_hat, y) for i, (y, y_hat) in enumerate(zip(self.style_targets, out_style))]
        content_losses = [self.content_weights[i] * self.mse(y_hat, y) for i, (y, y_hat) in enumerate(zip(content_targets, out_content))]

        return sum(style_losses), sum(content_losses), tv_loss
