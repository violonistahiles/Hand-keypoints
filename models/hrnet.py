import torch
import torch.nn as nn


def conv3x3(in_ch, out_ch, stride=1):
    # Conv with kernel 3x3 and stride 1
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    # Residual block
    expansion = 1
    BN_MOMENTUM = 0.1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=self.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=self.BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckHR(nn.Module):
    # Bottleneck block
    expansion = 4
    BN_MOMENTUM = 0.1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BottleneckHR, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=self.BN_MOMENTUM)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=self.BN_MOMENTUM)
        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion, momentum=self.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    BN_MOMENTUM = 0.1

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        # Checking num_branches == num_blocks == num_channels

        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES {num_branches} <> NUM_BLOCKS {len(num_blocks)}'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES {num_branches} <> NUM_CHENNELS {len(num_channels)}'
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = f'NUM_BRANCHES {num_branches} <> NUM_INCHANNELS {len(num_inchannels)}'
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        # Create branch with sequentially connected blocks

        downsample = None
        # If input number of filters for branch not equal configurated number of filters
        # make it equal
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index],
                                                 num_channels[branch_index] * block.expansion,
                                                 kernel_size=1, stride=stirde, bias=False),
                                       nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                                                      momentum=self.BN_MOMENTUM))

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        # Create all branches for current stage
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        # Create connections between branches
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []

        # If not final fusing where all branches upsamples for highest branch,
        # loop throught all branches
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []

            for j in range(num_branches):
                # For lower branches relatively current
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )

                # For equal branch
                elif j == i:
                    fuse_layer.append(None)

                # For higher branches relatively current
                else:
                    conv3x3s = []
                    for k in range(i - j):

                        # Last conv layer without activation
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(inplace=True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))

            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        # Returns information about branches input channels
        return self.num_inchannels

    def forward(self, x):

        # If HighResolutionModule with one branch return without fusing
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # Pass branches inputs through branches
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        # Loop through branch fuse layers
        for i in range(len(self.fuse_layers)):
            # If highest branch input without changes
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])

            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])

            x_fuse.append(self.relu(y))

        return x_fuse


class PoseHighResolutionNet(nn.Module):
    # Configuration for HRNet
    # - num_modules - number of HighResolutionModule in the stage
    # - num_branches - number of one shape branches in the stage
    # - num_blocks - number of BasicBlock's in each barnch in the stage
    # - num_channels - number of input/output filters for each branch in the stage

    stage_2 = {
        'num_modules': 1,
        'num_branches': 2,
        'num_blocks': [4, 4],
        'num_channels': [32, 64],
        'block': BasicBlock
    }

    stage_3 = {
        'num_modules': 1,
        'num_branches': 3,
        'num_blocks': [4, 4, 4],
        'num_channels': [32, 64, 128],
        'block': BasicBlock
    }

    stage_4 = {
        'num_modules': 1,
        'num_branches': 4,
        'num_blocks': [4, 4, 4, 4],
        'num_channels': [32, 64, 128, 256],
        'block': BasicBlock
    }

    BN_MOMENTUM = 0.1
    NUM_JOINTS = 21

    def __init__(self):
        self.inplanes = 64
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        # Downsample input image to 1/4 size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=self.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # 4 bottlenecks
        self.layer1 = self._make_layer(BottleneckHR, 64, 4)

        # Stage 2
        num_channels = self.stage_2['num_channels']
        # block = blocks_dict[stage_2['block']]
        block = self.stage_2['block']
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage_2, num_channels)

        # Stage 3
        num_channels = self.stage_3['num_channels']
        # block = blocks_dict[stage_3['block']]
        block = self.stage_3['block']
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage_3, num_channels)

        # Stage 4
        num_channels = self.stage_4['num_channels']
        # block = blocks_dict[stage_4['block']]
        block = self.stage_4['block']
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage_4, num_channels, multi_scale_output=False)

        # Final layer
        self.final_layer = nn.Conv2d(in_channels=pre_stage_channels[0],
                                     out_channels=self.NUM_JOINTS,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        '''
        Create transition layer between stages
          num_channels_pre_layer - list with output channel numbers for each branch from previous stage
          num_channels_cur_layer - list with input channel numbers for each branch in current stage
        '''

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layer = []
        for i in range(num_branches_cur):
            # For already existing branches
            if i < num_branches_pre:
                # If number of channels from previous branch not equal current
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layer.append(None)
            # For new branch
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layer.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layer)

    def _make_layer(self, block, planes, blocks, stride=1):
        # Create sequence of 'blocks' 'block'
        # Using for 4 bottleneck
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=self.BN_MOMENTUM)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):

        '''
        layer_config - stage configuration
        num_inchannels - input number of channels for each branch in the stage
        multi_scale_output - flag to fuse all branches to one output
        '''

        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        # block = blocks_dict[layer_config['block']]
        block = layer_config['block']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used in last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    reset_multi_scale_output
                )
            )

            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        # Downsample input image to 1/4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 4 Bottlenecks
        x = self.layer1(x)

        # Make transition from input
        x_list = []
        for i in range(self.stage_2['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # Pass through stage2
        y_list = self.stage2(x_list)

        # Make transition from stage2
        x_list = []
        for i in range(self.stage_3['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # Pass through stage3
        y_list = self.stage3(x_list)

        # Make transition from stage3
        x_list = []
        for i in range(self.stage_4['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # Pass through stage3
        y_list = self.stage4(x_list)
        # Fuse every branch and pass to final layer
        x = self.final_layer(y_list[0])

        return x

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.notmal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def get_pose_net(is_train):
        model = PoseHighResolutionNet()

        if is_train:
            model.init_weights()

        return model


def load_keypoint_net(path):
    # Create hr model
    chkp_path = path + 'hrnet.pt'
    keypoint_net = PoseHighResolutionNet.get_pose_net(False)
    checkpoint = torch.load(chkp_path)
    keypoint_net.load_state_dict(checkpoint['model_state_dict'])
    keypoint_net.eval()
    # keypoint_net.to(device)

    return keypoint_net
