import torch
import torch.nn as nn

from config import architecture

class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
       super(CNNBlock, self).__init__()

       self.conv = nn.Conv2d(in_channels= in_channels, out_channels = out_channels,bias = False, **kwargs)
       self.batch_norm = nn.BatchNorm2d(out_channels)
       self.act = nn.LeakyReLU(0.1)


    def forward(self, x):

        return self.conv(x)




# YOLO model

class YOLOv1(nn.Module):
    def __init__(self,split_size, boxes, num_classes, in_channels = 3, config_file = architecture):
        super(YOLOv1, self).__init__()

        self.cfg = config_file
        self.in_channels = in_channels
        self.split_size = split_size
        self.boxes = boxes
        self.classes = num_classes
        self.cnn = self._make_cnn_layers(self.cfg)
        self.linear = self._make_linear_layers(self.split_size, self.boxes, self.classes)

    def forward(self, x):
        x = self.cnn(x)
        print(x.shape)
        x = torch.flatten(x, start_dim= 1)
        x = self.linear(x)

        return x


    def _make_cnn_layers(self, cfg):
        layers = []
        in_channels = self.in_channels

        for i in cfg:
            if type(i) == tuple:
                layers += [CNNBlock(in_channels = in_channels,out_channels = i[1],kernel_size = i[0],stride = i[2],padding = i[3])]
            
                in_channels = i[1]
            
            elif type(i) == str:
                layers += [nn.MaxPool2d(kernel_size = (2,2),stride= (2,2))]
            
            elif type(i) == list:
                f_cnn = i[0]
                s_cnn = i[1]
                n_repeat = i[2]

                for r in range(n_repeat):
                    layers += [CNNBlock(in_channels = in_channels,out_channels = f_cnn[1],kernel_size = f_cnn[0], stride = f_cnn[2], padding = f_cnn[3])]

                    layers += [CNNBlock(in_channels = f_cnn[1],out_channels = s_cnn[1],kernel_size = s_cnn[0],stride = s_cnn[2], padding = s_cnn[3])]  

                    in_channels = s_cnn[1]

                
        return nn.Sequential(*layers)   

     
    def _make_linear_layers(self, S, B, C ):
        # S - > Split size
        # B -> Number of Bounding Boxes
        # C - > Number of classes
        
        fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)))

        return fcn


cfg = architecture
model = YOLOv1(split_size= 7,
                boxes= 2,
                num_classes= 20,
                in_channels= 3,
                config_file= cfg)


x = torch.randn(2, 3, 448, 448)

print(model(x).shape)