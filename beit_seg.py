from transformers import BeitModel, BeitImageProcessor
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class ThreeLayerCNN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        return x

class BeitMLASegmentation(nn.Module):
    def __init__(self, num_classes=1, model_name="microsoft/beit-base-patch16-224-pt22k"):
        super(BeitMLASegmentation, self).__init__()
        self.model = BeitModel.from_pretrained(model_name, output_hidden_states=True)
        self.processor = BeitImageProcessor.from_pretrained(model_name)

        self.three_layer_cnns = nn.ModuleList([
            ThreeLayerCNN(in_channels=768, mid_channels=384, out_channels=192)
            for _ in range(4)
        ])

        self.final_3x3_convs = nn.ModuleList([
            nn.Conv2d(192, num_classes, kernel_size=3, stride=1, padding=1)
            for _ in range(4)
        ])

    def process(self, image):
        image = image.resize((224, 224))
        inputs = self.processor(image, return_tensors="pt")
        return inputs

    def forward(self, x):
        outputs = self.model(**x)
        hidden_states = outputs.hidden_states
        hidden_states = [hs for i, hs in enumerate(hidden_states) if (i + 1) % 3 == 0]
        hidden_states = [hs[:, 1:, :] for hs in hidden_states]

        H, W = 224, 224
        C = hidden_states[0].shape[-1]
        hidden_states = [rearrange(hs, "b (h w) c -> b c h w", h=H//16, w=W//16) for hs in hidden_states]
        reshaped_conv = [self.three_layer_cnns[i](hs) for i, hs in enumerate(hidden_states)]

        aggregated = []
        for i in range(len(reshaped_conv)):
            sum_before = sum(aggregated[:i]) if aggregated else 0
            aggregated.append(sum_before + reshaped_conv[i])

        aggregated = [self.final_3x3_convs[i](agg) for i, agg in enumerate(aggregated)]
        final = aggregated[-1]

        final = F.interpolate(final, scale_factor=4, mode='bilinear', align_corners=False)

        return final

if __name__ == "__main__":
    image = Image.open("dog.jpg").convert("RGB")
    model = BeitMLASegmentation(num_classes=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        inputs = model.process(image)
        tensor = inputs["pixel_values"]

        print(torch.mean(tensor.flatten()))
        print(torch.std(tensor.flatten()))
        # model = model.to(device)
        # model.model.requires_grad_(False)
        # print(inputs["pixel_values"].shape)

        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        # # model = model.to(device)
        # outputs = model(inputs)
        # # print(outputs.shape)