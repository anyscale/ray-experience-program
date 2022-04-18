from io import BytesIO
from pydantic import BaseModel
from pprint import pprint

import requests
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

import ray
from ray import serve
from ray.experimental.dag.input_node import InputNode
from ray.serve.drivers import DAGDriver


@serve.deployment
class Preprocessor:
    """Image preprocessor with imagenet normalization."""

    def __init__(self):
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t[:3, ...]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, image_path: str) -> np.ndarray:
        pil_image = Image.open(image_path).convert("RGB")
        input_array = self.preprocessor(pil_image).unsqueeze(0)
        return input_array


@serve.deployment
class ImageClassifier:
    def __init__(self):
        self.model = models.resnet50(pretrained=True).eval()
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]

    def forward(self, input_tensor):
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        probabilities = torch.nn.functional.softmax(output_tensor[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        classify_result = [
            (self.categories[top5_catid[i]], top5_prob[i].item())
            for i in range(top5_prob.size(0))
        ]

        return {
            "classify_result": classify_result,
        }


# Python classes for upcoming tasks!

# class ContentInput(BaseModel):
#     image_url: str
#     user_id: int


# def downloader(inp: "ContentInput"):
#     """Download HTTP content, in production this can be business logic downloading from other services"""
#     image_bytes = requests.get(inp.image_url).content
#     return image_bytes


# class ImageDetector:
#     def __init__(self):
#         self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()

#     def forward(self, input_tensor):
#         with torch.no_grad():
#             return [
#                 (o["labels"].numpy().tolist(), o["boxes"].numpy().tolist())
#                 for o in self.model(input_tensor)
#             ]


# Let's Build the DAG here !!
preprocessor = Preprocessor.bind()
classifier = ImageClassifier.bind()


def input_adapter(path: str):
    return path


with InputNode() as user_input:
    image_tensor = preprocessor.preprocess.bind(user_input)
    local_dag = classifier.forward.bind(image_tensor)

    serve_entrypoint = DAGDriver.bind(local_dag, input_schema="starter.input_adapter")

if __name__ == "__main__":
    print("Started running DAG locally...")
    # url = "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01833805_hummingbird.JPEG?raw=true"
    path = "hummingbird.jpeg"
    rst = ray.get(local_dag.execute(path))
    pprint(rst)

# run `serve run starter.serve_entrypoint`
# run `curl localhost:8000/\?path=hummingbird.jpeg``
