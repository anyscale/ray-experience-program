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


class ContentInput(BaseModel):
    image_url: str
    user_id: int


@serve.deployment
def downloader(inp: "ContentInput"):
    """Download HTTP content, in production this can be business logic downloading from other services"""
    image_bytes = requests.get(inp.image_url).content
    return image_bytes


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

    def preprocess(self, image_payload_bytes: bytes) -> np.ndarray:
        pil_image = Image.open(BytesIO(image_payload_bytes)).convert("RGB")
        input_array = self.preprocessor(pil_image).unsqueeze(0)
        return input_array


@serve.deployment
class ImageClassifier:
    def __init__(self, version: int):
        self.version = version

        self.model = models.resnet50(pretrained=True).eval()
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]

    def forward(self, input_tensor):
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        probabilities = torch.nn.functional.softmax(output_tensor[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        classify_result = [
            (
                self.categories[top5_catid[i]],
                top5_prob[i].item(),
            )
            for i in range(top5_prob.size(0))
        ]

        return {
            "classify_result": classify_result,
            "model_version": self.version,
        }


@serve.deployment
class DynamicDispatch:
    def __init__(self, *classifier_models):
        self.classifier_models = classifier_models

    async def forward(self, inp_tensor, inp: "ContentInput"):
        chosen_idx = inp.user_id % len(self.classifier_models)
        chosen_model = self.classifier_models[chosen_idx]
        return await chosen_model.forward.remote(inp_tensor)


@serve.deployment
class ImageDetector:
    def __init__(self):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()

    def forward(self, input_tensor):
        with torch.no_grad():
            return [
                (o["labels"].numpy().tolist(), o["boxes"].numpy().tolist())
                for o in self.model(input_tensor)
            ]


@serve.deployment
def combine(classify_output, detection_output):
    return {
        "resnet_version": classify_output["model_version"],
        "classify_result": classify_output["classify_result"],
        "detection_output": detection_output,
    }


# Let's Build the DAG here !!
preprocessor = Preprocessor.bind()
classifiers = [ImageClassifier.bind(i) for i in range(3)]
dispatcher = DynamicDispatch.bind(*classifiers)
detector = ImageDetector.bind()


def input_adapter(image_url: str, user_id: int):
    return ContentInput(image_url=image_url, user_id=user_id)


with InputNode() as user_input:
    image_bytes = downloader.bind(user_input)
    image_tensor = preprocessor.preprocess.bind(image_bytes)

    classification_output = dispatcher.forward.bind(image_tensor, user_input)
    detection_output = detector.forward.bind(image_tensor)
    local_dag = combine.bind(classification_output, detection_output)

    serve_entrypoint = DAGDriver.bind(local_dag, input_schema="solution.input_adapter")

if __name__ == "__main__":
    print("Started running DAG locally...")
    url = "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01833805_hummingbird.JPEG?raw=true"

    user_input = ContentInput(image_url=url, user_id=1)
    rst = ray.get(local_dag.execute(user_input))
    pprint(rst)

# run `serve run solution.serve_entrypoint`
# go to localhost:8000/docs and use the OpenAPI UI
# or
# curl -X 'GET' \
#   'http://localhost:8000/?image_url=https%3A%2F%2Fgithub.com%2FEliSchwartz%2Fimagenet-sample-images%2Fblob%2Fmaster%2Fn01833805_hummingbird.JPEG%3Fraw%3Dtrue&user_id=1' \
#   -H 'accept: application/json'
