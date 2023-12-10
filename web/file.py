from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

import cv2
import io
import numpy as np

import torch
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def load_model_dpt(model_type):
  ## Load model

  # MiDaS v3 - Large
  # (highest accuracy, slowest inference speed)
  # model_type = "DPT_Large"

  # MiDaS v3 - Hybrid
  # (medium accuracy, medium inference speed)
  # model_type = "DPT_Hybrid"

  # (lowest accuracy, highest inference speed)
  # model_type = "MiDaS_small"  # MiDaS v2.1 - Small

  midas = torch.hub.load("intel-isl/MiDaS", model_type)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  midas.to(device)
  midas.eval()

  return midas

def pre_process_dpt(image, model_type):

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transfor

    # Load image
    img = cv2.imdecode(np.frombuffer(image.file.read(),
                                      np.uint8),
                        cv2.IMREAD_COLOR)

    # convert it to the correct format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform it so that it can be used by the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = transform(img).to(device)

    # Return this image so it can be used in postprocessing
    return input_batch, img

def post_process_dpt(original, prediction):

  prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=original.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

  output = prediction.cpu().numpy()
  # Create a figure using matplotlib which super-imposes the original
  # image and the prediction

  fig = Figure()
  canvas = FigureCanvas(fig)
  ax = fig.gca()

  # Render both images original as foreground
  ax.imshow(original)
  ax.imshow(output)

  ax.axis("off")
  canvas.draw()

  # Reshape output to be a numpy array
  width, height = fig.get_size_inches() * fig.get_dpi()
  width = int(width)
  height = int(height)
  output_image = np.frombuffer(canvas.tostring_rgb(),
                                dtype='uint8').reshape(height, width, 3)

  # Encode to png
  res, im_png = cv2.imencode(".png", output_image)

  return im_png


model_type = "DPT_Large"
model = load_model_dpt(model_type)

# Code from: https://fastapi.tiangolo.com/tutorial/request-files/
app = FastAPI()


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    """ Create API endpoint to send image to and specify
     what type of file it'll take

    :param files: Get image files, defaults to File(...)
    :type files: List[UploadFile], optional
    :return: A list of png images
    :rtype: list(bytes)
    """

    for image in files:

        # Return preprocessed input batch and loaded image
        input_batch, image = pre_process_dpt(image, model_type)

        # Run the model and postpocess the output
        with torch.no_grad():
            prediction = model(input_batch)

        # # Post process and stitch together the two images to return them
        output_image = post_process_dpt(image, prediction)

        return StreamingResponse(io.BytesIO(output_image.tobytes()),
                                 media_type="image/png")


@app.get("/")
async def main():
    """Create a basic home page to upload a file

    :return: HTML for homepage
    :rtype: HTMLResponse
    """

    content = """<body>
          <h3>Upload an image to get it's depth map from the MiDaS model</h3>
          <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
              <input name="files" type="file" multiple>
              <input type="submit">
          </form>
      </body>
      """
    return HTMLResponse(content=content)
# if __name__ == "__main__":
#     import uvicorn 
#     uvicorn.run(app)