import numpy as np

# from fastapi.middleware.cors import CORSMiddleware
# CORS (Cross-Origin Resource Sharing) middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Replace with your frontend's URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

def process_image(x, y):
    # Your image processing logic here
    # For now, let's just return whether the images are the same or not
    if np.array_equal(x, y):
        return "Images are the same"
    else:
        return "Images are different"