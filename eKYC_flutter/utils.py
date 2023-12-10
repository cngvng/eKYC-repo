import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)


def check(img1, img2):
    face_embedding_1 = face_embedding(img1)
    face_embedding_2 = face_embedding(img2)
    if face_embedding_2 is False:
        return "There are no faces in the image"
    similarity = 1 - cosine(face_embedding_1, face_embedding_2)
    return similarity
def face_embedding(img):
    face = face_app.get(img)
    if len(face) == 0:
        return False
    face_embedding = face[0].embedding
    return face_embedding
