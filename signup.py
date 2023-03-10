from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# 顔写真をベクトル化
def parse_vector(image_path):
    mtcnn = MTCNN(image_size=160, margin=10)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return resnet(mtcnn(Image.open(image_path)).unsqueeze(0)).squeeze().to('cpu').detach().numpy().copy()

# 生徒の顔写真データを保存する
def save_vector(vector, name):
    np.save(
        "./vectors/" + name + ".npy",
        vector
    )
    return "./vectors/" + name + ".npy"

if __name__ == '__main__':
    for i in range(1, 5):
        name = "image" + str(i)
        save_vector(parse_vector(name + ".jpg"), name)
