import numpy as np
import cv2

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# 顔認証用変数
IMG_SIZE = 160

# 類似度の関数
def cos_similarity(p1, p2):
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

# ユーザー認証
if __name__ == '__main__':
    users_data = {}
    for i in range(1,5):
        users_data[i] = np.load("./vectors/image" + str(i) + ".npy")

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=160, margin=10)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # cap = cv2.VideoCapture("rtsp://root:password@192.168.0.90/axis-media/media.amp")
    cap = cv2.VideoCapture(0)

    user_id = ""
    check_count = 0.0
    while True:
        ret, frame = cap.read()
        cv2.imshow('Face Match', frame)

        k = cv2.waitKey(10)
        if k == 100:
            break

        # numpy to PIL
        img_cam = Image.fromarray(frame)
        img_cam = img_cam.resize((IMG_SIZE, IMG_SIZE))

        img_cam_cropped = mtcnn(img_cam)
        if img_cam_cropped is not None:
            # if len(img_cam_cropped.size()) != 0:
            img_embedding = resnet(img_cam_cropped.unsqueeze(0))

            x2 = img_embedding.squeeze().to('cpu').detach().numpy().copy()

            user_id = ""
            for key in users_data:
                x1 = users_data[key]
                print(key)
                if cos_similarity(x1, x2) > 0.7:
                    user_id = key
                    break

            check_count += 1.0
        else:
            check_count += 0.2

        # 色々ループ抜ける条件
        if user_id:
            break

        if check_count > 100:
            break

    cap.release()
    cv2.destroyAllWindows()

    users = {1: "ブルース・ウィルス", 2: "ブルース・ウィルス", 3: "ジェイソン・ステイサム", 4: "ユーザー名"}

    if user_id:
        print("ログイン：" + users[user_id])
    else:
        print("登録済みユーザーと合致せず")
