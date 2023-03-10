from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# 顔検出のAI
# image_size: 顔を検出して切り取るサイズ
# margin: 顔まわりの余白
mtcnn = MTCNN(image_size=160, margin=10)

# 切り取った顔を512個の数字にするAI
# 1回目の実行では学習済みのモデルをダウンロードしますので、少し時間かかります。
resnet = InceptionResnetV1(pretrained='vggface2').eval()


# 三人分の比較をします。
# 1つ目をカメラで取得した人として
# 2、3つ目を登録されている人とします。
image_path1 = "image1.jpg"
image_path2 = "image2.jpg"
image_path3 = "image3.jpg"

# (仮)カメラで取得した方
# 画像データ取得
img1 = Image.open(image_path1) 
# 顔データを160×160に切り抜き
img_cropped1 = mtcnn(img1)
# save_pathを指定すると、切り取った顔画像が確認できます。
# img_cropped1 = mtcnn(img1, save_path="cropped_img1.jpg")
# 切り抜いた顔データを512個の数字に
img_embedding1 = resnet(img_cropped1.unsqueeze(0))

# (仮)登録されたカメラと同じ人
img2 = Image.open(image_path2)
img_cropped2 = mtcnn(img2)
img_embedding2 = resnet(img_cropped2.unsqueeze(0))

# (仮)登録されたカメラと違う人
img3 = Image.open(image_path3)
img_cropped3 = mtcnn(img3)
img_embedding3 = resnet(img_cropped3.unsqueeze(0))

# 類似度の関数
def cos_similarity(p1, p2): 
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

# 512個の数字にしたものはpytorchのtensorという型なので、numpyの方に変換
p1 = img_embedding1.squeeze().to('cpu').detach().numpy().copy()
p2 = img_embedding2.squeeze().to('cpu').detach().numpy().copy()
p3 = img_embedding3.squeeze().to('cpu').detach().numpy().copy()

# 類似度を計算して顔認証
img1vs2 = cos_similarity(p1, p2)
img1vs3 = cos_similarity(p1, p3)

print("1つ目と2つ目の比較", img1vs2)
print("1つ目と3つ目の比較", img1vs3)