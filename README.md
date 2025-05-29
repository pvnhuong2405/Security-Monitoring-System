# Real-Time Face Recognition
![image](https://github.com/user-attachments/assets/bd08632a-1687-4f6e-b9d7-9cc95e12f724)
## Thêm người mới vào cơ sở dữ liệu

- Tạo thư mục mới cho mỗi người:
  <pre> ``` datasets/new_persons/ 
    ├── ten-nguoi1/ 
      ├── image1.jpg 
      └── image2.jpg
    └── ten-nguoi2/ 
      ├── image1.jpg 
      └── image2.jpg 
    ``` </pre>
- chạy python add_persons.py: thêm nhiều khuôn mặt
- chạy python recognize.py: nhận diện khuôn mặt

## Công nghệ sử dụng

- Face Detection: Retinaface, Yolov5-face, SCRFD
- Face Recognition: ArcFace
- Face Tracking: ByteTrack

## Mô tả nhanh về các file

- detect.py: Phát hiện khuôn mặt từ ảnh/video.
- face_align.py: Căn chỉnh khuôn mặt trước khi nhận diện.
