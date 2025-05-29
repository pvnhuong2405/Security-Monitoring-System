# Real-Time Face Recognition

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
