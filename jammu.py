from deepface import DeepFace
import cv2
from mtcnn.mtcnn import MTCNN

def draw_facebox(image, result_list, label, color):
    # Draw rectangles around detected faces
    for result in result_list:
        x, y, width, height = result['box']
        # Draw rectangle with the specified color
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        # Add label text above the rectangle
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Path to the input image for verification
input_image_path = r"C:\Users\venka\OneDrive\Desktop\cctv\detected_faces1\WhatsApp Image 2024-10-25 at 18.04.14_26ccf182.jpg"  # Replace with your input image

# Path to the input video file
video_path = r"C:\Users\venka\OneDrive\Desktop\cctv\input video.mp4" # Replace with your video file

# Path to the output video file
output_video_path = r"C:\Users\venka\OneDrive\Desktop\cctv\verified_output_video1.mp4"  # Output video path

# Create MTCNN detector
detector = MTCNN()

# Open video capture
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create video writer to output the verified video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_number = 0  # To track the frame number
confidence_threshold = 0.5  # Set confidence threshold

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if no frames are returned
    if not ret:
        break

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = detector.detect_faces(frame_rgb)

    # Process each detected face
    for face in faces:
        x, y, width, height = face['box']
        detected_face = frame_rgb[y:y + height, x:x + width]

        # Save the detected face temporarily for verification
        detected_face_path = "face_temp.jpg"
        cv2.imwrite(detected_face_path, cv2.cvtColor(detected_face, cv2.COLOR_RGB2BGR))

        try:
            # Verify the detected face with the input image
            result = DeepFace.verify(
                img1_path=input_image_path,
                img2_path=detected_face_path,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='mtcnn'
            )

            # Determine verification result
            if result['verified'] and result['distance'] < confidence_threshold:
                print(f"Verified face in frame {frame_number} with distance {result['distance']}")
                frame = draw_facebox(frame, [face], label="Verified", color=(0, 255, 0))  # Green box for verified
            else:
                print(f"Unverified face in frame {frame_number} with distance {result['distance']}")
                frame = draw_facebox(frame, [face], label="Unverified", color=(0, 0, 255))  # Red box for unverified

        except ValueError as e:
            print(f"Error processing frame {frame_number}: {e}")

    # Write the processed frame to the output video
    out.write(frame)

    frame_number += 1  # Increment the frame number

# Release the video capture and writer objects
cap.release()
out.release()

print("Face detection and verification completed. Output video saved.")
