from ultralytics import YOLO
import argparse
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument('--model', type=str, default='my_model.pt', help='Path to the YOLO model file')
    parser.add_argument('--source', type=str, default='0', help='Input source (e.g., 0 for webcam, usb0, or video file path)')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Path to save the output video file')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the YOLO model
    model = YOLO(args.model)

    # Set up the source
    source = args.source.lower()
    if source in ['0', 'usb0']:
        source = 0  # Map to default camera

    # Open the video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return

    # Get the original width and height from the input source
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 files
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame, imgsz=(height, width))

        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()

        # Extract boxes and filter out 'watch' label
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                conf = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Skip 'watch' label
                if class_name != 'Watch':
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    # Add label and confidence
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the frame (optional, remove if not needed)
        cv2.imshow('YOLO Detection', annotated_frame)

        # Break on 'q' key press with a slight delay to reduce flicker
        if cv2.waitKey(1 if source == 0 else 30) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()