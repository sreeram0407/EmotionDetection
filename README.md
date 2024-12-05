# Real-Time Facial Emotion Detection

This project implements a Convolutional Neural Network (CNN) for real-time facial emotion detection using TensorFlow and OpenCV. The system can detect and classify seven different emotional states: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features

- Real-time emotion detection using webcam
- Video file processing with emotion detection
- Support for training on the FER2013 dataset
- Display of top two detected emotions with confidence scores
- FPS counter for performance monitoring
- GPU support with memory growth optimization

## Requirements

```
numpy
pandas
tensorflow
opencv-python (cv2)
```

## Project Structure

- `emotion_detection.ipynb`: Main Jupyter notebook containing the implementation
- `fer2013.csv`: Dataset file (not included, must be downloaded separately)
- `emotion1.mp4`: Sample input video for testing
- `emotion1_processed.mp4`: Processed output video showing emotion detection results
- `checkpoints/`: Directory containing saved model checkpoints

## Example Output

Here's an example of the emotion detection system in action:

### Screenshot
<img src="https://github.com/sreeram0407/EmotionDetection/raw/main/key_moment.png" alt="Emotion Detection Example" width="600"/>

The screenshot above demonstrates:
- Face detection with green bounding box
- Primary emotion (Neutral: 0.43) displayed in green
- Secondary emotion (Surprise: 0.19) displayed in yellow
- Real-time confidence scores for each detected emotion

### Full Videos
- [Watch input video](https://github.com/sreeram0407/EmotionDetection/blob/main/emotion1.mp4)
- [Watch processed video with emotion detection](https://github.com/sreeram0407/EmotionDetection/blob/main/emotion1_processed.mp4)

The system processes video in real-time, detecting faces and displaying:
- Bounding boxes around detected faces
- Primary emotion with confidence score (green label)
- Secondary emotion with confidence score (yellow label)
- Each emotion is displayed with its probability percentage

## Model Architecture

The CNN architecture consists of:
- 3 Convolution blocks with batch normalization and dropout
- Dense layers with 512 and 256 units
- Output layer with 7 units (one for each emotion)
- Batch normalization and dropout for regularization

## Usage

### Training the Model

```python
# Set continue_training to False for training from scratch
model, history = train_model(continue_training=True)
```

### Processing a Video File

```python
video_input_path = './videos/your_video.mp4'
video_output_path = './output/processed_video.mp4'
process_video(video_input_path, video_output_path)
```

### Real-time Webcam Detection

```python
start_webcam_detection()
```

## Model Performance

The model achieves approximately 63% accuracy on the validation set, which is competitive with many published benchmarks on the FER2013 dataset given the challenging nature of emotion recognition.

## Model Training Details

- Uses the FER2013 dataset
- Data augmentation with rotation, flipping, and zoom
- Learning rate reduction on plateau
- Early stopping to prevent overfitting
- Checkpointing to save best model weights
- Training/validation split based on dataset's Usage column

## Performance Optimization

- GPU memory growth management
- Batch normalization for faster training
- Dropout layers to prevent overfitting
- Adjustable learning rate with ReduceLROnPlateau
- Early stopping to prevent unnecessary training iterations

## Controls

- Press 'q' to quit the webcam or video processing mode
- Window can be resized or moved while running

## Known Limitations

- Performance may vary based on lighting conditions
- Multiple face detection may impact frame rate
- GPU recommended for optimal performance during training

## Future Improvements

- Support for different face detection models
- Emotion tracking over time
- Integration with other facial analysis features
- Export to different model formats (TFLite, ONNX)
- Support for different input resolutions

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Based on the FER2013 dataset
- Uses TensorFlow and OpenCV for implementation
- Inspired by various facial emotion detection research papers
