# Robolytics LabelGuard by USS Vision

Welcome to **Robolytics LabelGuard by USS Vision**! This tool helps you detect and manage mislabeled objects in your YOLOv8 datasets, combining the powerful technologies of Roboflow, Ultralytics, and USS Vision.

My name is **James Quaglia**, and I am the CTO of USS Vision, a Machine Vision and AI Integration company. I built this tool to give back to the Roboflow and Ultralytics communities. I am passionate about advancing machine vision and AI technologies, and I'm always willing to help anyone with their machine vision and AI projects. My advice and time for machine vision consultation is always free, so feel free to reach out to me for assistance! :grinning:

[Connect with me on LinkedIn](https://www.linkedin.com/in/james-quaglia-06143bb5/)

https://roboflow.com/ \
https://www.ultralytics.com/ \
https://www.ussvision.com/ \

## Features

- **Easy Integration**: Upload your YOLOv8 model and dataset to start detecting mislabeled objects.
- **Interactive Visualization**: Visualize labels and predictions with bounding boxes and polygons.
- **Detailed Metrics**: Get detailed metrics including True Positives (TP), False Positives (FP), and False Negatives (FN).
- **Customizable Thresholds**: Adjust IoU and confidence thresholds to fine-tune detection.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/robolytics-labelguard.git
    cd robolytics-labelguard
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Run the Streamlit app:
    ```sh
    streamlit run main.py
    ```

2. Upload your YOLOv8 model file (`.pt` format).

3. Enter the path to your dataset directory.

4. Adjust the percentage of the dataset to process, confidence threshold, and image size as needed.

5. Click "Process Images" to start detecting mislabeled objects.

6. Toggle between polygons and bounding boxes to visualize the results. Adjust the IoU threshold to refine detection.

### Configuration

Edit the `config.yaml` file to set default values for:
- `default_model_path`: Path to the default YOLOv8 model file.
- `dataset_directory`: Path to your dataset directory.
- `percentage`: Default percentage of the dataset to process.
- `confidence_threshold`: Default confidence threshold.
- `use_half_precision`: Use half precision for inference.
- `image_size`: Default image size.
- `initial_iou_threshold`: Initial IoU threshold.

## Contributing

We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

A huge thank you to [Roboflow](https://roboflow.com) and [Ultralytics](https://ultralytics.com) for their incredible contributions to the machine vision community. This project wouldn't be possible without their open-source efforts.

---

**Robolytics LabelGuard by USS Vision** - Making machine vision smarter, together.
