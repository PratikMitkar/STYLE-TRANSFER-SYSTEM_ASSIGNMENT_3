# STYLE-TRANSFER-SYSTEM_ASSIGNMENT_3

**COMPANY**: CODTECH IT Solutions

**NAME**: Pratik Mitkar

**INTERN ID**: CT08GJB

**DOMAIN**: Artificial Intelligence

**BATCH DURATION**: January 5th, 2025 to February 5th, 2025

**MENTOR NAME**: NEELA SANTHOSH

## PROJECT DESCRIPTION

The **Style Transfer System** project leverages advanced deep learning techniques to apply the artistic style of one image to the content of another. Using TensorFlow, TensorFlow Hub, and Streamlit, this project provides a user-friendly web interface that allows users to upload two images: a content image (e.g., a photograph) and a style image (e.g., a painting or artwork). The system then applies the artistic style of the second image to the content of the first image using a pre-trained model, creating a stylized version of the content image.

The core of the project utilizes the **Arbitrary Image Stylization** model from TensorFlow Hub, which is capable of transferring the visual style of an image onto another. The process involves loading the images, resizing them to the required input size, and then applying the style transfer model to generate the final output image.

### Key Features:
1. **Content and Style Image Upload**: The system accepts image uploads in JPG, JPEG, and PNG formats for both content and style images.
2. **Pre-trained Style Transfer Model**: The **Arbitrary Image Stylization** model from TensorFlow Hub is used to perform the style transfer operation.
3. **Streamlit Web Interface**: An intuitive interface allows users to upload images, view them, and apply style transfer with a single click.
4. **Real-Time Style Transfer**: The system processes the uploaded images and applies style transfer in real-time, displaying the stylized image immediately.

### Development Process:
1. **Researching Style Transfer Models**: I explored how style transfer models work, specifically the Arbitrary Image Stylization model from TensorFlow Hub, which can apply various artistic styles to images.
2. **Image Preprocessing**: I implemented functions to load and preprocess images, resizing them to 256x256 pixels, as required by the model.
3. **Building the Web Interface**: Using **Streamlit**, I created an easy-to-use interface where users can upload the content and style images, and apply style transfer with a single click.
4. **Model Integration**: Integrated the TensorFlow Hub pre-trained model into the system, enabling the application of style transfer to the uploaded images.
5. **Real-Time Image Display**: Displayed both the content and style images, along with the output stylized image, in real-time on the Streamlit app.

### Skills Gained:
- **Deep Learning and Style Transfer**: Hands-on experience with style transfer models and their implementation.
- **TensorFlow and TensorFlow Hub**: Familiarity with using TensorFlow for deep learning tasks and leveraging pre-trained models from TensorFlow Hub.
- **Python and Streamlit**: Gained experience in building interactive web applications with Streamlit, as well as integrating machine learning models into web interfaces.
- **Image Processing**: Learned how to handle image preprocessing, including resizing and converting images for model input.

### Future Enhancements:
- **Improved User Experience**: Adding options for users to customize the style transfer process, such as controlling the intensity of the style application.
- **Multiple Style Support**: Enabling users to apply multiple styles to the same content image and compare the results.
- **Higher Resolution Support**: Enhancing the system to handle higher-resolution images for better output quality.

## Installation Instructions:

To run the Style Transfer System, follow these steps:

1. **Clone the Repository**:
   ```bash
   https://github.com/PratikMitkar/STYLE-TRANSFER-SYSTEM_ASSIGNMENT_3.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd STYLE-TRANSFER-SYSTEM_ASSIGNMENT_3
   ```

3. **Install Required Packages**:
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application**:
   ```bash
   streamlit run task3.py
   ```

5. **Open the Application**:
   After running the above command, a new tab will open in your web browser with the Streamlit application.

## Usage Instructions:

1. Upload a content image and a style image using the provided upload buttons.
2. Click the "Apply Style Transfer" button to generate the stylized image.
3. View the content image, style image, and the resulting stylized image in real-time.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
