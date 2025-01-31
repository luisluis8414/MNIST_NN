#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // Path to your image file
    std::string imagePath = "resources/1_10.png"; // Replace with your actual file path

    // Read the image in grayscale
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    // Check if the image was loaded successfully
    if (img.empty())
    {
        std::cerr << "Error: Could not load image at " << imagePath << std::endl;
        return -1;
    }

    // Display the image
    cv::imshow("Loaded Image", img);

    // Wait for a key press indefinitely
    cv::waitKey(0);

    return 0;
}
