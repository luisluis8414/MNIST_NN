#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>

class FileReader
{
public:
    FileReader(int width)
        : m_width(width) {}

    std::vector<std::vector<double>> splitImage(const std::string &imagePath, int n)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("n must be greater than 0");
        }

        // Load the image in grayscale
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            throw std::runtime_error("Error: Could not load image at " + imagePath);
        }

        int imgWidth = img.cols;
        int imgHeight = img.rows;
        int chunkWidth = imgWidth / n;
        std::vector<std::vector<double>> chunks;

        for (int j = 0; j < n; ++j)
        {
            int x = j * chunkWidth;
            cv::Rect roi(x, 0, chunkWidth, imgHeight);
            cv::Mat chunk = img(roi);
            std::vector<double> flattenedChunk;

            // Convert the chunk to double and normalize
            chunk.convertTo(chunk, CV_64F, 1.0 / 255.0);

            // Correctly get the data pointers
            double *dataStart = (double *)chunk.data;
            double *dataEnd = dataStart + chunk.total();

            flattenedChunk.assign(dataStart, dataEnd);
            chunks.push_back(flattenedChunk);
        }

        return chunks;
    }

private:
    int m_width; // Target width for chunking
};
