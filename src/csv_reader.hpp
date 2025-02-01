#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

class CsvReader
{
private:
    std::ifstream csvfile;
    bool hasHeader;
    bool fileIsOpen;

public:
    CsvReader(const std::string &filePath)
        : fileIsOpen(false)
    {
        open(filePath);
    }

    ~CsvReader()
    {
        if (fileIsOpen)
        {
            csvfile.close();
        }
    }

    // Open the CSV file
    void open(const std::string &filePath)
    {
        csvfile.open(filePath);
        if (!csvfile.is_open())
        {
            throw std::runtime_error("Failed to open the CSV file: " + filePath);
        }
        fileIsOpen = true;
    }

    bool isOpen() const
    {
        return fileIsOpen && csvfile.is_open();
    }

    // Read the next row from the CSV file
    std::string readNextRow()
    {
        if (!isOpen())
        {
            throw std::runtime_error("CSV file is not open.");
        }

        std::string line;
        std::getline(csvfile, line);
        return line;
    }

    // Split a string into tokens based on commas
    static std::vector<std::string> splitString(const std::string &s)
    {
        std::vector<std::string> tokens;
        std::stringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            tokens.push_back(token);
        }
        return tokens;
    }

    // Read the next row and extract the label and pixels.
    // Returns a pair where first is the label as int and second is the image pixels.
    std::pair<int, std::vector<unsigned char>> getLabelAndPixels()
    {
        std::string line = readNextRow();
        std::vector<std::string> row = splitString(line);

        if (row.size() != 785)
        { // 1 label + 784 pixels
            throw std::runtime_error("Invalid row format in CSV file.");
        }

        int label = std::stoi(row[0]);
        std::vector<unsigned char> pixels;
        pixels.reserve(784);
        for (size_t i = 1; i < row.size(); ++i)
        {
            pixels.push_back(static_cast<unsigned char>(std::stoi(row[i])));
        }

        return {label, pixels};
    }

    // Check if there's data that can be read from the file
    bool eof() const
    {
        return csvfile.eof();
    }
};
