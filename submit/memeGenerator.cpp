//
// Created by Yuyang Tian on 2025/1/22.
// To generate meme using OpenCV
//
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class MemeEditor {
private:
    string topText;   // Stores the top text of the meme
    string bottomText; // Stores the bottom text of the meme
    Mat preview;  // Stores the processing frame
    bool isEditing;     // Keeps track of whether the editor is active

    /**
     * @brief Adds meme text to the given image with outlined styling.
     *
     * This function adds top and bottom text to the provided image.
     * The text is centered and outlined for better readability.
     *
     * @param img The image on which the text is drawn.
     * @param topText The text to be displayed at the top of the image.
     * @param bottomText The text to be displayed at the bottom of the image.
     */
    void addMemeText(Mat& img, const string& topText, const string& bottomText) {
        const int FONT_FACE = FONT_HERSHEY_SIMPLEX; // Font type for the text
        const double FONT_SCALE = 1.0;             // Scale of the font
        const int THICKNESS = 2;                   // Thickness of the text
        const Scalar TEXT_COLOR(255, 255, 255);    // White text color
        const Scalar OUTLINE_COLOR(0, 0, 0);       // Black outline color
        const int OUTLINE_THICKNESS = 3;           // Thickness of the text outline

        // Lambda function to add text with an outline
        auto addTextWithOutline = [&](const string& text, Point position) {
            int baseline = 0;
            Size textSize = getTextSize(text, FONT_FACE, FONT_SCALE, THICKNESS, &baseline);
            position.x = (img.cols - textSize.width) / 2; // Center the text horizontally

            // Draw the outline first
            putText(img, text, position, FONT_FACE, FONT_SCALE, OUTLINE_COLOR, OUTLINE_THICKNESS);
            // Draw the text over the outline
            putText(img, text, position, FONT_FACE, FONT_SCALE, TEXT_COLOR, THICKNESS);
        };

        // Add the top text if provided
        if (!topText.empty()) {
            addTextWithOutline(topText, Point(0, 50)); // Position near the top
        }

        // Add the bottom text if provided
        if (!bottomText.empty()) {
            int baseline = 0;
            addTextWithOutline(bottomText, Point(0, img.rows - baseline - 10)); // Position near the bottom
        }
    }

public:
    /**
     * @brief Launches an interactive meme editor for the user.
     *
     * Allows the user to add/edit top and bottom text on the image.
     * Provides options to save or discard changes and preview the meme in real-time.
     *
     * @param img The image to edit and apply meme text to.
     */
    void editMeme(Mat& img) {
        // Display controls for the user
        cout << "\nMeme Editor Controls:\n"
             << "t - Add/edit top text\n"
             << "b - Add/edit bottom text\n"
             << "c - Clear all text\n"
             << "s - Save and exit\n"
             << "q - Quit without saving\n\n";

        namedWindow("Meme Editor"); // Create a window for the meme editor
        isEditing = true;

        // Main editing loop
        while (isEditing) {
            preview = img.clone(); // Clone the original image for preview
            addMemeText(preview, topText, bottomText); // Add the current meme text to the preview
            imshow("Meme Editor", preview); // Show the preview to the user
            // Wait for a keypress and handle the user's input
            char key = static_cast<char>(waitKey(100));
            handleKeyPress(key);
            }
        }

        void handleKeyPress(int key) {
            switch (key) {
                case 't': // Edit top text
                    cout << "Enter top text: ";
                    getline(cin, topText);
                    break;
                case 'b': // Edit bottom text
                    cout << "Enter bottom text: ";
                    getline(cin, bottomText);
                    break;
                case 'c': // Clear all text
                    topText.clear();
                    bottomText.clear();
                    break;
                case 's': // Save the meme and exit
                    addMemeText(preview, topText, bottomText); // Apply the final text
                    imwrite("../outputs/meme_output.png", preview); // Save the edited meme to a file
                    cout << "Meme saved successfully.\n";
                    isEditing = false; // Exit the editor
                    destroyWindow("Meme Editor");
                case 'q': // Quit without saving
                    cout << "Exiting meme editor.\n";
                    isEditing = false; // Exit the editor
                    destroyWindow("Meme Editor");
                    break;
                default: // Ignore unrecognized inputs
                    break;
        }
        destroyWindow("Meme Editor"); // Close the editor window
    }

};
