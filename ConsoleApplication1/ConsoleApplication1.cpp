// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void FastDetector(string image_path) {
    Ptr<FeatureDetector> fastDetector = FastFeatureDetector::create();

    // ������������ ���������� � ���� �������
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
    }

    // ����������� �������� �����
    vector<KeyPoint> keypoints;
    fastDetector->detect(image, keypoints);

    // ��������� ����������
    Mat output;
    Scalar color(0, 0, 255);
    drawKeypoints(image, keypoints, output, color);
    imshow("Fast Keypoints", output);
    waitKey(0);
    
    cout << "Total Keypoints in Fast Detector: " << keypoints.size() << endl;
    string filename = "new_" + image_path;
    imwrite(filename, image);
}




void KLTDetector_new(string gif_path)
{
    VideoCapture cap(gif_path);

    if (!cap.isOpened()) {
        cerr << "Unable to open the video file." << endl;
        
    }

    Mat frame;
    Mat gray, prevGray;

    // ����������� ��� ��������� ����� �� ��������� ��������� KLT
    vector<Point2f> prevPoints, nextPoints;
    vector<uchar> status;
    vector<float> err;

    // ���������� ������� ����� �� ����������� ���� � ������ �����
    cap >> frame;
    cvtColor(frame, prevGray, COLOR_BGR2GRAY);

    // ��������� ����� �� ������� ����
    goodFeaturesToTrack(prevGray, prevPoints, 100, 0.3, 7);

    int frameCount = 0;

    while (true) {
        // ���������� ������ �����
        cap >> frame;

        if (frame.empty()) {
            break; // ʳ���� ����
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // ³��������� ����� �� ��������� ��������� KLT
        calcOpticalFlowPyrLK(prevGray, gray, prevPoints, nextPoints, status, err);

        // ���������� ����� �� ��������� �� �������
        prevPoints = nextPoints;
        frameCount++;

        int foundPoints = countNonZero(status);
        cout << "Frame: " << frameCount << ", Found Points: " << foundPoints << endl;

        // ³���������� ����� � ��������� ������� (��� ��������)
        for (size_t i = 0; i < nextPoints.size(); i++) {
            circle(frame, nextPoints[i], 3, Scalar(0, 255, 0), -1);
        }

        // ��������� ���������
        imshow("Optical Flow", frame);

        // ���������� ��� ��������� ������ 'Esc'
        if (waitKey(30) == 27) {
            break;
        }

        // ������������ ��������� ����� �� ���� ������ �����
        swap(prevGray, gray);
    }

    cap.release();
    destroyAllWindows();
}

int main()
{
    //���� FastDetector
    string images[] = { "signal-2023-12-14-212155_002.jpeg", "signal-2023-12-14-212155_003.jpeg" };
    for (int i = 0; i < 2; i++) {
        FastDetector(images[i]);
    }

    //���� KLTDetector

    KLTDetector_new("4.mp4");


   

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
