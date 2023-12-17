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

    // Завантаження зображення в сірих відтінках
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
    }

    // Знаходження ключових точок
    vector<KeyPoint> keypoints;
    fastDetector->detect(image, keypoints);

    // Виведення результатів
    Mat output;
    Scalar color(0, 0, 255);
    drawKeypoints(image, keypoints, output, color);
    cv::imshow("Fast Keypoints", output);
    cv::waitKey(0);
    
    std::cout << "Total Keypoints in Fast Detector: " << keypoints.size() << endl;
    string filename = "new_" + image_path;
    cv::imwrite(filename, output);
}




void KLTDetector_new(string gif_path)
{
    VideoCapture cap(gif_path);

    if (!cap.isOpened()) {
        cerr << "Unable to open the video file." << endl;
        
    }

    Mat frame;
    Mat gray, prevGray;

    // Ініціалізація для виявлення точок за допомогою алгоритму KLT
    vector<Point2f> prevPoints, nextPoints;
    vector<uchar> status;
    vector<float> err;

    // Зчитування першого кадру та конвертація його в відтінки сірого
    cap >> frame;
    cvtColor(frame, prevGray, COLOR_BGR2GRAY);

    // Знайдення точок на першому кадрі
    goodFeaturesToTrack(prevGray, prevPoints, 100, 0.3, 7);

    int frameCount = 0;

    while (true) {
        // Зчитування нового кадру
        cap >> frame;

        if (frame.empty()) {
            break; // Кінець відео
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Відстеження точок за допомогою алгоритму KLT
        calcOpticalFlowPyrLK(prevGray, gray, prevPoints, nextPoints, status, err);

        // Обновлення точок та виведення їх кількості
        prevPoints = nextPoints;
        frameCount++;

        int foundPoints = countNonZero(status);
        std::cout << "Frame: " << frameCount << ", Found Points: " << foundPoints << endl;

        // Відображення кадру з відміченими точками (для перевірки)
        for (size_t i = 0; i < nextPoints.size(); i++) {
            circle(frame, nextPoints[i], 3, Scalar(0, 255, 0), -1);
        }

        // Виведення відеокадру
        cv::imshow("Optical Flow", frame);

        // Завершення при натисканні клавіші 'Esc'
        if (cv::waitKey(30) == 27) {
            break;
        }

        // Переключення поточного кадру та його відтінків сірого
        swap(prevGray, gray);
    }

    cap.release();
    destroyAllWindows();
}
float Lumen(Vec3b pixel){
    //розрахунок  інтенсивності за допомогою Лумен (Ip = 0.299⋅R + 0.587⋅G + 0.114⋅B) так як зображення в сірих тонах

    // Отримуємо окремо значення кожного каналу
    int blue = pixel[0];
    int green = pixel[1];
    int red = pixel[2];
    float Ip = 0.299*red + 0.587*green + 0.114*blue;

    return Ip;

}

vector<pair<int, int>> find_pixels_around_circle(int y_center, int x_center, int radius) {
    vector<pair<int, int>> pixels; //оголошення вектора pixels для зберігання координатів

    int x = radius;
    int y = 0;
    int d = 3 - 2 * radius;

    while (x >= y) {
        pixels.push_back(make_pair(x_center + x, y_center + y));
        pixels.push_back(make_pair(x_center - x, y_center + y));
        pixels.push_back(make_pair(x_center + x, y_center - y));
        pixels.push_back(make_pair(x_center - x, y_center - y));
        pixels.push_back(make_pair(x_center + y, y_center + x));
        pixels.push_back(make_pair(x_center - y, y_center + x));
        pixels.push_back(make_pair(x_center + y, y_center - x));
        pixels.push_back(make_pair(x_center - y, y_center - x));
        if (d < 0) {
            d = d + 4 * y + 6;
        }
        else {
            d = d + 4 * (y - x) + 10;
            x--;
        }
        y++;

    }

    
    
    return (pixels);

}


bool isCorner(const Mat& image, int x, int y, int threshold = 50) {
    //Вибір порогового значення

    //обхід зображення

    // Визначення кроку для регулярної сітки
    int step = 5;
    int radius = 2;
    int n = 12;
    // Обходження зображення регулярною сіткою
    int rows = image.rows;
    int cols = image.cols;
    bool all_pass = true;
    //Для підрахунки чи є кути
    int number_of_corners = 0;
    int number_of_non_corners = 0;

    try {




        // Доступ до пікселя за координатами (x, y)

        uchar p_Value = image.at<uchar>(y, x);//отримати значення інтенсивності пікселя в конкретних координатах з чорно-білого зображення.

        //std::cout << "Value=" << static_cast<int>(pixelValue) << endl;

        //інтенсивність:
        int Ip = static_cast<int>(p_Value);
        vector<pair<int, int>> pixels = find_pixels_around_circle(y, x, radius);
        // Вивести координати знайдених пікселів
        int temp = 1;
        // std::cout << "\ncenter Pixel at (" << x_center << ", " << y_center << "): ";
        //std::cout << "Value center =" << Ip << endl;
            for (const auto& pixel : pixels) {
                //Якщо 12 точок в окрузі підходять, то це особлива точка
                while (temp < n) {
                    //std::cout <<"(" << pixel.first << " ";
                    //std::cout << pixel.second <<")" << " "<<"\n";
                    uchar circle_Value = image.at<uchar>(pixels[temp].second, pixels[temp].first);
                    //std::cout << "Value=" << static_cast<int>(circle_Value) << endl;
                    int Ic = static_cast<int>(circle_Value);

                    // Оголошення вектора для запису інтенсивності одновимірних значень
                    std::vector<int> IntenceVector;

                    // Додавання значень до вектора
                    IntenceVector.push_back(Ic);


                    for (const auto& value : IntenceVector) {
                        if (value <= Ip - threshold || value >= Ip + threshold) {
                            all_pass = false;
                            cout << "nope";
                            number_of_corners += 1;
                            break;
                             // Якщо знайдено елемент, який темніше, то перериває умову
                        }
                        else {
                            number_of_non_corners += 1;
                        }
                        if (all_pass == false) 
                        {
                            break;
                        }
                        
                    }



                    temp += 1;
                }
                
            }


        

    }

        
 

    catch (const cv::Exception& e) {
        // Обробка виключення OpenCV
        cerr << "OpenCV Exception: " << e.what() << endl;
    }
    catch (const exception& e) {
        // Обробка інших стандартних виключень
        cerr << "Standard Exception: " << e.what() << endl;
    }
    catch (...) {
        // Перехоплення інших непередбачуваних виключень
        cerr << "Unknown Exception" << endl;
    }
    
    return all_pass;
}

int main()
{
    //Вивід FastDetector
    //string images[] = { "signal-2023-12-14-212155_002.jpeg", "signal-2023-12-14-212155_003.jpeg" };
    //for (int i = 0; i < 2; i++) {
    //    FastDetector(images[i]);
    //}

    //Вивід KLTDetector

    //KLTDetector_new("4.mp4");



    Mat image = imread("signal-2023-12-14-212155_002.jpeg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
    }

    vector<Point2i> corners;
    int threshold = 60;
    for (int y = 3; y < image.rows - 3; ++y) {
        for (int x = 3; x < image.cols - 3; ++x) {
            if (isCorner(image, x, y, threshold)) {
                corners.push_back(Point2i(x, y));
                uchar circle_Value = image.at<uchar>(y, x) = 225; // Білий колір
            }
        }
    }

    // Відобразити кути на зображенні
    Mat result = image.clone();
    for (const auto& corner : corners) {
        circle(result, corner, 3, Scalar(0, 255, 0), 2);

    }
    
    // Відображення зображення
    imshow("FAST Corners", image);
    cv::imwrite("circle_image.png", image);
    cv::waitKey(0);
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
