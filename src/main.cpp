#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ctime>
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;

bool toggle_record(VideoWriter& record, int& record_counter, bool& record_to_file_a, Mat& capture_frame);

// Globals
const int COUNTER_SIZE = 50;
const int DELAY = 30;
const double PADDING = 0.15;
const int MAX_VIDEO_RECORDING_FRAMES = 60; // @ 15fps = 4 sec (testing for now)

int main()
{
    // Create the cascade classifier objects used for the face detection
    CascadeClassifier face_cascade;

    // Use the specific classifier
    face_cascade.load("haarcascade_frontalface_alt.xml");

    // Setup the video capture device and link it to the first capture device detected
    VideoCapture capture_device;
    capture_device.open(0);

    // Check if everything is ok
    if (face_cascade.empty())
    {
        cerr << "Error: Could not load classifier file.\n";
        exit(1);
    }

    if (!capture_device.isOpened()) 
    {
        cerr << "Error: Could not setup the VideoCapture device.\n";
        exit(1);
    }
    
    // Set video width and height
    capture_device.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture_device.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    // Create a window to present the results
    namedWindow("CameraCapture", 1);

    // Variables used for image capture
    Mat capture_frame;
    Mat grayscale_frame;    

    // Create a vector to store the faces found
    vector<Rect> faces;

    // Variables used for detectMultiScale()
    const double scale_factor = 1.1;
    const int num_buffers = 2;
    const Size face_size(30, 30);

    // Variables used for drawing the rectangle around the face
    const int rect_line_thickness           = 1;
    const int line_type                     = 8;
    const int num_fract_bits_in_pnt_coords  = 0;

    clock_t start, end, current;
    
    int counter         = 0; // too not discriptive
    int video_frame_counter = 0;
    int vertex2_x       = 0;
    int vertex2_y       = 0;
    int vertex1_x       = 0;
    int vertex1_y       = 0;
    int vertex2_x_avg   = 0;
    int vertex2_y_avg   = 0;
    int vertex1_x_avg   = 0;
    int vertex1_y_avg   = 0;
    int input           = 0;
    
    bool out_of_bounds  = false;
    bool silenced       = false;
    bool recalibrate    = false;
    bool draw           = true;
    bool input_ready    = false;
    bool record_to_file_a = true;

    VideoWriter record("log_a.avi", CV_FOURCC('X','V','I','D'), 15, capture_frame.size(), true);

    // Create a loop to capture and find faces
    do
    { 
        // debug output
        cout << input << endl;
        // R or r
        if (input == 82 || input == 114)
        {
            if (input_ready)
            {
                if (recalibrate) 
                {
                    recalibrate = false;
                } 
                else 
                {
                    recalibrate = true;
                }
                input_ready = false;
            }
        // D or d
        } else if (input == 68 || input == 100)
        {
            if (input_ready)
            {
                if (draw)   
                {
                    draw = false;
                } 
                else 
                {
                    draw = true;
                }
                input_ready = false;
            }
        // S or s
        } else if (input == 83 || input == 115)
        {
            if (input_ready)
            {
                if (silenced) 
                {
                    silenced = false;
                    cout << "silenced = false" << endl;
                    out_of_bounds = false;                    
                }
                else 
                {
                    silenced = true;
                    cout << "silenced = true" << endl;
                }
                input_ready = false;
            }
        } 
        else 
        {
            input_ready = true;
        }

        // Capture a new image frame
        capture_device >> capture_frame;
        if (capture_frame.empty())
            break;

        // Convert captured image to grayscale and equalize
        cvtColor(capture_frame, grayscale_frame, CV_BGR2GRAY);

        // Adjust the image contrast using histogram equalization
        equalizeHist(grayscale_frame, grayscale_frame);       

        // Detect faces and store them in the vector
        face_cascade.detectMultiScale(grayscale_frame, 
                                      faces, 
                                      scale_factor, 
                                      num_buffers, 
                                      CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, 
                                      face_size);
        if (!silenced)
        {
            if (faces.size() == 0)
            {
                if (!out_of_bounds)
                {
                    start = clock();
                    out_of_bounds = true;
                }    
                // Difference in time (delta T)
                current = clock() - start;
                    
                if ((double)current/CLOCKS_PER_SEC > 2)
                {
                    cout << "ALERT!" << endl;
                    record << capture_frame;
                    if (!toggle_record(record, video_frame_counter, record_to_file_a, capture_frame))
                    {
                        cerr << "Error: Recording subsequent video set" << endl;
                        exit(1);
                    }
                }
            }

            // For all found faces in the vector, draw a rectangle on the original image
            for(size_t i = 0; i < faces.size(); i++)
            {
                input = waitKey(DELAY);
                Point vertex2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
                Point vertex1(faces[i].x, faces[i].y);

                if (recalibrate)
                {
                    counter = 0;
                    recalibrate = false;
                    vertex1_x = 0;
                    vertex1_y = 0;
                    vertex2_x = 0;
                    vertex2_y = 0;
                }
           
                if (counter < COUNTER_SIZE)
                {
                    vertex1_x += faces[i].x;
                    vertex1_y += faces[i].y;
                    vertex2_x += faces[i].x + faces[i].width;
                    vertex2_y += faces[i].y + faces[i].height;
                    counter++;
                }
                else {
                    if (draw)
                    {
                        rectangle(capture_frame,
                                  Point(vertex1_x/COUNTER_SIZE - (PADDING * (vertex1_x/COUNTER_SIZE)), 
                                        vertex1_y/COUNTER_SIZE - (PADDING * (vertex1_y/COUNTER_SIZE))),
                                  Point(vertex2_x/COUNTER_SIZE + (PADDING * (vertex2_x/COUNTER_SIZE)), 
                                        vertex2_y/COUNTER_SIZE + (PADDING * (vertex2_y/COUNTER_SIZE))),
                                  cvScalar(0,255,0,100),
                                  rect_line_thickness,
                                  line_type,
                                  num_fract_bits_in_pnt_coords);   
                    } 
                
                    vertex1_x_avg = vertex1_x/COUNTER_SIZE - (PADDING * (vertex1_x/COUNTER_SIZE));
                    vertex1_y_avg = vertex1_y/COUNTER_SIZE - (PADDING * (vertex1_y/COUNTER_SIZE));

                    vertex2_x_avg = vertex2_x/COUNTER_SIZE + (PADDING * (vertex2_x/COUNTER_SIZE));
                    vertex2_y_avg = vertex2_y/COUNTER_SIZE + (PADDING * (vertex2_y/COUNTER_SIZE));

                    if (faces[i].x < vertex1_x_avg
                        || faces[i].x + faces[i].width > vertex2_x_avg
                        || faces[i].y < vertex1_y_avg
                        || faces[i].y + faces[i].height > vertex2_y_avg)
                    {
                        if (!out_of_bounds)
                        {
                            start = clock();
                            out_of_bounds = true;
                        }    
                        // Difference in time (delta T)
                        current = clock() - start;
                    
                        if ((double)current/CLOCKS_PER_SEC > 2)
                        {
                            cout << "ALERT!" << endl;
                            record << capture_frame;
                            if (!toggle_record(record, video_frame_counter, record_to_file_a, capture_frame))
                            {
                                cerr << "Error: Recording subsequent video set" << endl;
                                exit(1);
                            }
                        }
                    } 
                    else 
                    {
                        out_of_bounds = false;
                    }
                }
                rectangle(capture_frame, 
                          vertex1, 
                          vertex2, 
                          cvScalar(0, 255, 0, 0), 
                          rect_line_thickness, 
                          line_type, 
                          num_fract_bits_in_pnt_coords); 
            }
        }
        else 
        {
            input = waitKey(DELAY);
        }
        // Print the output
        imshow("CameraCapture", capture_frame);
    } while(waitKey(10) != 27); // Quit on "Esc" key, wait 10ms

    return 0;
}

bool toggle_record(VideoWriter& record, int& record_counter, bool& record_to_file_a, Mat& capture_frame)
{
    if (record_counter > MAX_VIDEO_RECORDING_FRAMES)
    {
        record_counter = 0;
        record.release();

        if (record_to_file_a == true)
        {
            record.open("log_b.avi", CV_FOURCC('X','V','I','D'), 15, capture_frame.size(), true);                                    
        }
        else
        {
            record.open("log_a.avi", CV_FOURCC('X','V','I','D'), 15, capture_frame.size(), true);                                    
        }
        record_to_file_a = !record_to_file_a;
        if (!record.isOpened())
        {
            cout << "Error: The VideoWriter failed to open.\n";
            exit(1);
        }
    }
    record_counter++;
}