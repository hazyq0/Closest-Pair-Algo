#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <limits>
#include <chrono>
#include<cfloat>


using namespace std;
using namespace std::chrono;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Function to calculate distance between two points
double distance(const Point& p1, const Point& p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// Sequential algorithm to find the closest pair of points
pair<Point, Point> closestPairSequential(const vector<Point>& points) {
    int n = points.size();
    pair<Point, Point> closest_pair;
    double min_distance = numeric_limits<double>::max();

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dist = distance(points[i], points[j]);
            if (dist < min_distance) {
                min_distance = dist;
                closest_pair = {points[i], points[j]};
            }
        }
    }

    return closest_pair;
}

// Function to find the closest pair of points in a set
pair<Point, Point> closestPair(const vector<Point>& points) {
    int n = points.size();
    pair<Point, Point> closest_pair;
    double min_distance = numeric_limits<double>::max();

    // If there are fewer than 2 points return an invalid pair
    if (n < 2) {
        return closest_pair;
    }

    // Parallelized divide and conquer algorithm
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                // Process left half of points
                vector<Point> left_points(points.begin(), points.begin() + n / 2);
                auto left_closest_pair = closestPair(left_points);
                #pragma omp critical
                {
                    if (distance(left_closest_pair.first, left_closest_pair.second) < min_distance) {
                        min_distance = distance(left_closest_pair.first, left_closest_pair.second);
                        closest_pair = left_closest_pair;
                    }
                }
            }
            #pragma omp task
            {
                // Process right half of points
                vector<Point> right_points(points.begin() + n / 2, points.end());
                auto right_closest_pair = closestPair(right_points);
                #pragma omp critical
                {
                    if (distance(right_closest_pair.first, right_closest_pair.second) < min_distance) {
                        min_distance = distance(right_closest_pair.first, right_closest_pair.second);
                        closest_pair = right_closest_pair;
                    }
                }
            }
        }
    }

    // Merge the closest pairs from left and right halves
    // (This can be done sequentially as it's a constant-time operation)
    return closest_pair;
}


bool compareX(Point a, Point b) {
    return a.x < b.x;
}

bool compareY(Point a, Point b) {
    return a.y < b.y;
}

float dist(Point p1, Point p2) {
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

float bruteForce(std::vector<Point>& P, int n) {
    float min_dist = FLT_MAX;
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            if (dist(P[i], P[j]) < min_dist) {
                min_dist = dist(P[i], P[j]);
            }
        }
    }
    return min_dist;
}

float closestPairUtil(std::vector<Point>& Px, std::vector<Point>& Py, int n) {
    if (n <= 3) {
        return bruteForce(Px, n);
    }

    int mid = n/2;
    Point midPoint = Px[mid];

    std::vector<Point> PyLeft, PyRight;
    for (int i = 0; i < n; i++) {
        if (Py[i].x <= midPoint.x) {
            PyLeft.push_back(Py[i]);
        } else {
            PyRight.push_back(Py[i]);
        }
    }

    // Parallelize the two recursive calls
    float distLeft, distRight;
    {
        #pragma omp section
        distLeft = closestPairUtil(Px, PyLeft, mid);
        #pragma omp section
        std::vector<Point> tempVector(Px.begin() + mid, Px.end());
        distRight = closestPairUtil(tempVector, PyRight, n - mid);

        //distRight = closestPairUtil(std::vector<Point>(Px.begin() + mid, Px.end()), PyRight, n - mid); // Corrected line
    }

    float min_dist = std::min(distLeft, distRight);

    // Merge the two sorted arrays
    std::vector<Point> strip;
    for (int i = 0; i < n; i++) {
        if (abs(Py[i].x - midPoint.x) < min_dist) {
            strip.push_back(Py[i]);
        }
    }

    // Check the points in the strip
    for (int i = 0; i < strip.size(); i++) {
        for (int j = i+1; j < strip.size() && (strip[j].y - strip[i].y) < min_dist; j++) {
            if (dist(strip[i], strip[j]) < min_dist) {
                min_dist = dist(strip[i], strip[j]);
            }
        }
    }

    return min_dist;
}

float closestPair(std::vector<Point>& P, int n) {
    std::vector<Point> Px(P.begin(), P.end());
    std::sort(Px.begin(), Px.end(), compareX);

    std::vector<Point> Py(P.begin(), P.end());
    std::sort(Py.begin(), Py.end(), compareY);

    return closestPairUtil(Px, Py, n);
}

int main() {
    std::vector<Point> points = {{1, 1}, {2, 3}, {4, 5}, {5, 7}, {6, 9}};

    int n = points.size();

    float closest_dist = closestPair(points, n);

    std::cout << "Closest Pair Distance: " << closest_dist << std::endl;
  
    // Sequential algorithm
    auto start_seq = high_resolution_clock::now();
    pair<Point, Point> closest_seq = closestPairSequential(points);
    auto end_seq = high_resolution_clock::now();
    cout << "Sequential closest pair of points: (" << closest_seq.first.x << ", " << closest_seq.first.y << ") and ("
         << closest_seq.second.x << ", " << closest_seq.second.y << ")" << endl;
    cout << "Sequential distance: " << distance(closest_seq.first, closest_seq.second) << endl;
    auto duration_seq = duration_cast<milliseconds>(end_seq - start_seq);
    cout << "Sequential execution time: " << duration_seq.count() << " milliseconds" << endl;

    // Parallel algorithm
    auto start_par = high_resolution_clock::now();
    pair<Point, Point> closest_par = closestPair(points);
    auto end_par = high_resolution_clock::now();
    cout << "Parallel closest pair of points: (" << closest_par.first.x << ", " << closest_par.first.y << ") and ("
         << closest_par.second.x << ", " << closest_par.second.y << ")" << endl;
    cout << "Parallel distance: " << distance(closest_par.first, closest_par.second) << endl;
    auto duration_par = duration_cast<milliseconds>(end_par - start_par);
    cout << "Parallel execution time: " << duration_par.count() << " milliseconds" << endl;

    return 0;
};


