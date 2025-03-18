#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

// Constants
const double G = 1.0;          // Gravitational constant
const double softening = 0.1;  // Softening length
const double dt = 0.01;        // Time step
const double board_size = 4.0; // Size of the board

// Function to calculate acceleration due to gravity
void getAcc(const double pos[][3], const double mass[], double acc[][3], int N) {
    for (int i = 0; i < N; i++) {
        acc[i][0] = 0.0;
        acc[i][1] = 0.0;
        acc[i][2] = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = pos[j][0] - pos[i][0];
                double dy = pos[j][1] - pos[i][1];
                double dz = pos[j][2] - pos[i][2];
                double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
                double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                acc[i][0] += G * (dx * inv_r3) * mass[j];
                acc[i][1] += G * (dy * inv_r3) * mass[j];
                acc[i][2] += G * (dz * inv_r3) * mass[j];
            }
        }
    }
}

// Save positions to a CSV file
// void savePositionsToCSV(const double pos[][3], int N, int step, const std::string &filename) {
//     std::ofstream file;
//     file.open(filename, std::ios_base::app);
//     if (!file.is_open()) {
//         std::cerr << "Error: Unable to open " << filename << " for writing!" << std::endl;
//         return;
//     }
//     file << step << ",[";
//     for (int i = 0; i < N; i++) {
//         file << "[" << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] << "]";
//         if (i < N - 1) file << ",";
//     }
//     file << "]\n";
//     file.close();
// }

int main(int argc, char *argv[]) {
    high_resolution_clock::time_point start = high_resolution_clock::now();

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number_of_particles> <simulation_end_time>" << std::endl;
        return 1;
    }

    int N = std::stoi(argv[1]);
    double tEnd = std::stod(argv[2]);
    std::string filename = "positions.csv";

    // Clear the file before starting
    std::ofstream file(filename, std::ofstream::out | std::ofstream::trunc);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to clear " << filename << "!" << std::endl;
        return 1;
    }
    file.close();

    double* mass = new double[N];
    double(*pos)[3] = new double[N][3];
    double(*vel)[3] = new double[N][3];
    double(*acc)[3] = new double[N][3];

    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    double t = 0.0;

    for (int i = 0; i < N; i++) {
        mass[i] = uniform_dist(generator);
        pos[i][0] = normal_dist(generator);
        pos[i][1] = normal_dist(generator);
        pos[i][2] = normal_dist(generator);
        vel[i][0] = normal_dist(generator);
        vel[i][1] = normal_dist(generator);
        vel[i][2] = normal_dist(generator);
    }

    double velCM[3] = {0.0, 0.0, 0.0};
    double totalMass = 0.0;
    for (int i = 0; i < N; i++) {
        velCM[0] += vel[i][0] * mass[i];
        velCM[1] += vel[i][1] * mass[i];
        velCM[2] += vel[i][2] * mass[i];
        totalMass += mass[i];
    }
    velCM[0] /= totalMass;
    velCM[1] /= totalMass;
    velCM[2] /= totalMass;
    for (int i = 0; i < N; i++) {
        vel[i][0] -= velCM[0];
        vel[i][1] -= velCM[1];
        vel[i][2] -= velCM[2];
    }

    getAcc(pos, mass, acc, N);

    int Nt = int(tEnd / dt);

    for (int step = 0; step < Nt; step++) {
        // (1/2) kick
        for (int i = 0; i < N; i++) {
            vel[i][0] += acc[i][0] * dt / 2.0;
            vel[i][1] += acc[i][1] * dt / 2.0;
            vel[i][2] += acc[i][2] * dt / 2.0;
        }
        // drift
        for (int i = 0; i < N; i++) {
            pos[i][0] += vel[i][0] * dt;
            pos[i][1] += vel[i][1] * dt;
            pos[i][2] += vel[i][2] * dt;
        }
        // ensure particles stay within the board limits
        for (int i = 0; i < N; i++) {
            for (int dim = 0; dim < 3; dim++) {
                if (pos[i][dim] > board_size) pos[i][dim] = board_size;
                else if (pos[i][dim] < -board_size) pos[i][dim] = -board_size;
            }
        }
        // update accelerations
        getAcc(pos, mass, acc, N);
        // (1/2) kick
        for (int i = 0; i < N; i++) {
            vel[i][0] += acc[i][0] * dt / 2.0;
            vel[i][1] += acc[i][1] * dt / 2.0;
            vel[i][2] += acc[i][2] * dt / 2.0;
        }
        // update time
        t += dt;
        // Save positions at each step
        // savePositionsToCSV(pos, N, step, filename);
    }

    delete[] mass;
    delete[] pos;
    delete[] vel;
    delete[] acc;

    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << "time: " << duration_sec.count() << "ms\n";

    return 0;
}