#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>
#include <omp.h>
#include <string>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

// Constants
const double G = 1.0;          // Gravitational constant
const double softening = 0.1;  // Softening length
const double dt = 0.01;        // Time step
const double board_size = 4.0; // Size of the board

// Function to calculate acceleration due to gravity
void getAcc(const double pos[][3], const double mass[], double acc[][3], int N, const std::string& schedule) {
    if (schedule == "static") {
        #pragma omp parallel for schedule(static)
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
    } else if (schedule == "dynamic") {
        #pragma omp parallel for schedule(dynamic)
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
    } else if (schedule == "guided") {
        #pragma omp parallel for schedule(guided)
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
    } else {
        std::cerr << "Error: Unknown scheduling type '" << schedule << "'. Use 'static', 'dynamic', or 'guided'." << std::endl;
        exit(1);
    }
}

// Optional: Save positions to a CSV file (commented out for performance runs)
void savePositionsToCSV(const double pos[][3], int N, int step, const std::string &filename) {
    std::ofstream file;
    file.open(filename, std::ios_base::app);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filename << " for writing!" << std::endl;
        return;
    }
    file << step << ",[";
    for (int i = 0; i < N; i++) {
        file << "[" << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] << "]";
        if (i < N - 1) file << ",";
    }
    file << "]\n";
    file.close();
}

int main(int argc, char *argv[]) {
    high_resolution_clock::time_point start = high_resolution_clock::now();

    // Check command-line arguments
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <number_of_particles> <simulation_end_time> <num_threads> <schedule_type>" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    int N = std::stoi(argv[1]);            // Number of particles
    double tEnd = std::stod(argv[2]);      // Simulation end time
    int num_threads = std::stoi(argv[3]);  // Number of threads
    std::string schedule = argv[4];        // Scheduling type

    // Set number of OpenMP threads
    omp_set_num_threads(num_threads);

    // Allocate dynamic arrays
    double* mass = new double[N];
    double(*pos)[3] = new double[N][3];
    double(*vel)[3] = new double[N][3];
    double(*acc)[3] = new double[N][3];

    // Random number generation
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // Initialize particles
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

    // Convert to center-of-mass frame
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

    // Initial accelerations
    getAcc(pos, mass, acc, N, schedule);

    // Number of timesteps
    int Nt = static_cast<int>(tEnd / dt + 0.5);

    // Main simulation loop
    for (int step = 0; step < Nt; step++) {
        // (1/2) kick
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            vel[i][0] += acc[i][0] * dt / 2.0;
            vel[i][1] += acc[i][1] * dt / 2.0;
            vel[i][2] += acc[i][2] * dt / 2.0;
        }
        // Drift
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            pos[i][0] += vel[i][0] * dt;
            pos[i][1] += vel[i][1] * dt;
            pos[i][2] += vel[i][2] * dt;
        }
        // Ensure particles stay within bounds
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            for (int dim = 0; dim < 3; dim++) {
                if (pos[i][dim] > board_size) pos[i][dim] = board_size;
                else if (pos[i][dim] < -board_size) pos[i][dim] = -board_size;
            }
        }
        // Update accelerations with specified schedule
        getAcc(pos, mass, acc, N, schedule);
        // (1/2) kick
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            vel[i][0] += acc[i][0] * dt / 2.0;
            vel[i][1] += acc[i][1] * dt / 2.0;
            vel[i][2] += acc[i][2] * dt / 2.0;
        }
        t += dt;
        // Optional: Uncomment for debugging
        // savePositionsToCSV(pos, N, step, "positions.csv");
    }

    // Clean up memory
    delete[] mass;
    delete[] pos;
    delete[] vel;
    delete[] acc;

    // Timing
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << "threads: " << num_threads << " schedule: " << schedule << " time: " << duration_sec.count() << "ms\n";

    return 0;
}
