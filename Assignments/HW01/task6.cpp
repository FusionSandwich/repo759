#include <iostream>
#include <cstdio>  // For printf
#include <cstdlib> // For std::atoi

int main(int argc, char *argv[])
{
    // Check if exactly one command-line argument (N) is provided
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
        return 1;
    }

    // Convert the command-line argument to an integer
    int N = std::atoi(argv[1]);

    // Print integers from 0 to N (inclusive) in ascending order using printf
    for (int i = 0; i <= N; ++i)
    {
        printf("%d", i);
        if (i < N)
        {
            printf(" "); // Print a space after each number except the last
        }
    }
    printf("\n"); // Newline after the ascending sequence

    // Print integers from N to 0 (inclusive) in descending order using std::cout
    for (int i = N; i >= 0; --i)
    {
        std::cout << i;
        if (i > 0)
        {
            std::cout << " "; // Print a space after each number except the last
        }
    }
    std::cout << std::endl; // Newline after the descending sequence

    return 0;
}