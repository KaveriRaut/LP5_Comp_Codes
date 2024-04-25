#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Function to perform Bubble Sort
void sequential_bubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort
void parallelBubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i)
    {
#pragma omp parallel for
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main()
{
    int n;
    cout << "\n enter total no of elements=>";
    cin >> n;
    vector<int>arr(n);
    vector<int>arr_copy(n);
    cout << "\n enter elements=>";
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
        arr_copy[i]=arr[i];
    }

    double start_time = omp_get_wtime(); // start timer for sequential algorithm
    sequential_bubbleSort(arr);
    double end_time = omp_get_wtime(); // end timer for sequential algorithm
    cout << "Time taken by sequential algorithm: " << end_time - start_time << " seconds" << endl;

    cout << "\n sorted array is=>";
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << endl;
    }


    start_time = omp_get_wtime(); // start timer for parallel algorithm
    parallelBubbleSort(arr_copy);
    end_time = omp_get_wtime(); // end timer for parallel algorithm
    cout << "Time taken by parallel algorithm: " << end_time - start_time << " seconds" << endl;

    cout << "\n sorted array is=>";
    for (int i = 0; i < n; i++)
    {
        cout << arr_copy[i] << endl;
    }


    return 0;
}
