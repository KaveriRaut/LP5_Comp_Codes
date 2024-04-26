#include<iostream>
#include<stdlib.h>
#include<omp.h>
#include <vector>
using namespace std;

// void mergesort(int a[],int i,int j);
// void merge(int a[],int i1,int j1,int i2,int j2);



 
// Function to merge two sorted subarrays
void merge(vector<int> &arr,int i1,int j1,int i2,int j2)
{
    int temp[1000];    
    int i,j,k;
    i=i1;    
    j=i2;    
    k=0;
    
    while(i<=j1 && j<=j2)    
    {
        if(arr[i]<arr[j])
        {
            temp[k++]=arr[i++];
        }
        else
        {
            temp[k++]=arr[j++];
	    }    
    }
    
    while(i<=j1)    
    {
        temp[k++]=arr[i++];
    }
        
    while(j<=j2)    
    {
        temp[k++]=arr[j++];
    }
        
    for(i=i1,j=0;i<=j2;i++,j++)
    {
        arr[i]=temp[j];
    }    
}

// Function to perform Merge Sort recursively
void recursive_mergeSort(vector<int> &arr, int l, int r)
{
    if (l < r)
    {
        int mid = l + (r - l) / 2;

        recursive_mergeSort(arr, l, mid);
        recursive_mergeSort(arr, mid + 1, r);

        merge(arr, l, mid, mid+1, r);
    }
}

// parallel MergeSort algo
void parallel_mergesort(vector<int> &arr,int i,int j)
{
    int mid;
    if(i<j)
    {
        mid=(i+j)/2;
        
        #pragma omp parallel sections 
        {
            #pragma omp section
                parallel_mergesort(arr,i,mid);        
            
            #pragma omp section
                parallel_mergesort(arr,mid+1,j);    
        }

        merge(arr,i,mid,mid+1,j);    
    }
}


int main()
{
    int n,i;
    double start_time, end_time, seq_time, par_time;
    
    cout<<"\n enter total no of elements=>";
    cin>>n;
    vector<int>arr(n);
    vector<int>arr_copy(n);
    cout << "\n enter elements=>";
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
        arr_copy[i]=arr[i];
    }

    // Sequential algorithm
    start_time = omp_get_wtime();
    recursive_mergeSort(arr, 0, n-1);
    end_time = omp_get_wtime();
    seq_time = end_time - start_time;
    cout << "\nSequential Time: " << seq_time << endl;
    
    // Parallel algorithm
    start_time = omp_get_wtime();
    parallel_mergesort(arr_copy,0,n-1);
    end_time = omp_get_wtime();
    par_time = end_time - start_time;
    cout << "\nParallel Time: " << par_time << endl;
    
    cout<<"\n sequentially sorted array is=>";
    for(i=0;i<n;i++)
    {
        cout<<arr[i]<<" ";
    }
    cout<<endl;
    cout<<"\n parallely sorted array is=>";
    for(i=0;i<n;i++)
    {
        cout<<arr_copy[i]<<" ";
    }
       
    return 0;
}
