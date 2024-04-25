#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <stdlib.h>

using namespace std;

const int MAX = 100000;
vector<int> graph[MAX];
bool visited[MAX];

void parallelBFS(int start_node) {
    queue<int> q;
    q.push(start_node);
    visited[start_node] = true;

    while (!q.empty()) {
        int curr_node = q.front();
        q.pop();

        cout << curr_node << " ";

        #pragma omp parallel for
        for (int i = 0; i < graph[curr_node].size(); i++) {
            int adj_node = graph[curr_node][i];
            if (!visited[adj_node]) {
                visited[adj_node] = true;
                q.push(adj_node);
            }
        }
    }
}

int main() {
    int n, m, start_node;
    cout << "Enter number of Nodes, number of Edges, and the Starting Node of the graph:\n";
    cin >> n >> m >> start_node;

    cout << "Enter pairs of nodes representing edges:\n";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    double start_time_bfs = omp_get_wtime(); // start timer for parallel BFS

    cout << "Parallel BFS traversal: ";
    parallelBFS(start_node);
    cout << endl;

    double end_time_bfs = omp_get_wtime(); // end timer for parallel BFS

    cout << "\nTime taken by parallel BFS: " << end_time_bfs - start_time_bfs << " seconds" << endl;


    return 0;
}

// Sequential BFS time calculation and code

// #include <iostream>
// #include <vector>
// #include <queue>
// #include <omp.h>
// #include <stdlib.h>

// using namespace std;

// const int MAX = 100000;
// vector<int> graph[MAX];
// bool visited[MAX];

// void sequentialBFS(int start_node) {
//     queue<int> q;
//     q.push(start_node);
//     visited[start_node] = true;

//     while (!q.empty()) {
//         int curr_node = q.front();
//         q.pop();

//         cout << curr_node << " ";

//         for (int i = 0; i < graph[curr_node].size(); i++) {
//             int adj_node = graph[curr_node][i];
//             if (!visited[adj_node]) {
//                 visited[adj_node] = true;
//                 q.push(adj_node);
//             }
//         }
//     }
// }

// int main() {
//     int n, m, start_node;
//     cout << "Enter number of Nodes, number of Edges, and the Starting Node of the graph:\n";
//     cin >> n >> m >> start_node;

//     cout << "Enter pairs of nodes representing edges:\n";
//     for (int i = 0; i < m; i++) {
//         int u, v;
//         cin >> u >> v;
//         graph[u].push_back(v);
//         graph[v].push_back(u);
//     }

//     for (int i = 0; i < n; i++) {
//         visited[i] = false;
//     }

//     double start_time_bfs = omp_get_wtime(); // start timer for parallel BFS

//     cout << "Sequential BFS traversal: ";
//     sequentialBFS(start_node);
//     cout << endl;

//     double end_time_bfs = omp_get_wtime(); // end timer for parallel BFS

//     cout << "\nTime taken by sequential BFS: " << end_time_bfs - start_time_bfs << " seconds" << endl;


//     return 0;
// }
