#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <stdlib.h>

using namespace std;

const int MAX = 100000;
vector<int> graph[MAX];
bool visited[MAX];

void parallelBFS(int source) {
    queue<int> q;

    visited[source] = true;
    q.push(source);

    while (!q.empty()) {
        int u;
        #pragma omp parallel shared(q, visited) 
        {
            #pragma omp single
            {
                u = q.front();
                q.pop();
                cout << u << " ";
            }

            if (!(graph[u].empty())) {
                #pragma omp for
                for (int i = 0; i < graph[u].size(); ++i) {
                    int v = graph[u][i];
                    #pragma omp critical
                    {
                        if (!visited[v]) {
                            visited[v] = true;
                            q.push(v);
                        }
                    }
                }
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
