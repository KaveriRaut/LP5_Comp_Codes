#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>

using namespace std;

const int MAX = 100000;
vector<int> graph[MAX];
bool visited[MAX];

void dfs(int start_node) {
    stack<int> s;
    s.push(start_node);

    while (!s.empty()) {
        int curr_node;
        #pragma omp critical
        {
            curr_node = s.top();
            s.pop();
        }

        if (!visited[curr_node]) {
            visited[curr_node] = true;
            cout << curr_node << " ";
            
            // Traverse all adjacent vertices
            #pragma omp parallel for
            for (int i = 0; i < graph[curr_node].size(); i++) {
                int adj_node = graph[curr_node][i];
                if (!visited[adj_node]) {
                    #pragma omp critical
                    {
                        s.push(adj_node);
                    }
                }
            }
        }
    }
}

int main() {
    int n, m, start_node;
    cout << "Enter no. of Node, no. of Edges, and Starting Node of graph:\n";
    cin >> n >> m >> start_node;

    cout << "Enter pair of node and edges:\n";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    // Mark all vertices as not visited
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    double start_time_dfs = omp_get_wtime(); // start timer for parallel DFS

    cout << "Parallel DFS traversal: ";
    dfs(start_node);
    cout << endl;

    double end_time_dfs = omp_get_wtime(); // end timer for parallel DFS

    cout << "\nTime taken by parallel DFS: " << end_time_dfs - start_time_dfs << " seconds" << endl;

    return 0;
}


/*output
Enter no. of Node,no. of Edges and Starting Node of graph:
4 3 0
Enter pair of node and edges:
0 1
0 2
2 4
0 2 4 1 
*/



/*
#pragma omp parallel creates a parallel region.
#pragma omp critical is used to ensure that only one thread at a time accesses and modifies the stack s.
Inside the parallel region, each thread will pop a node from the stack if it's not empty and then process it if it hasn't been visited.
If the node hasn't been visited, it will be marked as visited, printed, and its adjacent nodes will be pushed onto the stack if they haven't been visited.
This approach ensures that each thread operates on its own private copy of curr_node, preventing data races.
The critical section around accessing and modifying the stack s ensures that only one thread at a time modifies the stack to avoid inconsistencies.
*/
