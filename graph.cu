// A CUDA program for Dijkstra's single source shortest path algorithm.
// The program is for adjacency matrix representation of the graph

#include <stdio.h>
#include <limits.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <stdlib.h>

#define max 12393

/*
 * A class to read data from a csv file.
 */
class CSVReader {
    std::string fileName;
    std::string delimeter;

public:
    CSVReader(std::string filename, std::string delm = ",") :
            fileName(filename), delimeter(delm) {}

    // Function to fetch data from a CSV File
    std::vector <std::vector<std::string>> getData();
};

/*
* Parses through csv file line by line and returns the data
* in vector of vector of strings.
*/
std::vector <std::vector<std::string>> CSVReader::getData() {
    std::ifstream file(fileName);

    std::vector <std::vector<std::string>> dataList;

    std::string line = "";
    // Iterate through each line and split the content using delimeter
    while (getline(file, line)) {
        std::vector <std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        dataList.push_back(vec);
    }
    // Close the File
    file.close();

    return dataList;
}


// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
__host__ __device__

unsigned int minDistance(unsigned int dist[], bool sptSet[], int V) {
    // Initialize min value
    unsigned int min = INT_MAX, min_index;

    for (int v = 0; v < V; v++)
        if (sptSet[v] == false && dist[v] <= min)
            min = dist[v], min_index = v;

    return min_index;
}

// A utility function to print the constructed distance array
__host__ __device__
void printSolution(unsigned int dist[], int n, int src, int V)
{
   printf("Vertex   Distance from Source %d\n", src);
   for (int i = 0; i < V; i++)
      if(dist[i]!=INT_MAX)
       printf("%d \t\t %d\n", i, dist[i]);
}

// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
__device__
unsigned int *dijkstra(short **graph, int src, unsigned long long int *sum, int V) {
    unsigned int *dist; // The output array.  dist[i] will hold the shortest
    // distance from src to i

    dist = new unsigned int[V];

    bool *sptSet;
    sptSet = new bool[V]; // sptSet[i] will true if vertex i is included in shortest
    // path tree or shortest distance from src to i is finalized

    // Initialize all distances as INFINITE and stpSet[] as false
    for (int i = 0; i < V; i++)
    {
        dist[i] = INT_MAX;
        sptSet[i] = false;
    }

    // Distance of source vertex from itself is always 0
    dist[src] = 0;

    // Find shortest path for all vertices
    for (int count = 0; count < V - 1; count++) {
        // Pick the minimum distance vertex from the set of vertices not
        // yet processed. u is always equal to src in first iteration.
        unsigned int u = minDistance(dist, sptSet, V);

        // Mark the picked vertex as processed
        sptSet[u] = true;

        // Update dist value of the adjacent vertices of the picked vertex.
        for (int v = 0; v < V; v++)

            // Update dist[v] only if is not in sptSet, there is an edge from
            // u to v, and total weight of path from src to  v through u is
            // smaller than current value of dist[v]
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX
                && dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }
    
    for (int i = 0; i < V; i++)
    {
        
        if (dist[i] != INT_MAX)
            atomicAdd(sum, dist[i]);

    }
    // print the constructed distance array
//      printSolution(dist, V, src);
    return dist;
}

__global__
void allPaths(short **graph, unsigned long long int *sum, int V) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < V; 
         i += (blockDim.x * gridDim.x)) {
        dijkstra(graph, i, sum, V);
    }
}

// driver program to test above function
int main() {

    // Number of nodes is initially max, while reading in the main matrix
    int V = max;
    short **graph;
    graph = new short *[V];
    for (int i = 0; i < V; i++) {
        graph[i] = new short[V];
    }

    {
        // Creating an object of CSVWriter
        CSVReader reader("adj.csv");

        // Get the data from CSV File
        std::vector <std::vector<std::string>> dataList = reader.getData();

        for (unsigned i = 0; i < dataList.size(); ++i) {
            for (unsigned j = 0; j < dataList[i].size(); ++j) {
                if(i<V && j<V)
                graph[i][j] = atoi(dataList[i][j].c_str());
            }
        }
    }

    for(int i = 10; i < max; i *= 10) {

        // Change apparent size of matrix
        V = i;

        short **pGraph;

        cudaMalloc((void **) &pGraph, (V * V) * sizeof(short));
        cudaMemcpy(pGraph, graph, (V * V) * sizeof(short), cudaMemcpyHostToDevice);
        
        unsigned long long int sum = 0;
        unsigned long long int *cuda_sum;
        
        cudaMalloc((void**)&cuda_sum, sizeof(unsigned long long int));
        cudaMemcpy(cuda_sum, &sum, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    
        // cudaMalloc((void**)&pDist, (V*V)*sizeof(int));
        // cudaMemcpy(pDist, graph, (V*V)*sizeof(int), cudaMemcpyHostToDevice);

        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

        // std::cout << numSMs;

        dim3 threadsPerBlock(V, V);
        std::clock_t start;

        start = std::clock();
        std::cout << "Parallel exec starting now:" << std::endl;
    
        std::cout << "Number of nodes = " << V << std::endl;

        allPaths <<< 32 * numSMs, 512 >>> (pGraph, cuda_sum, V);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        std::cout<<err;

        cudaDeviceSynchronize();
        
        cudaFree(pGraph);

        cudaMemcpy(&sum, cuda_sum, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaFree(cuda_sum);
        
        std::cout << "\nSum = " << sum << std::endl;
        std::cout << "Time: " << (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    }

    
//    std::cout << "sync exec now:" << std::endl;
    //allSyncPaths(graph);
    return 0;
}
