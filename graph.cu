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
#include <iostream>
#include <fstream>

// #define max 12393
#define max 500
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
        if(dataList.size() > max) break;
    }
    // Close the File
    file.close();

    return dataList;
}


// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
__device__
unsigned short minDistance(unsigned short dist[], bool sptSet[], int V) {
    // Initialize min value
    unsigned short min = SHRT_MAX, min_index;
    for (short v = 0; v < V; v++)
        if (sptSet[v] == false && dist[v] <= min)
            min = dist[v], min_index = v;

    return min_index;
}

// A utility function to print the constructed distance array
__device__
void printSolution(unsigned int dist[], int n, int src)
{
   printf("Vertex   Distance from Source %d\n", src);
   for (int i = 0; i < n; i++)
      if(dist[i]!=INT_MAX)
       printf("%d \t\t %d\n", i, dist[i]);
}

// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
__device__
void dijkstra(short *graph, int src, unsigned long long int *sum, int V) {

    unsigned short *dist; // The output array.  dist[i] will hold the shortest
    // distance from src to i

    dist = new unsigned short[V];

    if(threadIdx.x == 96 && blockIdx.x == 24) {
        printf("dist[0] = %d\n", dist[0]);
    }
            
            
    bool *sptSet;
    sptSet = new bool[V]; // sptSet[i] will true if vertex i is included in shortest
    // path tree or shortest distance from src to i is finalized

    // Initialize all distances as INFINITE and stpSet[] as false
    for (short i = 0; i < V; i++)
    {
        if(threadIdx.x == 96 && blockIdx.x == 24) {
            printf("i100 = %d\n", i);
            printf("dist[i] = %d\n", dist[i]);
            }
        dist[i] = SHRT_MAX;
        sptSet[i] = false;
    }

    // Distance of source vertex from itself is always 0
    dist[src] = 0;
    // Find shortest path for all vertices
    for (short count = 0; count < V - 1; count++) {

        // Pick the minimum distance vertex from the set of vertices not
        // yet processed. u is always equal to src in first iteration.
        unsigned int u = minDistance(dist, sptSet, V);

        
        // Mark the picked vertex as processed
        sptSet[u] = true;

        // Update dist value of the adjacent vertices of the picked vertex.
        for (int v = 0; v < V; v++)
        {          
            if (!sptSet[v] && 
                graph[(u * (V-1)) + v] && 
                dist[u] != SHRT_MAX &&
                dist[u] + graph[(u * (V-1)) + v] < dist[v])
                
                dist[v] = dist[u] + graph[(u * (V-1)) + v];
        }
    }
    
    
    for (short i = 0; i < V; i++)
    {
        if (dist[i] != SHRT_MAX)
            atomicAdd(sum, dist[i]);

    }
    // print the constructed distance array
//      printSolution(dist, V, src);
}

__global__
void allPaths(short *graph, unsigned long long int *sum, int V) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < V; 
         i += blockDim.x * gridDim.x) {
         
//         if (threadIdx.x == 992 && blockIdx.x == 0) printf ("i = %d\n", i);
        dijkstra(graph, i, sum, V);

    }
}

// driver program to test above function
int main() {

    
    printf("Starting...\n");
    // Number of nodes is initially max, while reading in the main matrix
    int V = max;
    short *graph;
    graph = new short [V*V];

    {
        // Creating an object of CSVWriter
        CSVReader reader("adj.csv");

        // Get the data from CSV File
        std::vector <std::vector<std::string>> dataList = reader.getData();

        for (unsigned i = 0; i < dataList.size(); ++i) {
            for (unsigned j = 0; j < dataList[i].size(); ++j) {
                if(i<V && j<V)
                graph[V * i + j] = atoi(dataList[i][j].c_str());
            }
        }
    }
    

    int results[3][V];
    
    std::cout << "Parallel exec starting now..." << std::endl;

//     for(int i = 500; i < max; i+=500) {
    
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
       
        short *pGraph;

        cudaMalloc((void **) &pGraph, (V * V) * sizeof(short));
        cudaMemcpy(pGraph, graph, (V * V) * sizeof(short), cudaMemcpyHostToDevice);

        unsigned long long int sum = 0;
        unsigned long long int *cuda_sum;

        // std::cout << numSMs;
    
        sum = 0;
        cudaMalloc((void**)&cuda_sum, sizeof(unsigned long long int));
        cudaMemcpy(cuda_sum, &sum, sizeof(unsigned long long int), cudaMemcpyHostToDevice);


// Change apparent size of matrix
//         V = i;

        int *pV;
        cudaMalloc((void**)&pV, sizeof(int));
        cudaError_t s = cudaMemcpy(pV, &V, sizeof(int), cudaMemcpyHostToDevice);
        
        std::cout << "Number of nodes = " << V << std::endl;

        std::clock_t start;

        start = std::clock();
        
        allPaths <<< 32 * numSMs, 1024 >>> (pGraph, cuda_sum, V);
//         allPaths <<<1,1>>>(pGraph, cuda_sum, V);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error %d: %s\n", err, cudaGetErrorString(err));
//             break;
        }
        
        cudaMemcpy(&sum, cuda_sum, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

        
        std::cout << "\nSum = " << sum << std::endl;
        double ms = (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000);
        std::cout << "Time: " << ms << " ms" << std::endl;
        cudaDeviceSynchronize();
        
        cudaFree(cuda_sum);
        cudaFree(pV);
        
//         results[0][i] = i;
//         results[1][i] = ms;
//         results[2][i] = sum;
        
        cudaDeviceSynchronize();
        
        cudaFree(pGraph);

//     }

//     std::ofstream myfile;
//     myfile.open ("results.csv");
//     myfile << "nodes,time,sum\n";
    
//     for (int i = 0; i < max; i+=500) {
//         myfile << results[0][i] << ',' << results[1][i] << ',' << results [2][i] << '\n';
//     }
//     myfile.close();
    
    
    return 0;
}
