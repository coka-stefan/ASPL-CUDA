/*
 ============================================================================
 Name        : aspl.cu
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

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
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define max 12393
// #define max 5000

struct distStruct {
	unsigned short *dist;
};


/*
 * A class to read data from a csv file.
 */
 class CSVReader {
	std::string fileName;
	std::string delimeter;

public:
	CSVReader(std::string filename, std::string delm = ",") :
			fileName(filename), delimeter(delm) {
	}

	// Function to fetch data from a CSV File
	std::vector<std::vector<std::string>> getData();
};

/*
 * Parses through csv file line by line and returns the data
 * in vector of vector of strings.
 */
std::vector<std::vector<std::string>> CSVReader::getData() {
	std::ifstream file(fileName);

	std::vector<std::vector<std::string>> dataList;

	std::string line = "";
	// Iterate through each line and split the content using delimeter
	while (getline(file, line)) {
		std::vector<std::string> vec;
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
	for (int v = 0; v < V; v++)
		if (sptSet[v] == false && dist[v] <= min)
			min = dist[v], min_index = v;

	return min_index;
}

// A utility function to print the constructed distance array
__device__
void printSolution(unsigned int dist[], int n, int src) {
	printf("Vertex   Distance from Source %d\n", src);
	for (int i = 0; i < n; i++)
		if (dist[i] != SHRT_MAX)
			printf("%d \t\t %d\n", i, dist[i]);
}

// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
__device__
void dijkstra(short *graph, int src, unsigned long long int *sum,
		int V, unsigned short *dist, bool *sptSet) {

	// unsigned int *dist; // The output array.  dist[i] will hold the shortest
	// // distance from src to i

	// dist = new unsigned int[V];

	// bool *sptSet;
	// sptSet = new bool[V]; // sptSet[i] will true if vertex i is included in shortest
	// path tree or shortest distance from src to i is finalized

	// Initialize all distances as INFINITE and stpSet[] as false
	for (int i = 0; i < V; i++) {
		if (threadIdx.x == 296 && blockIdx.x == 16)
			printf("i = %d\n", i);
		dist[i] = SHRT_MAX;
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
		for (int v = 0; v < V; v++) {
			if (!sptSet[v] && graph[(u * (V - 1)) + v] && dist[u] != INT_MAX
					&& dist[u] + graph[(u * (V - 1)) + v] < dist[v])

				dist[v] = dist[u] + graph[(u * (V - 1)) + v];
		}
	}

	for (int i = 0; i < V; i++) {

		if (dist[i] != SHRT_MAX && dist[i] != 0) {
			// printf("sum add = %d + %d\n", *sum, dist[i]);
			atomicAdd(sum, dist[i]);
		}
	}
	// print the constructed distance array
//      printSolution(dist, V, src);
	// return dist;
}

__global__
void allPaths(short *graph, unsigned long long int *sum, int V, unsigned short *dists, bool* sptSets) {

	// extern __shared__ unsigned short sharedMem[];
	// unsigned short *dist = &sharedMem[threadIdx.x * V];
	unsigned short *dist = &dists[(blockIdx.x * blockDim.x + threadIdx.x) * V];
	bool *sptSet = &sptSets[(blockIdx.x * blockDim.x + threadIdx.x) * V];

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V;
			i += blockDim.x * gridDim.x) {

		dijkstra(graph, i, sum, V, dist, sptSet);

	}
}

// driver program to test above function
int main() {

	printf("Starting...\n");
	// Number of nodes is initially max, while reading in the main matrix
	int V = max;
	short *graph;
	graph = new short[V * V];

	{
		// Creating an object of CSVWriter
		CSVReader reader("adj.csv");

		// Get the data from CSV File
		std::vector<std::vector<std::string>> dataList = reader.getData();

		for (unsigned i = 0; i < dataList.size(); ++i) {
			for (unsigned j = 0; j < dataList[i].size(); ++j) {
				if (i < V && j < V)
					graph[V * i + j] = atoi(dataList[i][j].c_str());
			}
		}
	}

	// int results[3][V];

	std::cout << "Exec starting now..." << std::endl;

    // for(int i = 50; i < max; i+=10) {

	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	short *pGraph;

	CUDA_CHECK_RETURN(cudaMalloc((void **) &pGraph, (V * V) * sizeof(short)));
	CUDA_CHECK_RETURN(cudaMemcpy(pGraph, graph, (V * V) * sizeof(short), cudaMemcpyHostToDevice));

	unsigned long long int sum = 0;
	unsigned long long int *cuda_sum;

	// std::cout << numSMs;

	sum = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &cuda_sum, sizeof(unsigned long long int)));
	CUDA_CHECK_RETURN(cudaMemcpy(cuda_sum, &sum, sizeof(unsigned long long int),
			cudaMemcpyHostToDevice));

	// Change apparent size of matrix
	// V = i;

	int *pV;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &pV, sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemcpy(pV, &V, sizeof(int), cudaMemcpyHostToDevice));


	int gridDim = 32 * numSMs;
	int blockDim = 1024;

	// gridDim = 1;
	// blockDim = 1;

	unsigned int sharedMemSize = V * blockDim * sizeof(unsigned short);
	unsigned int globalArraySize = sharedMemSize * gridDim;
	unsigned short *deviceDist;
	bool *deviceSptSet;
	
	std::cout << "Global mem size = " << globalArraySize << "\n";

	CUDA_CHECK_RETURN(cudaMalloc((void**) &deviceDist, globalArraySize));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &deviceSptSet, globalArraySize));

	// std::cout << "Shared mem size = " << sharedMemSize << "\n";

	// distStruct *hostDist; 
	// hostDist = new distStruct[blockDim * gridDim];
	// distStruct *deviceDist;

	// for(int k = 0; k < blockDim * gridDim; k++) {
	// 	hostDist[k].dist = new unsigned short[V];

	// 	unsigned short *d;

	// 	// size_t * free, *total;
	// 	// cuMemGetInfo(free, total);
	// 	// std::cout << "mem avail " << *free << "\n";
	// 	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, V * sizeof(unsigned short)));
	// 	hostDist[k].dist = d;
	// }

	// CUDA_CHECK_RETURN(cudaMalloc((void**) &deviceDist, blockDim * gridDim * sizeof(distStruct)));
	// CUDA_CHECK_RETURN(cudaMemcpy(deviceDist, hostDist, (V * sizeof(distStruct)), cudaMemcpyHostToDevice));

	std::cout << "Number of nodes = " << V << std::endl;

	std::clock_t start;

	start = std::clock();

	
	allPaths<<<gridDim, blockDim/*, sharedMemSize*/>>>(pGraph, cuda_sum, V, deviceDist, deviceSptSet);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error %d: %s\n", err, cudaGetErrorString(err));
			// break;
	}

	cudaMemcpy(&sum, cuda_sum, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	std::cout << "\nSum = " << sum << std::endl;
	double ms = (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000);
	std::cout << "Time: " << ms << " ms" << std::endl;
	cudaDeviceSynchronize();

	cudaFree(cuda_sum);
	cudaFree(pV);

	// results[0][i] = i;
	// results[1][i] = ms;
	// results[2][i] = sum;

	cudaFree(pGraph);
	
	cudaDeviceSynchronize();

    // }

    // std::ofstream myfile;
    // myfile.open ("results.csv");
    // myfile << "nodes,time,sum\n";

    // for (int i = 0; i < max; i+=500) {
    //     myfile << results[0][i] << ',' << results[1][i] << ',' << results [2][i] << '\n';
    // }
    // myfile.close();

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

