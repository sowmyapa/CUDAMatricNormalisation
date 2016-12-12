//
// Created by Sowmya Parameshwara on 11/10/16.
//

/**
 *
 *  1) Input is stored by transposing the matrix, so that the attributes of a column are stored in a single row. This
 *      will optimise the algorithm since all threads in a block will access nearby elements, while normalising.
 *  2) Each row is normalised at a time for calculating standardscore, the calculated values are stored in output matrix by transposing.
 *  3) Number of threads in a block is set as 16 (This value determined by checking performance for different values). The number of blocks
 *     is decided based on matrix size "N" and number of threads.
 *  4) The contents of a row are divided among the blocks. In each block,Each thread populates one elements of the block into shared data.
 *     We then calculate partial sum without divergence, on the data stored in shared memory.
 *  5) Once all blocks compute partial sum, we launch a kernel function on a single block by passing the calculated values from the previous step.
 *     This will calculate the final sum and final squared sum. To this final block we ensure the size of the partial sum array passed equals
        the next nearest power of 2 of "the number of blocks", as partial sum algorithm works only for powers of 2.
 *  6)  The above data is used to calculate standard deviation for that row using the formula ((totalSquareSum + N*powf(mean, 2.0) - 2 * mean * totalSum)/(float)N)
 *  7)  The above value is used to calculate standard score for every element in that row.
 *  8)  The above step repeats for every row, calculating the standard score for all elements in the row.
 *
 *  Steps to compile and execute on Jarvis :
 *  1)  qlogin -q interactive.q  (Launches interactive session).
 *  2)  nvcc matrixNorm.cu -o matrixNorm (Compile code on jarvis).
 *  3)  cd hw4 (Code is available here).
 *  4) ./matrixNorm 15000 4   <Argument 1 : Size of matrix, Argument 2 : Random seed value>
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>

/* Program Parameters */
#define MAXN 15000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices */
volatile float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
	struct timeval t;
	struct timezone tzdummy;

	gettimeofday(&t, &tzdummy);
	return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
	int seed = 0;  /* Random seed */
	char uid[32]; /*User name */

	/* Read command-line arguments */
	srand(time_seed());  /* Randomize */

	if (argc == 3) {
		seed = atoi(argv[2]);
		srand(seed);
		printf("Random seed = %i\n", seed);
	}
	if (argc >= 2) {
		N = atoi(argv[1]);
		if (N < 1 || N > MAXN) {
			printf("N = %i is out of range.\n", N);
			exit(0);
		}
	}
	else {
		printf("Usage: %s <matrix_dimension> [random seed]\n",
				argv[0]);
		exit(0);
	}

	/* Print parameters */
	printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B*/
void initialize_inputs() {
	int row, col;

	printf("\nInitializing...\n");
    for (row = 0; row < N; row++) {
		for (col = 0; col < N; col++) {
			A[row][col] = (float)rand() / 32768.0;
			B[row][col] = 0.0;
		}
	}

}

/* Print input matrices */
void print_inputs() {
	int row, col;

	if (N < 10) {
		printf("\nA =\n\t");
		for (col = 0; col < N; col++) {
			for (row = 0; row < N; row++) {
				printf("%5.2f%s", A[row][col], (row < N-1) ? ", " : ";\n\t");
			}
		}
	}
}

void print_B() {
	int row, col;

	if (N < 10) {
		printf("\nB =\n\t");
		for (row = 0; row < N; row++) {
			for (col = 0; col < N; col++) {
				printf("%1.10f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
			}
		}
	}
}

int main(int argc, char **argv) {
	/* Timing variables */
	struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
	struct timezone tzdummy;
	clock_t etstart2, etstop2;  /* Elapsed times using times() */
	unsigned long long usecstart, usecstop;
	struct tms cputstart, cputstop;  /* CPU times for my processes */

	/* Process program parameters */
	parameters(argc, argv);

	initialize_inputs();

	/* Print input matrices */
	print_inputs();

	/* Start Clock */
	printf("\nStarting clock.\n");
	gettimeofday(&etstart, &tzdummy);
	etstart2 = times(&cputstart);

	/* Gaussian Elimination */
	matrixNorm();

	/* Stop Clock */
	gettimeofday(&etstop, &tzdummy);
	etstop2 = times(&cputstop);
	printf("Stopped clock.\n");
	usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
	usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

	/* Display output */
	print_B();

	/* Display timing results */
	printf("\nElapsed time = %g ms.\n",
			(float)(usecstop - usecstart)/(float)1000);

	printf("(CPU times are accurate to the nearest %g ms)\n",
			1.0/(float)CLOCKS_PER_SEC * 1000.0);
	printf("My total CPU time for parent = %g ms.\n",
			(float)( (cputstop.tms_utime + cputstop.tms_stime) -
				(cputstart.tms_utime + cputstart.tms_stime) ) /
			(float)CLOCKS_PER_SEC * 1000);
	printf("My system CPU time for parent = %g ms.\n",
			(float)(cputstop.tms_stime - cputstart.tms_stime) /
			(float)CLOCKS_PER_SEC * 1000);
	printf("My total CPU time for child processes = %g ms.\n",
			(float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
				(cputstart.tms_cutime + cputstart.tms_cstime) ) /
			(float)CLOCKS_PER_SEC * 1000);
	/* Contrary to the man pages, this appears not to include the parent */
	printf("--------------------------------------------\n");

	exit(0);
}

/**
*  Method to calculate the partial sum without divergence in all the blocks.
*/
__global__ void block_sum(const float *hostInput, float *sumResults, float *squareResults, const size_t n)
{
	__shared__ float sharedSumData[1024];
	__shared__ float sharedSquareData[1024];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	float x = 0;
	if(i < n) {
		x = hostInput[i];
	}
	sharedSumData[tx] = x;
	sharedSquareData[tx] = x*x;
	__syncthreads();

	// block-wide reduction in _shared_ mem
	for(int offset = blockDim.x / 2;
			offset > 0;
			offset >>= 1)
	{
		if(tx < offset)
		{
			sharedSumData[tx] += sharedSumData[tx + offset];
			sharedSquareData[tx] += sharedSquareData[tx + offset];
		}
		__syncthreads();
	}

	// finally, thread 0 writes the calculated result of this block
	if(threadIdx.x == 0)
	{
		// note that the result is per-block
		// not per-thread
		sumResults[blockIdx.x] = sharedSumData[0];
		squareResults[blockIdx.x] = sharedSquareData[0];
	}
}

/**
*  Method to calculate the sum of the results calculated from all the blocks in the previous step.
*/
__global__ void single_block_reduction(float *sumResults, float *squareResults, const size_t n)
{
	__shared__ float sharedSumData[256];
	__shared__ float sharedSquareData[256];

	int tx = threadIdx.x;
	if(tx < n) {
		sharedSumData[tx] = sumResults[tx];
        sharedSquareData[tx] = squareResults[tx];
	}

	__syncthreads();

	// block-wide reduction in _shared_ mem
	for(int offset = n/2;
			offset > 0;
			offset >>= 1)
	{
		if(tx < offset)
		{
			sharedSumData[tx] += sharedSumData[tx + offset];
			sharedSquareData[tx] += sharedSquareData[tx + offset];
		}
		__syncthreads();
	}

	// finally, thread 0 writes the calculated result
	if(threadIdx.x == 0)
	{
		// note that the result is per-block
		// not per-thread
		sumResults[0] = sharedSumData[0];
		squareResults[0] = sharedSquareData[0];
	}
}

float* sum (float *hostInput, size_t n, float *deviceMatrixA, float *sumResults, float *squareResults,int numOfThreadsPerBlock,int numOfBlocks,int next) {

	cudaMemcpy( deviceMatrixA, hostInput, sizeof(float) * N, cudaMemcpyHostToDevice );
   /* dim3 dimGrid(numOfBlocks, 1, 1);
    dim3 dimBlock(numOfThreadsPerBlock, 1, 1);*/


    block_sum<<<numOfBlocks,numOfThreadsPerBlock>>> (deviceMatrixA, sumResults, squareResults, n);

   /* float *sumResultsInput = 0;
    float *squareResultsInput = 0;

    cudaMalloc((void**)&sumResultsInput, sizeof(float) * next);
    cudaMemset(sumResultsInput, 0.0, sizeof(float) * next);

	cudaMemcpy(sumResultsInput, sumResults, sizeof(float) * numOfBlocks , cudaMemcpyDeviceToDevice );
    cudaMalloc((void**)&squareResultsInput, sizeof(float) * next);
    cudaMemset(squareResultsInput, 0.0, sizeof(float) * next);

	cudaMemcpy(squareResultsInput, squareResults, sizeof(float) * numOfBlocks , cudaMemcpyDeviceToDevice ); */

    single_block_reduction<<<1,numOfBlocks>>>(sumResults,squareResults,next);

	float * results = (float *)malloc(sizeof(float) * 2);
	cudaMemcpy(&results[0], &sumResults[0], sizeof(float) , cudaMemcpyDeviceToHost );
	cudaMemcpy(&results[1], &squareResults[0], sizeof(float) , cudaMemcpyDeviceToHost );

	return results;
}


void matrixNorm() {
	int row, col;
	float mu, sigma;

	printf("Computing Parallely.\n");

	size_t numOfThreadsPerBlock = 16;
	size_t numOfBlocks = N/numOfThreadsPerBlock + (((N)%numOfThreadsPerBlock) ? 1 : 0);
	int next = pow(2, ceil(log(numOfBlocks)/log(2)));


	float *sumResults = 0;
	cudaMalloc((void**)&sumResults, sizeof(float) * (next));
    cudaMemset(sumResults, 0.0, sizeof(float) * next);

	float *squareResults = 0;
	cudaMalloc((void**)&squareResults, sizeof(float) * (next));
    cudaMemset(squareResults, 0.0, sizeof(float) * next);

	float *deviceMatrixA = 0;
	cudaMalloc( (void**)&deviceMatrixA, sizeof(float) * N);


	for (row=0; row < N; row++) {
		mu = 0.0;
		float *hostResults;
		hostResults = sum ((float *)A[row], N, deviceMatrixA, sumResults, squareResults,numOfThreadsPerBlock,numOfBlocks,next);
		mu = hostResults[0] / (float) N;
		sigma = (hostResults[1] + N*powf(mu, 2.0) - 2 * mu * hostResults[0])/(float)N;
		for (col=0; col < N; col++) {
			if (sigma == 0.0) {
				B[col][row] = 0.0;
			} else {
				B[col][row] = (A[row][col] - mu) / sigma;
			}
		}
	}
	cudaFree(sumResults);
	cudaFree(squareResults);
	cudaFree(deviceMatrixA);
}
