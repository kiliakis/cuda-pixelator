#include "cuda.cuh"

#include <cstring>
#include <cstdlib>

#include "helper.h"

///
/// Algorithm storage
///
// Host copy of input image
Image cuda_input_image;

// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;

// unsigned char* d_output_global_average;

// Pointers to host memory to use for validatio
Image cuda_output_image;
unsigned long long* cpu_mosaic_sum;
unsigned char* cpu_mosaic_value;
unsigned char* cpu_output_image_data;
// This is for kernel_stage1
const int BLOCK_SIZE_Y = 4;
// this is for kernel_stage2
const int BLOCK_SIZE = 128;

void cuda_begin(const Image *input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));

    const size_t image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image.channels = input_image->channels;
    cuda_input_image.width = input_image->width;
    cuda_input_image.height = input_image->height;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    cuda_output_image.channels = input_image->channels;
    cuda_output_image.width = input_image->width;
    cuda_output_image.height = input_image->height;
    cuda_output_image.data = (unsigned char*)malloc(image_data_size);
    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));

    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));
    // CUDA_CALL(cudaMalloc(&d_output_global_average, input_image->channels * sizeof(unsigned char)));

    // Allocate buffer for calculating the sum of each tile mosaic
    cpu_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    cpu_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char));

    // Allocate buffer for storing the output pixel value of each tile
    // cpu_output_image_data = (unsigned char*)malloc(image_data_size);


}

// I need a 1d grid, 1 block per tile
// The thread blocks are 2d (4-8-16-32 x 32)
// each thread might do one or more elements
// the thread block sum is in shared memory
// threads first sum their data privately and then update the thread block value
// atomically
// I find tile index and tile offset
// then do a loop per thread, threads should access in-order the input elements

__global__ void kernel_stage1(
    unsigned char *image, int width, int channels,
    unsigned long long *mosaic_sum)
{
    const unsigned int tile_index = blockIdx.x * channels;
    const unsigned int tile_offset = blockIdx.x * TILE_SIZE * TILE_SIZE * channels;
    
    __shared__ unsigned char block_pixel[4];

    unsigned char thread_pixel[4];
    // unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned pixel_offset = (threadIdx.y * width + threadIdx.x) * channels;
    while (pixel_offset < (TILE_SIZE * width) * channels) {
        thread_pixel[0] += image[tile_offset + pixel_offset + 0];
        thread_pixel[1] += image[tile_offset + pixel_offset + 1];
        thread_pixel[2] += image[tile_offset + pixel_offset + 2];
        // thread_pixel[3] += image[tile_offset + pixel_offset + 3];

        // only works when blockDim.x == TILE_SIZE
        pixel_offset += blockDim.y * width * channels;
    }

    // Now I need to add to block_pixel
    atomicAdd((unsigned int *)&(block_pixel[0]), (unsigned int) thread_pixel[0]);
    atomicAdd((unsigned int *)&(block_pixel[1]), (unsigned int) thread_pixel[1]);
    atomicAdd((unsigned int *)&(block_pixel[2]), (unsigned int) thread_pixel[2]);
    // atomicAdd(&(block_pixel[3]), thread_pixel[3]);
    __syncthreads();

    // Then copy to global mosaic sum
    if (threadIdx.x < channels && threadIdx.y == 0) {
        mosaic_sum[tile_index + threadIdx.x] = block_pixel[threadIdx.x];
    }

}


// Here we have one thread per TILE
// Both grid and thread block are 1-d
__global__ void kernel_stage2(
    unsigned long long *mosaic_sum, int channels,
    int tiles_x, int tiles_y,
    unsigned char * mosaic_value,
    unsigned long long *global_pixel_sum) 
{
    const unsigned int tile_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ unsigned long long block_sum[4];

    mosaic_value[tile_index + 0] = (unsigned char) (mosaic_sum[tile_index + 0] / TILE_PIXELS);
    mosaic_value[tile_index + 1] = (unsigned char) (mosaic_sum[tile_index + 1] / TILE_PIXELS);
    mosaic_value[tile_index + 2] = (unsigned char) (mosaic_sum[tile_index + 2] / TILE_PIXELS);
    // mosaic_value[tile_index + 3] = (unsigned char) (mosaic_sum[tile_index + 3] / TILE_PIXELS);

    atomicAdd((unsigned int *)&(block_sum[0]), (unsigned int) mosaic_value[tile_index + 0]);
    atomicAdd((unsigned int *)&(block_sum[1]), (unsigned int) mosaic_value[tile_index + 1]);
    atomicAdd((unsigned int *)&(block_sum[2]), (unsigned int) mosaic_value[tile_index + 2]);
    // atomicAdd(&(block_sum[3]), mosaic_value[tile_index + 3]);

    __syncthreads();

    if (threadIdx.x < channels){
        // atomicAdd(&(d_output_global_average[threadIdx.x]), block_sum[threadIdx.x]/(cuda_TILES_X * cuda_TILES_Y));
        atomicAdd(&(global_pixel_sum[threadIdx.x]), block_sum[threadIdx.x]/(tiles_x * tiles_y));
    }
}


// This assumes one output image pixel per thread
// and one block per tile
__global__ void kernel_stage3(
    unsigned char *mosaic_value, int width, int channels,
    unsigned char *output_image) 
{
    const unsigned int tile_index = blockIdx.x * channels;
    // const unsigned int tile_offset = blockIdx.x * TILE_SIZE * TILE_SIZE * channels;
    
    __shared__ unsigned char block_pixel[4];

    if (threadIdx.x < channels){
        block_pixel[threadIdx.x] = mosaic_value[tile_index + threadIdx.x];
    }

    __syncthreads();

    unsigned pixel_offset = (threadIdx.y * width + threadIdx.x) * channels;

    output_image[tile_index + pixel_offset +0] = block_pixel[0];
    output_image[tile_index + pixel_offset +1] = block_pixel[1];
    output_image[tile_index + pixel_offset +2] = block_pixel[2];

    // while (pixel_offset < (TILE_SIZE * width) * channels) {
    //     thread_pixel[0] += d_input_image_data[tile_offset + pixel_offset + 0];
    //     thread_pixel[1] += d_input_image_data[tile_offset + pixel_offset + 1];
    //     thread_pixel[2] += d_input_image_data[tile_offset + pixel_offset + 2];
    //     // thread_pixel[3] += d_input_image_data[tile_offset + pixel_offset + 3];

    //     // only works when blockDim.x == TILE_SIZE
    //     pixel_offset += blockDim.y * width * channels;
    // }

}


void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);
    // call the kernel, make sure the x dimension is equal to TILE_SIZE
    int grid_size = cuda_TILES_X * cuda_TILES_Y;
    dim3 block_size(TILE_SIZE, BLOCK_SIZE_Y);
    kernel_stage1<<< grid_size, block_size>>>(cuda_input_image.data, cuda_input_image.width, cuda_input_image.channels, d_mosaic_sum);
#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    CUDA_CALL(cudaMemcpy(cpu_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    validate_tile_sum(&cuda_input_image, cpu_mosaic_sum);
#endif
}



void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);
    int grid_size = (cuda_TILES_X * cuda_TILES_Y + BLOCK_SIZE -1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE);
    kernel_stage2<<< grid_size, block_size>>>(
        d_mosaic_sum, cuda_input_image.channels,
        cuda_TILES_X, cuda_TILES_Y,
        d_mosaic_value, d_global_pixel_sum);
#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    unsigned long long global_pixel_sum[4];
    CUDA_CALL(cudaMemcpy(cpu_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(global_pixel_sum, d_global_pixel_sum, cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for (int i = 0; i < cuda_input_image.channels; ++i) {
        output_global_average[i] = (unsigned char)(global_pixel_sum[i]/TILE_PIXELS);
    }
    validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, cpu_mosaic_sum, cpu_mosaic_value, output_global_average);
#endif    
}

// again one tile per 
void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);
    int grid_size = cuda_TILES_X * cuda_TILES_Y;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    kernel_stage3<<< grid_size, block_size>>>(
        d_mosaic_value, cuda_output_image.width, cuda_output_image.channels,
        d_output_image_data);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    CUDA_CALL(cudaMemcpy(cuda_output_image.data, d_output_image_data, cuda_output_image.width * cuda_output_image.height * cuda_output_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    validate_broadcast(&cuda_input_image, cpu_mosaic_value, &cuda_output_image);
#endif    
}


void cuda_end(Image *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    output_image->channels = cuda_input_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_mosaic_sum));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_global_pixel_sum));

    free(cuda_output_image.data);
    free(cpu_mosaic_sum);
    free(cpu_mosaic_value);
    free(cpu_output_image_data);
}
