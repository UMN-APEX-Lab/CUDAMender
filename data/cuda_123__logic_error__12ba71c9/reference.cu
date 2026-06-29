#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "image_erosion.h"
#include "erosion_constants.h"

using namespace cooperative_groups;

__global__ void k_imageErosion(const unsigned char *inputImage_d, const unsigned char *structuringElement_d, unsigned char *outputImage_d, unsigned char *outputImageIntermediateBuffer_d, int inputImageWidth, int inputImageHeight, int structuringElementWidth, int structuringElementHeight, int numberOfErosionIterations) {

    //compute index
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    //Compute row and column of the image
    int inputImageRowIndex = threadId / inputImageWidth;
    int inputImageColumnIndex = threadId % inputImageWidth;
    int pixelIndex = inputImageRowIndex * inputImageWidth + inputImageColumnIndex;

    //Compute Structuring Element width, height offset
    int structuringElementWidthOffset = (floorf)((structuringElementWidth - 1) / 2);
    int structuringElementHeightOffset = (floorf)((structuringElementHeight - 1) / 2);

    //Initializations
    int structuringElementRow = 0;
    int structuringElementColumn = 0;
    unsigned char outputValue = 0;

    //Only Process pixels that are within the boundary
    if((pixelIndex >= MIN_IMAGE_PIXEL_INDEX) && (pixelIndex < (inputImageWidth * inputImageHeight))) {
        //Erosion for the first iteration
        outputValue = inputImage_d[pixelIndex];
        for(int rowIndex = (inputImageRowIndex - structuringElementHeightOffset); rowIndex <= (inputImageRowIndex + structuringElementHeightOffset); rowIndex++) {
            for(int columnIndex = (inputImageColumnIndex - structuringElementWidthOffset); columnIndex <= (inputImageColumnIndex + structuringElementWidthOffset); columnIndex++ ) {
                if((rowIndex >= MIN_IMAGE_ROW_INDEX) && (rowIndex < inputImageHeight)) {
                    if((columnIndex >= MIN_IMAGE_COLUMN_INDEX) && (columnIndex < inputImageWidth)) {
                        if(structuringElement_d[structuringElementRow * structuringElementWidth +  structuringElementColumn] == FOREGROUND_PIXEL) {
                           outputValue = min(outputValue, inputImage_d[rowIndex * inputImageWidth + columnIndex]);
                        }
                    }
                }
                structuringElementColumn++;  
            }
            structuringElementRow++;
            structuringElementColumn = 0;
        }
        outputImage_d[pixelIndex] = outputValue;
    }

    //All threads shall participate in synchronization
    grid.sync();

    //Erosion in successive iterations
    for(int iteration = SECOND_ITERATION; iteration < numberOfErosionIterations; iteration++) {
        //Only Process pixels that are within the boundary
        if((pixelIndex >= MIN_IMAGE_PIXEL_INDEX) && (pixelIndex < (inputImageWidth * inputImageHeight))) {
            structuringElementRow = 0;
            structuringElementColumn = 0;
            unsigned char* input_d;
            unsigned char* output_d;

            if(iteration % 2 == 1) {
                input_d  = outputImage_d;
                output_d = outputImageIntermediateBuffer_d; 
            } else {
                input_d  = outputImageIntermediateBuffer_d;
                output_d = outputImage_d;
            }

            outputValue = input_d[pixelIndex];
            for(int rowIndex = (inputImageRowIndex - structuringElementHeightOffset); rowIndex <= (inputImageRowIndex + structuringElementHeightOffset); rowIndex++) {
                for(int columnIndex = (inputImageColumnIndex - structuringElementWidthOffset); columnIndex <= (inputImageColumnIndex + structuringElementWidthOffset); columnIndex++ ) {
                    if((rowIndex >= MIN_IMAGE_ROW_INDEX) && (rowIndex < inputImageHeight)) {
                        if((columnIndex >= MIN_IMAGE_COLUMN_INDEX) && (columnIndex < inputImageWidth)) {
                            if(structuringElement_d[structuringElementRow * structuringElementWidth +  structuringElementColumn] == FOREGROUND_PIXEL) {
                               outputValue = min(outputValue, input_d[rowIndex * inputImageWidth + columnIndex]);
                            }
                        }
                    }
                    structuringElementColumn++;  
                }
        
                structuringElementRow++;
                structuringElementColumn = 0;
            }
        
            output_d[pixelIndex] = outputValue;
        }

        //All threads shall participate in synchronization
        grid.sync();
    }

    //Only Process pixels that are within the boundary
    if((pixelIndex >= MIN_IMAGE_PIXEL_INDEX) && (pixelIndex < (inputImageWidth * inputImageHeight))) {
        //Update the final output in outputImage_d, only if the number of iterations are even
        //Else the output is already available in outputImage_d
        if(numberOfErosionIterations % 2 == FALSE) {
            outputImage_d[pixelIndex] = outputImageIntermediateBuffer_d[pixelIndex];
        }
    }
}