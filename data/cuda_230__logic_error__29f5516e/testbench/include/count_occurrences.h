#ifndef COUNT_OCCURRENCES_H
#define COUNT_OCCURRENCES_H

__global__ void k_countOccurrences(int * array1_d, 
                                   int * array2_d, 
                                   int len1, 
                                   int len2,
                                   int * count_d);

#endif // COUNT_OCCURRENCES_H