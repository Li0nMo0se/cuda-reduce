#include "reduce.cuh"

#include <stdio.h>


int main()
{
    printf("OK\n");
    int a = 7;
    reduce_add(&a, &a, 1);
    return 1;
}