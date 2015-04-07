#include <stdio.h>
#include <stdlib.h>

// void test1(){
//     int *a = 3; // This initializes the pointer value to 3,
//                 // we do not know what is at this address,
//                 // or if it is accessible, so this is incorrect
//     *a = *a + 2;
//     printf("%d",*a);
// }

// void test1_c(){
//     int *a = (int *) malloc(sizeof(int)); // Allocate memory first
//     *a = 3;
//     *a = *a + 2;
//     printf("%d", *a);
//     free(a); 
// }

// void test2(){
//     int* a,b; // here, b is declared as an int
//               // but we want it to be an int pointer
//     a = (int*) malloc(sizeof(int));
//     b = (int*) malloc(sizeof(int));

//     if (!(a && b)){
//         printf("Out of memory");
//         exit(-1);
//     }
//     *a = 2;
//     *b = 3;
// }

// CORRECTED VERSION
void test2_c(){
    int *a, *b; // now b is an int pointer
    a = (int*) malloc(sizeof(int));
    b = (int*) malloc(sizeof(int));

    if (!(a && b)){
        printf("Out of memory");
        exit(-1);
    }
    *a = 2;
    *b = 3;
    free(a);
    free(b);
}


// void test3(){
//     int i, *a = (int*) malloc(1000);

//     if (!a){
//         printf("Out of memory");
//         exit(-1);
//     }
//     for (i = 0; i < 1000; i++)
//         *(i+a)=i;
//     for (i = 0; i < 1000; ++i)
//     {
//         printf("%d\n", *(a+i));
//     }
// }

void test3_c(){
    // allocate an appropriate size for 1000 ints
    int i, *a = (int*) malloc(1000 * sizeof(int));

    if (!a){
        printf("Out of memory");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        *(a+i)=i; // Do pointer arithmetic to account for size
                  // of pointer value type
    free(a);
}

// void test4(){
//     int **a = (int**) malloc(3*sizeof(int*));
//     a[1][1] = 5;
// }

void test4_c() {
    int **a = (int**) malloc(3*sizeof(int*));

    for (int i = 0; i < 3; ++i) {
        *(a + i) = (int*) malloc(sizeof(int[100]));
        if (! (*(a + i))) {
            printf("Out of memory");
            exit(-1);
        }
    }

    a[1][1] = 5;

    for (int i = 0; i < 3; ++i) {
        free(*(a + i));
    }

    free(a);
}

void test5(){
    int *a = (int*) malloc(sizeof(int));
    scanf("%d",a);
    if (!a) // should be *a == 0, we want the value of the int, not address
        printf("Value is 0\n");
}

void test5_c(){
    int *a = (int*) malloc(sizeof(int));
    // Check if memory allocation succeeded
    if (!a){
        printf("Out of memory");
        exit(-1);
    }

    scanf("%d",a);
    if (*a == 0)
        printf("Value is 0\n");

    free(a);
}
int main(int argc, char const *argv[]) {
    /* code */
    // test1_c();
    // test2_c();
    // test3();
    // test3_c();
    // test4_c();
    // test5();
    test5_c();
    return 0;
}