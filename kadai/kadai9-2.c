#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10

void set(int a[], int n){
  srand(time(NULL));
  for(int i=0; i<n; i++){
      a[i] = rand()%10000;
  }
}

void swap(int *na, int *nb){
  int temp = *na;
  *na = *nb;
  *nb = temp;
}

void sort(int a[], int n){
  int i,j;
  for (i=1; i<n; i++){
    for (j=0; j<n-i; j++){
      if (a[j] > a[j+1]){
        swap(&a[j],&a[j+1]);
      }
    }
  }
}

void show(int a[]){
  printf("Data: ");
  for(int i=0; i<N; i++){
    printf("%d ",a[i]);
  }
  printf("\n");
}

int main(void){
  int data[N];
  set(data,N);
  printf("N: %d\n",N);
  show(data);
  sort(data,N);
  show(data);
  return 0;
}
