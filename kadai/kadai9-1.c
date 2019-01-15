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
  return 0;
}
