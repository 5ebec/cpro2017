#include <stdio.h>

#define NUMBER 6

void swap(int *na, int *nb){
  int temp = *na;
  *na = *nb;
  *nb = temp;
}

void bsort (int a[], int n){
  int i,j;
  for (i=1; i<n; i++){
    for (j=0; j<n-i; j++){
      if (a[j] > a[j+1]){
        swap(&a[j],&a[j+1]);
      }
    }
    printf("loop%d: ",i);
    for (int k=0;k<n;k++){
      printf("%d ",a[k]);
    }
    printf("\n");
  }
}

int main(void){
  int i;
  int data[NUMBER]={64,30,8,87,45,13};
  printf("Data: ");
  for (i=0; i<NUMBER; i++){
    printf("%d ",data[i]);
  }
  printf("\nN: %d\n\n",NUMBER);
  bsort(data,NUMBER);
  return 0;
}
