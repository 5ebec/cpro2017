#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void set(int a[], int n){
  srand(time(NULL));
  for(int i=0; i<n; i++){
      a[i] = rand()%10000;
  }
}

void sort(int a[], int n){
  int i,j;
  for (i=1; i<n; i++){
    for (j=0; j<n-i; j++){
      if (a[j] > a[j+1]){
        int temp = a[j];
        a[j] = a[j+1];
        a[j+1] = temp;
      }
    }
  }
}

void show(FILE *fp, int a[], int n){
  fprintf(fp, "Data: ");
  for(int i=0; i<n; i++){
    fprintf(fp, "%d ",a[i]);
  }
  fprintf(fp, "\n");
}

int main(int argc, char const *argv[]){
  int N = 0;
  if(argc == 3){
    char *endptr;
    long M = strtol(argv[1], &endptr, 0);
    if(*endptr || M<0){
      printf("Invalid input\n");
      return 0;
    }
    N = (int)M;
    printf("N: %d\n",N);
    FILE *fp;
    fp = fopen(argv[2], "w");
    if(!fp){
      printf("File cannot open\n");
      return 0;
    }
    int data[N];
    set(data,N);
    sort(data,N);
    show(fp,data,N);
    printf("File output: OK\n");
  }else{
    printf("Invalid input\n");
  }
  return 0;
}
