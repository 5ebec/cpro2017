#include <stdio.h>

//Print
void print(int m, int n, const float *x){
  for(int i=0; i<m*n; i++){
    printf("%.4f ", x[i]);
    if((i+1)%n == 0){
      printf("\n");
    }
  }
}

int main(){
  float m[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  print(3, 4, m);
  return 0;
}
