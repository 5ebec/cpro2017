#include "nn.h"

//Print
void print(int m, int n, const float *x){
  int i;
  for (i=0; i<m*n; i++){
    printf("%f ", x[i]);
  }
}

int main(){
  print(1, 10, b_784x10);
  return 0;
}
