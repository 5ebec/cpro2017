#include "nn.h"

//Fully-Connected layer
void fc(int m, int n, const float *x, const float *A, const float *b, float *y){
  //y=A*x+b
  for (int s=0; s<m; s++){
    y[s] = 0;
    for (int t=0; t<n; t++){
      y[s] += A[s*n+t]*x[t];
    }
    y[s] += b[s];
  }
}

//Rectified Linear Unit
void relu(int n, const float *x, float *y){
  //y={x(x>0),0(x<=0)}
  for (int u=0; u<n; u++){
      y[u] = (x[u]>0)?x[u]:0;
  }
}

//Print
void print(int m, int n, const float *x){
  for (int i=0; i<m*n; i++){
    printf("%f ", x[i]);
  }
}

//test
int main(){
  float *train_x = NULL;
  unsigned char *train_y = NULL;
  int train_count = -1;
  float *test_x = NULL;
  unsigned char * test_y = NULL;
  int test_count = -1;
  int width = -1;
  int height = -1;
  load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
  float *y = malloc(sizeof(float)*10);
  fc(10, 784, train_x, A_784x10, b_784x10, y);
  relu(10, y, y);
  print(1, 10, y);
  return 0;
}
