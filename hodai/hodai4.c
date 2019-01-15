#include "nn.h"

//Fully-Connected layer
void fc(int m, int n, const float *x, const float *A, const float *b, float *y){
  //y=Ax+b
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
  for (int t=0; t<n; t++){
    if(x[t]>0){
      y[t] = x[t];
    }else{
      y[t] = 0;
    }
  }
}

//Softmax
void softmax(int n, const float *x, float *y){
  //y=exp(x-x_max)/sigma[exp(x-x_max)]
  float x_max = 0;
  for (int t=0; t<n; t++){
    x_max = (x_max < x[t])?x[t]:x_max;
  }
  float total = 0;
  for (int t=0; t<n; t++){
    total += exp(x[t]-x_max);
  }
  for (int t=0; t<n; t++){
    y[t] = exp(x[t]-x_max)/total;
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
  softmax(10, y, y);
  print(1, 10, y);
  return 0;
}
