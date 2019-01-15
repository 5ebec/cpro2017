#include "nn.h"

//Fully-Connected layer
void fc(int m, int n, const float *x, const float *A, const float *b, float *y){
  //y=Ax+b
  for (int r=0; r<m; r++){
    y[r] = 0;
    for (int c=0; c<n; c++){
      y[r] += A[r*n+c]*x[c];
    }
    y[r] += b[r];
  }
}

//Rectified Linear Unit
void relu(int n, const float *x, float *y){
  //y={x(x>0),0(x<=0)}
  for (int r=0; r<n; r++){
    y[r] = (x[r]>0)?x[r]:0;
  }
}

//Softmax
void softmax(int n, const float *x, float *y){
  //y=exp(x-x_max)/sigma[exp(x-x_max)]
  float x_max = 0;
  for (int r=0; r<n; r++){
    x_max = (x_max < x[r])?x[r]:x_max;
  }
  float sum_e = 0;
  for (int r=0; r<n; r++){
    sum_e += exp(x[r]-x_max);
  }
  for (int r=0; r<n; r++){
    y[r] = exp(x[r]-x_max)/sum_e;
  }
}

//Print
void print(int m, int n, const float *x){
  for (int i=0; i<m*n; i++){
    printf("%f ", x[i]);
  }
}

//Softmax With Loss BackWarD
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dx){
  //dx=y-t
  for(int i=0; i<n; i++){
    dx[i] = (i==t)?y[i]-1:y[i];
  }
}

//ReLU BackWarD
void relu_bwd(int n, const float *x, const float *dy, float *dx){
  //dx={dy(x>0),0(x<=0)}
  for (int j=0; j<n; j++){
    dx[j] = (x[j]>0)?dy[j]:0;
  }
}

//FC BackWarD
void fc_bwd(int m, int n, const float *x, const float *dy, const float *A, float *dA, float *db, float *dx){
  //dA=dy*x,db=dy,dx=A*dy

}

//Backward3
void backward3(const float *A, const float *b, const float *x, unsigned char t, float *y, float *dA, float *db){

}

//Inference3
int inference3(const float *A, const float *b, const float *x){
  float *y = malloc(sizeof(float)*10);
  fc(10, 784, x, A, b, y);
  relu(10, y, y);
  softmax(10, y, y);
  float y_max = 0;
  int r_y = 0;
  for (int r=0; r<10; r++){
    if(y_max < y[r]){
      y_max = y[r];
      r_y = r;
    }
  }
  return r_y;
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
  float *dA = malloc(sizeof(float)*784*10);
  float *db = malloc(sizeof(float)*10);
  backward3(A_784x10, b_784x10, train_x + 784*8, train_y[8], y, dA, db);
  print(10, 784, dA);
  print(1, 10, db);
  return 0;
}
