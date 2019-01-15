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
  for (int s=0; s<m; s++){
    db[s] = dy[s];
    for (int t=0; t<n; t++){
      dA[s*n+t] = dy[s]*x[t];
    }
  }
  for (int t=0; t<n; t++){
    dx[t] = 0;
    for (int s=0; s<m; s++){
      dx[t] += dA[s*n+t]*dy[s];
    }
  }
}

//Backward3
void backward3(const float *A, const float *b, const float *x, unsigned char t, float *y, float *dA, float *db){
  float *x1 = malloc(sizeof(float)*10); // =Ax+b
  float *x2 = malloc(sizeof(float)*10); // ={x(x>0),0(x<=0)}
  fc(10, 784, x, A, b, x1);
  relu(10, x1, x2);
  softmax(10, x2, y);
  float *dx1 = malloc(sizeof(float)*10); // =y-t
  float *dx2 = malloc(sizeof(float)*10); // ={dy(x>0),0(x<=0)}
  float *dx = malloc(sizeof(float)*10); // =A*dy
  softmaxwithloss_bwd(10, y, t, dx1);
  relu_bwd(10, x1, dx1, dx2);
  fc_bwd(10, 784, x, dx2, A, dA, db, dx);
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
  float *dA = malloc(sizeof(float)*784*10);
  float *db = malloc(sizeof(float)*10);
  backward3(A_784x10, b_784x10, train_x + 784*8, train_y[8], y, dA, db);
  print(10, 784, dA);
  print(1, 10, db);
  return 0;
}
