#include "nn.h"

#define L0 784
#define L1 50
#define L2 100
#define L3 10

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

//inference3
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

//inference6
int inference6(const float * A1, const float * b1, const float * A2, const float * b2, const float * A3, const float * b3, const float * x){
  float *y1 = malloc(sizeof(float)*L1);
  float *y2 = malloc(sizeof(float)*L2);
  float *y3 = malloc(sizeof(float)*L3);
  fc(L1, L0, x, A1, b1, y1);
  relu(L1, y1, y1);
  fc(L2, L1, y1, A2, b2, y2);
  relu(L2, y2, y2);
  fc(L3, L2, y2, A3, b3, y3);
  softmax(L3, y3, y3);
  float y3_max = 0;
  int r_y3 = 0;
  for (int r=0; r<L3; r++){
    if(y3_max < y3[r]){
      y3_max = y3[r];
      r_y3 = r;
    }
  }
  return r_y3;
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
  int sum = 0;
  for (int i=0; i<test_count; i++){
    if(inference6(A1_784_50_100_10, b1_784_50_100_10, A2_784_50_100_10, b2_784_50_100_10, A3_784_50_100_10, b3_784_50_100_10, test_x + i*width*height) == test_y[i]){
      sum++;
    }
  }
  printf("%f%%\n", sum*100.0/test_count);
  return 0;
}
