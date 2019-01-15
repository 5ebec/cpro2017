#include "nn.h"
#include <time.h>

#define L0 784
#define L1 50
#define L2 100
#define L3 10

#define EPOCH 10
#define BATCH 100
#define ETA 0.01



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

//inference3
int inference3(const float *A, const float *b, const float *x){
  float *y = malloc(sizeof(float)*10);
  fc(10, 784, x, A, b, y);
  relu(10, y, y);
  softmax(10, y, y);
  float y_max = 0;
  int ans = 0;
  for (int r=0; r<10; r++){
    if(y_max < y[r]){
      y_max = y[r];
      ans = r;
    }
  }
  return ans;
}

//inference6
int inference6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x){
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
      dx[t] += A[s*n+t]*dy[s];
    }
  }
}

//Backward3
void backward3(const float *A, const float *b, const float *x, unsigned char t, float *y, float *dA, float *db){
  float *y1 = malloc(sizeof(float)*10); // =Ax+b
  float *y2 = malloc(sizeof(float)*10); // ={x(x>0),0(x<=0)}
  fc(10, 784, x, A, b, y1);
  relu(10, y1, y2);
  softmax(10, y2, y);
  float *dx2 = malloc(sizeof(float)*10); // =y-t
  float *dx1 = malloc(sizeof(float)*10); // ={dy(x>0),0(x<=0)}
  float *dx = malloc(sizeof(float)*10); // =A*dy
  softmaxwithloss_bwd(10, y, t, dx2);
  relu_bwd(10, y2, dx2, dx1);
  fc_bwd(10, 784, x, dx1, A, dA, db, dx);
}

//Backward6
void backward6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, unsigned char t, float *y, float *dA1, float *db1, float *dA2, float *db2, float *dA3, float *db3){
  float *y1_1 = malloc(sizeof(float)*L1);
  float *y2_1 = malloc(sizeof(float)*L1);
  float *y1_2 = malloc(sizeof(float)*L2);
  float *y2_2 = malloc(sizeof(float)*L2);
  float *y3 = malloc(sizeof(float)*L3);
  fc(L1, L0, x, A1, b1, y1_1);
  relu(L1, y1_1, y2_1);
  fc(L2, L1, y2_1, A2, b2, y1_2);
  relu(L2, y1_2, y2_2);
  fc(L3, L2, y2_2, A3, b3, y3);
  softmax(L3, y3, y);
  float *dx3 = malloc(sizeof(float)*L3);
  float *dx2 = malloc(sizeof(float)*L2);
  float *dx1 = malloc(sizeof(float)*L1);
  float *dx = malloc(sizeof(float)*L0);
  softmaxwithloss_bwd(L3, y, t, dx3);
  fc_bwd(L3, L2, y2_2, dx3, A3, dA3, db3, dx2);
  relu_bwd(L2, y2_2, dx2, dx2);
  fc_bwd(L2, L1, y2_1, dx2, A2, dA2, db2, dx1);
  relu_bwd(L1, y2_1, dx1, dx1);
  fc_bwd(L1, L0, x, dx1, A1, dA1, db1, dx);
}

//Print
void print(int m, int n, const float *x){
  for (int i=0; i<m*n; i++){
    printf("%f ", x[i]);
  }
}

// Shuffle
void shuffle(int n, int *x){
  for(int i=n-1; i>0; i--){
    int j = rand()%(i+1);
    int t = x[i];
    x[i] = x[j];
    x[j] = t;
  }
}

// Loss
float loss(const float *y, int t){
  return -log(y[t]+1e-7);
}

float cross_entropy_error(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, unsigned char t){
  float *y1e = malloc(sizeof(float)*L1);
  float *y2e = malloc(sizeof(float)*L2);
  float *y3e = malloc(sizeof(float)*L3);
  fc(L1, L0, x, A1, b1, y1e);
  relu(L1, y1e, y1e);
  fc(L2, L1, y1e, A2, b2, y2e);
  relu(L2, y2e, y2e);
  fc(L3, L2, y2e, A3, b3, y3e);
  softmax(L3, y3e, y3e);
  return loss(y3e, t);
}

//Add
void add(int n, const float *x, float *o){
  for(int i=0; i<n; i++){
    o[i] += x[i];
  }
}

//Scale
void scale(int n, float x, float *o){
  for(int i=0; i<n; i++){
    o[i] *= x;
  }
}

//Init
void init(int n, float x, float *o){
  for(int i=0; i<n; i++){
    o[i] = x;
  }
}

//Rand_init
void rand_init(int n, float *o){
  for(int i=0; i<n; i++){
    o[i] = (float)2*rand()/RAND_MAX - 1;
  }
}

//Save
void save(const char * filename, int m, int n, const float * A, const float * b){
  FILE *fp;
  fp = fopen(filename, "w");
  for(int i=0; i< m*n; i++){
    fprintf(fp, "%f ", A[i]);
  }
  for(int i=0; i< m; i++){
    fprintf(fp, "%f ", b[i]);
  }
  fclose(fp);
  printf("File %s saved.\n", filename);
}

//load
void load(const char * filename, int m, int n, float * A, float * b){
  FILE *fp = NULL;
  fp = fopen(filename, "r");
  for(int i=0; i< m*n; i++){
    fscanf(fp, "%f ", &A[i]);
  }
  for(int i=0; i< m; i++){
    fscanf(fp, "%f ", &b[i]);
  }
  fclose(fp);
  printf("file %s loaded.\n", filename);
}

//test
int main(int argc, char const *argv[]) {
  srand(time(NULL));
  float *train_x = NULL;
  unsigned char *train_y = NULL;
  int train_count = -1;
  float *test_x = NULL;
  unsigned char * test_y = NULL;
  int test_count = -1;
  int width = -1;
  int height = -1;
  load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
  float *A1 = malloc(sizeof(float)*width*height*L1);
  float *b1 = malloc(sizeof(float)*L1);
  float *A2 = malloc(sizeof(float)*L1*L2);
  float *b2 = malloc(sizeof(float)*L2);
  float *A3 = malloc(sizeof(float)*L2*L3);
  float *b3 = malloc(sizeof(float)*L3);
  float *xx = load_mnist_bmp(argv[1]);

  load("fc1.dat", L1, L0, A1, b1);
  load("fc2.dat", L2, L1, A2, b2);
  load("fc3.dat", L3, L2, A3, b3);

  printf("%d\n", inference6(A1, b1, A2, b2, A3, b3, xx));

  return 0;
}
