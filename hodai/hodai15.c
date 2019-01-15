#include "nn.h"
#include <time.h>

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

//test
int main(){
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
  float *A = malloc(sizeof(float)*width*height*10);
  float *b = malloc(sizeof(float)*10);
  float *y = malloc(sizeof(float)*10);
  float *dA = malloc(sizeof(float)*width*height*10);
  float *db = malloc(sizeof(float)*10);
  float *dA_avg = malloc(sizeof(float)*width*height*10);
  float *db_avg = malloc(sizeof(float)*10);
  rand_init(width*height*10, A);
  rand_init(10, b);
  int *index = malloc(sizeof(int)*train_count);
  int *index_mini = malloc(sizeof(int)*BATCH);
  for(int i=0; i<train_count; i++){
    index[i] = i;
  }
  for(int j=0; j<EPOCH; j++){
    shuffle(train_count, index);
    for(int k=0; k<train_count/BATCH; k++){
      init(width*height*10, 0, dA_avg);
      init(10, 0, db_avg);
      for(int l=0; l<BATCH; l++){
        index_mini[l] = index[k*BATCH+l];
        backward3(A, b, train_x + width*height*index_mini[l], train_y[index_mini[l]], y, dA, db);
        add(width*height*10, dA, dA_avg);
        add(10, db, db_avg);
      }
      scale(width*height*10, -ETA/BATCH, dA_avg);
      scale(10, -ETA/BATCH, db_avg);
      add(width*height*10, dA_avg, A);
      add(10, db_avg, b);
    }
    float sum = 0;
    for(int m=0; m<test_count; m++){
      if(inference3(A, b, test_x + width*height*m) == test_y[m]){
        sum++;
      }
    }
    float sum_e = 0;
    for(int n=0; n<test_count; n++){
      float *y_e = malloc(sizeof(float)*10);
      fc(10, 784, test_x + width*height*n, A, b, y_e);
      relu(10, y_e, y_e);
      softmax(10, y_e, y_e);
      sum_e += loss(y_e, test_y[n]);
    }
    printf("Epoch %2d: 正解率 %f%%, 損失関数 %f\n", j+1, sum*100.0/test_count, sum_e/test_count);
  }
  return 0;
}
