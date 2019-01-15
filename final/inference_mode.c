#include "nn.h"

#define N0 784
#define N1 50
#define N2 100
#define N3 10

//グラフ作成
void graph(int n, const float *x){
  printf("\nnum|probability|graph\n");
  for(int i=0; i<n; i++){
    printf("%3d|%10.1f%%|", i, x[i]*100);
    for(int j=1; j<=x[i]*150; j++){
      printf("=");
    }
    printf("\n");
  }
}

//FC層
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

//活性化層(ReLU関数)
void relu(int n, const float *x, float *y){
  //y={x(x>0),0(x<=0)}
  for (int r=0; r<n; r++){
    y[r] = (x[r]>0)?x[r]:0;
  }
}

//出力層(Softmax関数)
void softmax(int n, const float *x, float *y){
  //y=exp(x-x_max)/sigma[exp(x-x_max)]
  //xの最大値x_maxを用いることによりオーバーフローを防ぐ
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

//inference6
int inference6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, float *y3){
  //A.は重み行列，b.はバイアスベクトル，xは入力ベクトル，y3は出力ベクトル
  //fc(),relu(),softmax()を用いた6層NNにより推論する
  float *y1 = malloc(sizeof(float)*N1);
  float *y2 = malloc(sizeof(float)*N2);
  fc(N1, N0, x, A1, b1, y1);
  relu(N1, y1, y1);
  fc(N2, N1, y1, A2, b2, y2);
  relu(N2, y2, y2);
  fc(N3, N2, y2, A3, b3, y3);
  softmax(N3, y3, y3);
  float y3_max = 0;
  int r_y3 = 0;
  for (int r=0; r<N3; r++){
    if(y3_max < y3[r]){
      y3_max = y3[r];
      r_y3 = r;
    }
  }
  return r_y3;
}

//ファイルをロード
void load(const char * filename, int m, int n, float * A, float * b){
  //filenameは読み込むファイル，A[m*n],b[m]は配列
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

int main(int argc, char const *argv[]){
  float *train_x = NULL;
  unsigned char *train_y = NULL;
  int train_count = -1;
  float *test_x = NULL;
  unsigned char * test_y = NULL;
  int test_count = -1;
  int width = -1;
  int height = -1;
  //nn.hから読み込み
  load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
  //変数の用意
  float *A1 = malloc(sizeof(float)*width*height*N1);
  float *b1 = malloc(sizeof(float)*N1);
  float *A2 = malloc(sizeof(float)*N1*N2);
  float *b2 = malloc(sizeof(float)*N2);
  float *A3 = malloc(sizeof(float)*N2*N3);
  float *b3 = malloc(sizeof(float)*N3);
  float *y3 = malloc(sizeof(float)*N3);
  //コマンドライン引数のデータの読み込み
  float *xx = load_mnist_bmp(argv[1]);
  if(argc == 5){
    load(argv[2], N1, N0, A1, b1);
    load(argv[3], N2, N1, A2, b2);
    load(argv[4], N3, N2, A3, b3);
  }else{
    //コマンドライン引数の数が5でないならエラー
    printf("Error!\nUsage: num.bmp fc1.dat fc2.dat fc3.dat\n");
    exit(1);
  }
  int ans = inference6(A1, b1, A2, b2, A3, b3, xx, y3);
  graph(N3, y3);
  printf("\nThis number is %d.\n\n", ans);
  return 0;
}
