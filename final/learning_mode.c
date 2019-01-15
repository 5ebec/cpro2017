#include "nn.h"
#include "MT.h" //乱数にメルセンヌ=ツイスタを用いる。
#include <time.h>

#define N0 784
#define N1 50
#define N2 100
#define N3 10

#define EPOCH 15 //エポック回数
#define BATCH 100 //ミニバッチサイズ
#define ETA 0.01 //学習率

//配列の加算
void add(int n, const float *x, float *o){
  //配列oに配列xを足す。
  for(int i=0; i<n; i++){
    o[i] += x[i];
  }
}

//配列の乗算
void scale(int n, float x, float *o){
  //配列o[n]の要素にxを掛ける。
  for(int i=0; i<n; i++){
    o[i] *= x;
  }
}

//配列の初期化
void init(int n, float x, float *o){
  //配列o[n]の要素をxに置き換える。
  for(int i=0; i<n; i++){
    o[i] = x;
  }
}

//一様分布乱数による配列初期化
void rand_init(int n, float *o){
  //配列の要素を[-1:1]の一様分布乱数に置き換える。
  for(int i=0; i<n; i++){
    o[i] = genrand_real1()*2 - 1.0;
  }
}

//正規分布乱数
double rand_normal(double mu, double sigma){
  //ボックス=ミュラー法
  double z = sqrt(-2.0*log(genrand_real3())) * sin(2.0*M_PI*genrand_real3());
  return mu + sigma*z;
}

//Heの初期値による配列初期化
void he_init(int n, float *o){
  //配列の要素を，平均値0,標準偏差√(2/n)とする正規分布による乱数に置き換える。
  for(int i=0; i<n; i++){
    o[i] = rand_normal(0, sqrt(2.0/n));
  }
}

//標準偏差0.01の正規分布乱数による配列初期化
void norm_init(int n, float *o){
  //配列の要素を，平均値0,標準偏差0.01とする正規分布による乱数に置き換える。
  for(int i=0; i<n; i++){
    o[i] = rand_normal(0, 0.01);
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

//6層NNによる推論
int inference6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x){
  //A.は重み行列，b.はバイアスベクトル，xは入力ベクトル
  //fc(),relu(),softmax()を用いた6層NNにより推論する
  float *y1 = malloc(sizeof(float)*N1);
  float *y2 = malloc(sizeof(float)*N2);
  float *y3 = malloc(sizeof(float)*N3);
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

//誤差逆伝播法(Softmax層)
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dx){
  //dx=y-t
  for(int i=0; i<n; i++){
    dx[i] = (i==t)?y[i]-1:y[i];
  }
}

//誤差逆伝播法(ReLU層)
void relu_bwd(int n, const float *x, const float *dy, float *dx){
  //dx={dy(x>0),0(x<=0)}
  for (int j=0; j<n; j++){
    dx[j] = (x[j]>0)?dy[j]:0;
  }
}

//誤差逆伝播法(FC層)
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

//6層NNによる誤差逆伝播法
void backward6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, unsigned char t, float *y, float *dA1, float *db1, float *dA2, float *db2, float *dA3, float *db3){
  //A.は重み行列，b.はバイアスベクトル，d..はそれぞれの偏微分
  float *y1_1 = malloc(sizeof(float)*N1);
  float *y2_1 = malloc(sizeof(float)*N1);
  float *y1_2 = malloc(sizeof(float)*N2);
  float *y2_2 = malloc(sizeof(float)*N2);
  float *y3 = malloc(sizeof(float)*N3);
  fc(N1, N0, x, A1, b1, y1_1);
  relu(N1, y1_1, y2_1);
  fc(N2, N1, y2_1, A2, b2, y1_2);
  relu(N2, y1_2, y2_2);
  fc(N3, N2, y2_2, A3, b3, y3);
  softmax(N3, y3, y);
  float *dx3 = malloc(sizeof(float)*N3);
  float *dx2 = malloc(sizeof(float)*N2);
  float *dx1 = malloc(sizeof(float)*N1);
  float *dx = malloc(sizeof(float)*N0);
  softmaxwithloss_bwd(N3, y, t, dx3);
  fc_bwd(N3, N2, y2_2, dx3, A3, dA3, db3, dx2);
  relu_bwd(N2, y2_2, dx2, dx2);
  fc_bwd(N2, N1, y2_1, dx2, A2, dA2, db2, dx1);
  relu_bwd(N1, y2_1, dx1, dx1);
  fc_bwd(N1, N0, x, dx1, A1, dA1, db1, dx);
}

// ランダムシャッフル
void shuffle(int n, int *x){
  //配列x[n]の要素をシャッフルする。
  for(int i=n-1; i>0; i--){
    int j = genrand_int32()%(i+1);
    int t = x[i];
    x[i] = x[j];
    x[j] = t;
  }
}

//交差エントロピー誤差
float cross_entropy_error(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, unsigned char t){
  //A.は重み行列，b.はバイアスベクトル，xは入力ベクトル，tは正解値
  //-log(y[t])を計算，yに微少量1e-7を加えることでオーバーフローを防ぐ
  float *y1e = malloc(sizeof(float)*N1);
  float *y2e = malloc(sizeof(float)*N2);
  float *y3e = malloc(sizeof(float)*N3);
  fc(N1, N0, x, A1, b1, y1e);
  relu(N1, y1e, y1e);
  fc(N2, N1, y1e, A2, b2, y2e);
  relu(N2, y2e, y2e);
  fc(N3, N2, y2e, A3, b3, y3e);
  softmax(N3, y3e, y3e);
  return -log(y3e[t]+1e-7);
}

//ファイルにセーブ
void save(const char * filename, int m, int n, const float * A, const float * b){
  //filenameは書き込むファイル，A[m*n],b[m]は配列
  FILE *fp;
  fp = fopen(filename, "w");
  for(int i=0; i<m*n; i++){
    fprintf(fp, "%f ", A[i]);
  }
  for(int j=0; j<m; j++){
    fprintf(fp, "%f ", b[j]);
  }
  fclose(fp);
  printf("File %s saved.\n", filename);
}

int main(int argc, char const *argv[]){
  //コマンドライン引数の数が4でないならエラー
  if(argc != 4){
    printf("Error!\nUsage: fc1.dat fc2.dat fc3.dat\n");
    exit(1);
  }
  init_genrand((unsigned)time(NULL));
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
  float *y = malloc(sizeof(float)*N3);
  float *dA1 = malloc(sizeof(float)*width*height*N1);
  float *db1 = malloc(sizeof(float)*N1);
  float *dA2 = malloc(sizeof(float)*N1*N2);
  float *db2 = malloc(sizeof(float)*N2);
  float *dA3 = malloc(sizeof(float)*N2*N3);
  float *db3 = malloc(sizeof(float)*N3);
  float *dA1_avg = malloc(sizeof(float)*width*height*N1);
  float *db1_avg = malloc(sizeof(float)*N1);
  float *dA2_avg = malloc(sizeof(float)*N1*N2);
  float *db2_avg = malloc(sizeof(float)*N2);
  float *dA3_avg = malloc(sizeof(float)*N2*N3);
  float *db3_avg = malloc(sizeof(float)*N3);
  //パラメータの初期値を，Heの初期値/一様分布乱数/標準偏差0.01の正規分布乱数 から選択
  printf("HeNormal[1] or Uniform[2] or SD=0.01[3] ?: ");
  int flag, r;
  while(1){
    r = scanf("%d", &flag);
    if((flag == 1) && (r == 1)){
      //Heの初期値で初期化
      he_init(width*height*N1, A1);
      he_init(N1, b1);
      he_init(N1*N2, A2);
      he_init(N2, b2);
      he_init(N2*N3, A3);
      he_init(N3, b3);
      break;
    }
    if((flag == 2) && (r == 1)){
      //一様分布乱数で初期化
      rand_init(width*height*N1, A1);
      rand_init(N1, b1);
      rand_init(N1*N2, A2);
      rand_init(N2, b2);
      rand_init(N2*N3, A3);
      rand_init(N3, b3);
      break;
    }
    if((flag == 3) && (r == 1)){
      //std=0.01の正規分布乱数で初期化
      norm_init(width*height*N1, A1);
      norm_init(N1, b1);
      norm_init(N1*N2, A2);
      norm_init(N2, b2);
      norm_init(N2*N3, A3);
      norm_init(N3, b3);
      break;
    }
    printf("Invalid input\n");
    scanf("%*s");
  }
  int *index = malloc(sizeof(int)*train_count);
  int *index_mini = malloc(sizeof(int)*BATCH);
  for(int i=0; i<train_count; i++){
    index[i] = i;
  }
  printf("Epoch|LossFanction|AccuracyRate|graph(AccuracyRate)\n");
  for(int j=0; j<EPOCH; j++){
    shuffle(train_count, index);
    for(int k=0; k<train_count/BATCH; k++){
      init(width*height*N1, 0, dA1_avg);
      init(N1, 0, db1_avg);
      init(N1*N2, 0, dA2_avg);
      init(N2, 0, db2_avg);
      init(N2*N3, 0, dA3_avg);
      init(N3, 0, db3_avg);
      for(int l=0; l<BATCH; l++){
        index_mini[l] = index[k*BATCH+l];
        backward6(A1, b1, A2, b2, A3, b3, train_x+width*height*index_mini[l], train_y[index_mini[l]], y, dA1, db1, dA2, db2, dA3, db3);
        add(width*height*N1, dA1, dA1_avg);
        add(N1, db1, db1_avg);
        add(N1*N2, dA2, dA2_avg);
        add(N2, db2, db2_avg);
        add(N2*N3, dA3, dA3_avg);
        add(N3, db3, db3_avg);
      }
      scale(width*height*N1, -1*ETA/BATCH, dA1_avg);
      scale(N1, -1*ETA/BATCH, db1_avg);
      scale(N1*N2, -1*ETA/BATCH, dA2_avg);
      scale(N2, -1*ETA/BATCH, db2_avg);
      scale(N2*N3, -1*ETA/BATCH, dA3_avg);
      scale(N3, -1*ETA/BATCH, db3_avg);
      add(width*height*N1, dA1_avg, A1);
      add(N1, db1_avg, b1);
      add(N1*N2, dA2_avg, A2);
      add(N2, db2_avg, b2);
      add(N2*N3, dA3_avg, A3);
      add(N3, db3_avg, b3);
    }
    float sum = 0;
    for(int m=0; m<test_count; m++){
      if(inference6(A1, b1, A2, b2, A3, b3, test_x+width*height*m) == test_y[m]){
        sum++;
      }
    }
    float sum_e = 0;
    for(int n=0; n<test_count; n++){
      sum_e += cross_entropy_error(A1, b1, A2, b2, A3, b3, test_x+width*height*n, test_y[n]);
    }
    printf("%5d|%12f|%11f%%|", j+1, sum_e/test_count, sum*100.0/test_count);
    for(int j=1; j<=sum*150.0/test_count; j++){
      printf("=");
    }
    printf("\n");
  }
  //コマンドライン引数の読み込み,ファイルへの書き込み
    save(argv[1], N1, N0, A1, b1);
    save(argv[2], N2, N1, A2, b2);
    save(argv[3], N3, N2, A3, b3);
  printf("All tasks has been completed.\n");
  return 0;
}
