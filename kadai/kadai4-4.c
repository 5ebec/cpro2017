#include <stdio.h>

#define M 3
#define N 4

int a[M][N];

int sub(int i, int j){
  a[i][j]=10*i+j;
  return 0;
}

int prints(int i, int j){
  printf("%2d ",a[i][j]);
  return 0;
}

int main(){
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      sub(i,j);
      prints(i,j);
    }
    putchar('\n');
  }
  return 0;
}
