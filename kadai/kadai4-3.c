#include <stdio.h>
// 20までの非負の整数に対応。
int n;

int input(){
  do{
    float x;
    printf("n = ");
    int b = scanf("%f", &x);
    n = (int)x;
    //負の数、20より大きな数、小数、文字列はエラー
    if(x < 0 || 20 < x || x != n || b != 1){
      puts("Invalid input");
      n = -1;
      if(b != 1){
        scanf("%*s");
      }
    }
  }while (n < 0);
  return n;
}

unsigned long long factorial(int s){
  int t;
  unsigned long long ans = 1;
  for(t=1;t<=s;t++){
    ans *= t;
  }
  return ans;
}

unsigned long long perm(int n, int r){
  return factorial(n)/factorial(n-r);
}

int main(){
  int i;
  input();
  for(i=0;i<=n;i++){
    printf("perm(%d,%d) = %llu\n",n,i,perm(n,i));
  }
  return 0;
}
