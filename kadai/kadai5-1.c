#include <stdio.h>
int n,r;

int input_n(){
  do{
    float x;
    printf("n = ");
    int b1 = scanf("%f", &x);
    n = (int)x;
    //負の数、小数、文字列はエラー
    if(x < 0 || x != n || b1 != 1){
      puts("Invalid input");
      n = -1;
      if(b1 != 1){
        scanf("%*s");
      }
    }
  }while (n < 0);
  return n;
}

int input_r(){
  do{
    float y;
    printf("r = ");
    int b2 = scanf("%f", &y);
    r = (int)y;
    //負の数、小数、文字列、nより大きい値はエラー
    if(y < 0 || y != r || b2 != 1 || y > n){
      puts("Invalid input");
      r = -1;
      if(b2 != 1){
        scanf("%*s");
      }
    }
  }while (r < 0);
  return r;
}

long long comb(int n, int r){
  if(r == 0 || r == n){
    return 1;
  }else if(r == 1){
    return n;
  }else{
    return comb(n-1, r-1) + comb(n-1, r);
  }
}

int main(void){
  printf("C(%d,%d) = %lld\n",input_n(),input_r(),comb(n,r));
  return 0;
}
