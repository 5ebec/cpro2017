#include <stdio.h>
int n;

int input(){
  float x;
  int b;
  do{
    printf("n = ");
    b = scanf("%f", &x);
    n = (int)x;
    //小数、文字列はエラー
    if(x != n || b != 1){
      puts("Invalid input");
      if(b != 1){
        scanf("%*s");
      }
    }
  }while(x != n || b != 1);
  return n;
}

int printbit(int n){
  printf("%d (10) = ", n);
  for(int i=31;i>=0;i--){
    printf("%d", n>>i & 1);
  }
  printf(" (2)\n");
  return 0;
}

int main(){
  printbit(input());
  return 0;
}
