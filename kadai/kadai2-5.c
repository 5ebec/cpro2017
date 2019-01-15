#include <stdio.h>

int main(){
  int s;
  int t;

  for(s=0; s<=9; s++){
    for(t=s; t<=9; t++){
      printf("%d ", t);
    }
    printf("\n");
  }
  
  return 0;
}
