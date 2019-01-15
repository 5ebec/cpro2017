#include <stdio.h>

int main(){
  int x[5] = {4,1,8,2,9};
  int i;
  int range;

  for(i=0; i<5; i++){
    printf("x[%d]=%d\n",
           i, x[i]);
  }

  range=x[0];
  for(i=1;i<5;i++){
    range=(x[i]>range)?x[i]:range;
  }

  printf("max=%d\n", range);
  return 0;
}
