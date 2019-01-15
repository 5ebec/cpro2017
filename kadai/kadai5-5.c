#include <stdio.h>

int main(){
  float a = 0, b = 0;
  double c = 0, d = 0;
  for(int i=0;i<100000000;i++){
    a += 0.00000001;
    c += 0.00000001;
  }
  b = 0.00000001*100000000;
  d = 0.00000001*100000000;
  printf("a = %f\nb = %f\nc = %lf\nd = %lf\n", a,b,c,d);
  return 0;
}
