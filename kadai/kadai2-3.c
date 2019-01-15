#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {

  double a,b,c,x1,x2;

  printf("a = ");
  scanf("%lf", &a);
  printf("b = ");
  scanf("%lf", &b);
  printf("c = ");
  scanf("%lf", &c);

  x1 = -b/(2*a);
  x2 = fabs(sqrt(fabs(b*b - 4*a*c))/(2*a));

  if(b*b - 4*a*c > 0){
    printf("%f\n%f\n" ,x1+x2 ,x1-x2);
  }else if(b*b - 4*a*c == 0){
    printf("%f\n(multiple root)\n" ,x1);
  }else if(b*b - 4*a*c < 0){
    printf("%f+%fi\n%f-%fi\n" ,x1,x2,x1,x2);
  }
  
  return 0;
}
