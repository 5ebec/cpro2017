#include <stdio.h>
#include <math.h>

int main() {
  double a,b,c;
  printf("a = ");
  scanf("%lf", &a);
  printf("b = ");
  scanf("%lf", &b);
  printf("c = ");
  scanf("%lf", &c);
  printf("%f\n%f\n" ,( -b + sqrt(b*b - 4*a*c) )/( 2*a ) ,( -b - sqrt(b*b - 4*a*c) )/( 2*a ));
  return 0;
}
