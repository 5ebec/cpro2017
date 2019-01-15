#include <stdio.h>
#include <math.h>

#define sq(n) n*n

typedef struct {
  double x;
  double y;
} Vector2d;

double GetLength(Vector2d vector){
  return sqrt(sq(vector.x)+sq(vector.y));
}

int main(void){
  Vector2d vec;
  printf("Input 2D Vector: ");
  scanf("%lf%lf",&vec.x,&vec.y);
  printf("Length: %lf\n",GetLength(vec));
  return 0;
}
