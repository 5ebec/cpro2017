#include <stdio.h>
#include <math.h>

#define sq(n) n*n
typedef struct {
  double x;
  double y;
} Vector2d;

void ScaleVector(Vector2d *v,double s){
  v->x = s*(v->x);
  v->y = s*(v->y);
}

double GetLength(Vector2d vector){
  return sqrt(sq(vector.x)+sq(vector.y));
}

int main(void){
  Vector2d vec;
  double n=0;
  printf("Input 2D Vector: ");
  scanf("%lf%lf",&vec.x,&vec.y);
  printf("Input scale value: ");
  scanf("%lf",&n);
  ScaleVector(&vec,n);
  printf("Result : %lf %lf\n",vec.x,vec.y);
  printf("Length : %lf\n",GetLength(vec));
  return 0;
}
