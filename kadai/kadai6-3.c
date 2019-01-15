#include <stdio.h>

void add(int n, const float *x, float *o){
  for(int i=0; i<n; i++){
    o[i] += x[i];
  }
}

//Print_octave
void print_oct(int m, int n, const float *x, const char *name){
  printf("%s = [ ", name);
  for(int i=0; i<m*n; i++){
    printf("%.4f", x[i]);
    if((i+1)%n == 0){
      printf(" ;\n");
      if(i != m*n-1){
        printf(" ");
      }
    }else{
      printf("  ");
    }
  }
  printf("];\n");
}

int main(){
  float x[6] = {0, 1, 2, 3, 4, 5};
  float y[6] = {1, 1, 2, 3, 5, 8};
  print_oct(2, 3, x, "x");
  print_oct(2, 3, y, "y");
  add(6, x, y);
  print_oct(2, 3, y, "y");
  return 0;
}
