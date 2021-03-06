#include <stdio.h>

void init(int n, float x, float *o){
  for(int i=0; i<n; i++){
    o[i] = x;
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
  float y[6];
  print_oct(2, 3, y, "y");
  init(6, 7, y);
  print_oct(2, 3, y, "y");
  return 0;
}
