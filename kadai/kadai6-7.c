#include <stdio.h>

void mul(int m, int n, const float *x, const float *A, float *o){
//o=Ax
  for (int s=0; s<m; s++){
    o[s] = 0;
    for (int t=0; t<n; t++){
      o[s] += A[s*n+t]*x[t];
    }
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
  float A[6] = {1, 2, 3, 4, 5, 6};
  float x[3] = {2, 3, 5};
  float o[2];
  mul(2, 3, x, A, o);
  print_oct(2, 3, A, "A");
  print_oct(3, 1, x, "x");
  print_oct(2, 1, o, "o");
  return 0;
}
