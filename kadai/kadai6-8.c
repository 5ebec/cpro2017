#include <stdio.h>

//Fully-Connected layer
void fc(int m, int n, const float *x, const float *A, const float *b, float *o){
  //y=Ax+b
  for (int s=0; s<m; s++){
    o[s] = 0;
    for (int t=0; t<n; t++){
      o[s] += A[s*n+t]*x[t];
    }
    o[s] += b[s];
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
  float b[2] = {0.5, 0.25};
  float x[3] = {2, 3, 5};
  float o[2];
  fc(2, 3, x, A, b, o);
  print_oct(2, 3, A, "A");
  print_oct(2, 1, b, "b");
  print_oct(3, 1, x, "x");
  print_oct(2, 1, o, "o");
  return 0;

}
